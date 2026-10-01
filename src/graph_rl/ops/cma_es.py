# Standard library
from typing import Dict, Tuple, Callable
from functools import partial

# Third-party
import jax
from jax import numpy as jnp
import jax.nn as jnn
from evosax.algorithms.distribution_based.cma_es import CMA_ES, State
from logging_mod.logger import get_logger


from graph_rl.grammar.constants import Constants

logger = get_logger()


@partial(jax.jit, static_argnames=("num_steps", "population_size", "rule_fn"))
def apply_cmaes_to_rule(
    constants: Constants,
    params_init: Dict[str, jnp.ndarray],
    idx: jnp.ndarray,
    rule_fn: Callable,
    key: jnp.ndarray,
    num_steps: int = 4,
    population_size: int = 100,
) -> Tuple[Constants, Dict[str, jnp.ndarray], Dict[str, jnp.ndarray], jnp.ndarray]:
    """Optimize rule 1 parameters using evosax CMA-ES in a fully jittable loop.

    Params: 4 raw angles normalized to [0,1] + 3 edge lengths.
    Returns the best constants, losses, params, and loss trace.
    """

    def _flatten_params(params_: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        return jnp.concatenate([params_["angles"].reshape(-1), params_["lens"].reshape(-1)])

    def _unflatten_params(vec: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        angles = vec[: params_init["angles"].shape[0]]
        lens = vec[params_init["angles"].shape[0] :]
        return {"angles": angles, "lens": lens}

    def _to_unconstrained(x: jnp.ndarray) -> jnp.ndarray:
        # Transform [0,1] -> R via logit; clamp to avoid infinities.
        y = jnp.clip(x, 1e-6, 1 - 1e-6)
        return jnp.log(y) - jnp.log1p(-y)

    def _to_bounded(x: jnp.ndarray) -> jnp.ndarray:
        # Transform R -> [0,1].
        return jnn.sigmoid(x)

    def _loss_from_bounded(vec_bounded: jnp.ndarray) -> Tuple[jnp.ndarray, Tuple[Constants, Dict[str, jnp.ndarray]]]:
        params_dict = _unflatten_params(vec_bounded)
        constants_new, losses, _ = rule_fn(constants, params_dict, idx)
        # Encourage staying near the provided start while minimizing intersections.
        deviation = jnp.linalg.norm(params_dict["angles"] - params_init["angles"]) + jnp.linalg.norm(
            params_dict["lens"] - params_init["lens"]
        )
        loss = (losses["edge_intersection_geom"] + losses["boundary_edge_length_geom"]) * 10 + deviation
        return loss, (constants_new, losses)

    init_bounded = _flatten_params(params_init)
    init_unconstrained = _to_unconstrained(init_bounded)

    algo = CMA_ES(population_size=population_size, solution=jnp.zeros_like(init_unconstrained))
    algo_params = algo.default_params
    algo_params = algo.default_params.replace(std_init=0.2)

    key_init, key_scan = jax.random.split(key)
    state = algo.init(key_init, init_unconstrained, algo_params)

    def step(carry: Tuple[jnp.ndarray, State, jnp.ndarray, jnp.ndarray, Constants, Dict[str, jnp.ndarray]], _):
        key_step, state_step, best_loss, best_raw, best_consts, best_losses = carry
        key_ask, key_tell = jax.random.split(key_step)

        population_raw, state_step = algo.ask(key_ask, state_step, algo_params)
        population_bounded = jax.vmap(_to_bounded)(population_raw)

        losses, _ = jax.vmap(_loss_from_bounded, in_axes=0, out_axes=(0, 0))(population_bounded)

        state_step, _ = algo.tell(key_tell, population_raw, losses, state_step, algo_params)

        best_idx = jnp.argmin(losses)
        best_loss_step = losses[best_idx]
        is_better = best_loss_step < best_loss
        best_loss = jnp.where(is_better, best_loss_step, best_loss)
        candidate_raw = population_raw[best_idx]
        best_raw = jnp.where(is_better, candidate_raw, best_raw)

        def _update_best(_):
            best_bounded = _to_bounded(candidate_raw)
            consts_new, losses_new, _ = rule_fn(constants, _unflatten_params(best_bounded), idx)
            return candidate_raw, consts_new, losses_new

        def _keep_best(_):
            return best_raw, best_consts, best_losses

        best_raw, best_consts, best_losses = jax.lax.cond(is_better, _update_best, _keep_best, operand=None)

        return (
            key_tell,
            state_step,
            best_loss,
            best_raw,
            best_consts,
            best_losses,
        ), best_loss_step

    init_loss, (init_consts, init_losses_dict) = _loss_from_bounded(init_bounded)
    carry0 = (key_scan, state, init_loss, init_unconstrained, init_consts, init_losses_dict)

    (_, _, _, best_raw, best_consts, best_losses), loss_hist = jax.lax.scan(step, carry0, None, length=num_steps)

    best_params = _unflatten_params(_to_bounded(best_raw))
    return best_consts, best_losses, best_params, loss_hist
