# Standard library
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

# Third-party
import numpy as np
import torch as th
from omegaconf import DictConfig, OmegaConf


def set_random_seed(seed: int, using_cuda: bool = False, cuda_deterministic=False) -> None:
    """
    Seed the different random generators.

    :param seed:
    :param using_cuda:
    :param cuda_deterministic:
    """
    # Seed python RNG
    random.seed(seed)
    # Seed numpy RNG
    np.random.seed(seed)
    # seed the RNG for all devices (both CPU and CUDA)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.cuda.manual_seed_all(seed)

    if cuda_deterministic:
        th.use_deterministic_algorithms(True)
        if using_cuda:
            # Deterministic operations for CuDNN, it may impact performances
            th.backends.cudnn.deterministic = True
            th.backends.cudnn.benchmark = False


@dataclass(frozen=True)
class GrammarConstraints:
    """Container for constraints configuration parameters."""

    edge_length_min: float = 0.7
    edge_length_max: float = 2.5
    sector_eps: float = 10
    max_vertex_degree: int = 8
    boundary_edge_intersection: bool = True


def _default_config_path() -> Path:
    """Return the first found default config path."""
    repo_dir = Path(__file__).resolve().parents[3]
    package_dir = Path(__file__).resolve().parent
    candidates = (
        repo_dir / "conf" / "constraints" / "default.yaml",
        package_dir / "config.yml",
        package_dir / "config.yaml",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("No default constraints configuration file found.")


def load_config(path: Optional[Path] = None) -> GrammarConstraints:
    """Load constraints config from a YAML file and return as GrammarConstraints."""
    config_path = Path(path) if path else _default_config_path()
    cfg = OmegaConf.load(config_path)
    if isinstance(cfg, DictConfig):
        cfg = OmegaConf.to_object(cfg)
    return GrammarConstraints(**cfg)


PARAMS = load_config()


def set_grammar_constraints(cfg: GrammarConstraints | DictConfig | Mapping[str, object]) -> GrammarConstraints:
    """Set grammar constraints globally from a Hydra/runtime config.

    Several modules import ``PARAMS`` directly, so propagate the replacement to
    already-imported graph_rl modules as well. Call this before importing PPO,
    grammar, or ops modules when using a Hydra config from an old training run.
    """
    if isinstance(cfg, GrammarConstraints):
        params = cfg
    else:
        cfg_obj = OmegaConf.to_object(cfg) if isinstance(cfg, DictConfig) else dict(cfg)
        if not isinstance(cfg_obj, Mapping):
            raise TypeError(f"Expected constraints mapping, got {type(cfg_obj)!r}")
        params = GrammarConstraints(**cfg_obj)

    old_params = globals().get("PARAMS")
    globals()["PARAMS"] = params

    package = sys.modules.get("graph_rl.utils")
    if package is not None:
        setattr(package, "PARAMS", params)

    for module in list(sys.modules.values()):
        if module is None:
            continue
        module_name = getattr(module, "__name__", "")
        if not module_name.startswith("graph_rl."):
            continue
        module_params = getattr(module, "PARAMS", None)
        if module_params is old_params or isinstance(module_params, GrammarConstraints):
            setattr(module, "PARAMS", params)

    return params
