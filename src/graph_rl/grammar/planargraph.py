"""PlanarGraph class for generating and manipulating graphs."""

# Standard library
import time
from functools import partial
from pathlib import Path
from typing import Dict, Optional

# Third-party
import jax
import jax.numpy as jnp
from logging_mod.logger import get_logger
from matplotlib import pyplot as plt

# First-party
from graph_rl.ops.make_aabb import make_cp_rectangle_wrapper
from graph_rl.ops.polygon_area_perim import polygon_area_perimeter_from_boundary
from graph_rl.ops.sector_angle import update_all_sector_angles_vmap

from graph_rl.grammar.constants import Constants
from graph_rl.grammar.rule1 import rule_1
from graph_rl.grammar.rule2 import rule_2
from graph_rl.grammar.rule3 import rule_3
from graph_rl.grammar.rule4 import rule_4
from graph_rl.grammar.rule5 import rule_5 as rule_5_fn
from graph_rl.grammar.rule6 import rule_6 as rule_6_fn
from graph_rl.grammar.rule7 import rule_7 as rule_7_fn
from graph_rl.utils import PARAMS
from graph_rl.utils.adjacency_utils import (
    get_symmetric_adjacency,
    num_vertices,
)
from graph_rl.utils.plotting import plot_graph

logger = get_logger()


class PlanarGraph:
    """A class representing a planar graph, containing methods for generating and manipulating the
    graph."""

    def __init__(self, job_dir: Path = Path(""), max_cardinality: int = 50) -> None:
        # Meta level parameters
        self.job_dir = job_dir
        self.max_cardinality = max_cardinality
        self.has_aabb = False

        root_vertices = jnp.zeros((1,), dtype=jnp.int32)
        # Stores adjacency mask and dihedral paths (for generator)
        adjacency = jnp.zeros((max_cardinality, max_cardinality), dtype=jnp.uint8)
        adjacency_boundaries = jnp.zeros((max_cardinality, max_cardinality), dtype=jnp.uint8)

        # Vertex locations & rigid body modes
        vertex_positions = jnp.zeros((max_cardinality, 3))

        sector_angles = -1 * jnp.ones((max_cardinality, PARAMS.max_vertex_degree + 1))
        neighbors = -1 * jnp.ones((max_cardinality, PARAMS.max_vertex_degree + 1), dtype=int)
        ref_vector_idx = jnp.zeros(
            (max_cardinality, 2), dtype=int
        )  # Reference vector indices for each vertex, and if incoming angles span more than 180 degrees

        self.constants = Constants(
            root_vertices=root_vertices,
            adj=adjacency,
            adj_b=adjacency_boundaries,
            sector_angles=sector_angles,
            reference_neighbor=ref_vector_idx,
            neighbors=neighbors,
            vp=vertex_positions,
        )
        self.initial_edge()

    def initial_edge(self) -> None:
        """Initializes the graph with a single driving edge."""
        self.constants = self.constants.replace(
            # Direct the initial edge from vertex 0 → 1 so that vertex 1 is a boundary sink.
            adj=self.constants.adj.at[(0, 1)].set(1),
            vp=self.constants.vp.at[1].set([1, 0, 0]),
            root_vertices=self.constants.root_vertices.at[0].set(0),
        )


    def rule_1(
        self, params: Dict, idx: Optional[jnp.ndarray] = None, apply_rule=True
    ) -> tuple[Dict[str, jnp.ndarray], Optional[jnp.ndarray], Constants]:
        """Rule 1: Vertex expansion."""
        t1 = time.time()
        logger.debug(f"Rule 1: vertex {idx}")
        assert len(params["angles"]) == 4
        assert len(params["lens"]) == 3
        const_new, logs, _ = rule_1(self.constants, params, idx)
        if apply_rule:
            self.constants = const_new
        logger.debug(
            f"Rule 1: Expanding vertex {idx} Finished in %.4f seconds loss {logs['full']}." % (time.time() - t1)
        )
        if hasattr(rule_1, "_cache_size"):
            logger.debug("Current cache count of rule_1: %d" % (rule_1._cache_size()))  # mypy: ignore
        return logs, None, const_new

    def rule_2(
        self, params: Dict, idx: Optional[jnp.ndarray] = None, apply_rule=True
    ) -> tuple[Dict[str, jnp.ndarray], Optional[jnp.ndarray], Constants]:
        """Rule 1: Vertex expansion with merging of left and right edge."""
        t1 = time.time()
        logger.debug(f"Rule 2: vertex {idx}")
        assert len(params["angles"]) == 2
        assert len(params["lens"]) == 1
        const_new, logs, cb_idx = rule_2(self.constants, params, idx)
        if apply_rule:
            self.constants = const_new
        logger.debug(
            f"Rule 2: Expanding vertex {idx} Finished in %.4f seconds with loss {logs['full']}." % (time.time() - t1)
        )
        if hasattr(rule_2, "_cache_size"):
            logger.debug("Current cache count of rule_2: %d" % (rule_2._cache_size()))  # type: ignore
        return logs, cb_idx, const_new

    def rule_3(
        self, params: Dict, idx: Optional[jnp.ndarray] = None, apply_rule=True
    ) -> tuple[Dict[str, jnp.ndarray], Optional[jnp.ndarray], Constants]:
        """Rule 3: Vertex expansion with merging of left edge."""
        t1 = time.time()
        logger.debug(f"Rule 3: vertex {idx}")
        assert len(params["angles"]) == 3
        assert len(params["lens"]) == 2
        const_new, logs, cb_idx = rule_3(self.constants, params, idx)
        if apply_rule:
            self.constants = const_new
        logger.debug(
            f"Rule 3: Expanding vertex {idx} Finished in %.4f seconds with loss {logs['full']}." % (time.time() - t1)
        )
        if hasattr(rule_3, "_cache_size"):
            logger.debug("Current cache count of rule_3: %d" % (rule_3._cache_size()))
        return logs, cb_idx, const_new

    def rule_4(
        self, params: Dict, idx: Optional[jnp.ndarray] = None, apply_rule=True
    ) -> tuple[Dict[str, jnp.ndarray], Optional[jnp.ndarray], Constants]:
        """Rule 4: Vertex expansion with merging of right edge."""
        t1 = time.time()
        logger.debug(f"Rule 4: vertex {idx}")
        assert len(params["angles"]) == 3
        assert len(params["lens"]) == 2
        const_new, logs, cb_idx = rule_4(self.constants, params, idx)
        if apply_rule:
            self.constants = const_new
        logger.debug(
            f"Rule 4: Expanding vertex {idx} Finished in %.4f seconds with loss {logs['full']}." % (time.time() - t1)
        )
        if hasattr(rule_4, "_cache_size"):
            logger.debug("Current cache count of rule_4: %d" % (rule_4._cache_size()))
        return logs, cb_idx, const_new


    def rule_5(
        self, params: Dict, idx: Optional[jnp.ndarray] = None, apply_rule=True
    ) -> tuple[Dict[str, jnp.ndarray], Optional[jnp.ndarray], Constants]:
        """Rule 5: Single-vertex boundary sprout."""
        t1 = time.time()
        logger.debug(f"Rule 5: vertex {idx}")
        assert len(params["angles"]) == 3
        assert len(params["lens"]) == 2
        const_new, logs, _ = rule_5_fn(self.constants, params, idx)
        if apply_rule:
            self.constants = const_new
        logger.debug(
            f"Rule 5: Expanding vertex {idx} Finished in %.4f seconds loss {logs['full']}." % (time.time() - t1)
        )
        return logs, None, const_new

    def rule_6(
        self, params: Dict, idx: Optional[jnp.ndarray] = None, apply_rule=True
    ) -> tuple[Dict[str, jnp.ndarray], Optional[jnp.ndarray], Constants]:
        """Rule 6: Left merge with single-edge expansion."""
        t1 = time.time()
        logger.debug(f"Rule 6: vertex {idx}")
        assert len(params["angles"]) == 2
        assert len(params["lens"]) == 1
        const_new, logs, cb_idx = rule_6_fn(self.constants, params, idx)
        if apply_rule:
            self.constants = const_new
        logger.debug(
            f"Rule 6: Expanding vertex {idx} Finished in %.4f seconds with loss {logs['full']}." % (time.time() - t1)
        )
        if hasattr(rule_6_fn, "_cache_size"):
            logger.debug("Current cache count of rule_6: %d" % (rule_6_fn._cache_size()))
        return logs, cb_idx, const_new

    def rule_7(
        self, params: Dict, idx: Optional[jnp.ndarray] = None, apply_rule=True
    ) -> tuple[Dict[str, jnp.ndarray], Optional[jnp.ndarray], Constants]:
        """Rule 7: Right merge with single-edge expansion."""
        t1 = time.time()
        logger.debug(f"Rule 7: vertex {idx}")
        assert len(params["angles"]) == 2
        assert len(params["lens"]) == 1
        const_new, logs, cb_idx = rule_7_fn(self.constants, params, idx)
        if apply_rule:
            self.constants = const_new
        logger.debug(
            f"Rule 7: Expanding vertex {idx} Finished in %.4f seconds with loss {logs['full']}." % (time.time() - t1)
        )
        if hasattr(rule_7_fn, "_cache_size"):
            logger.debug("Current cache count of rule_7: %d" % (rule_7_fn._cache_size()))
        return logs, cb_idx, const_new



    def update_sector_angles(self):
        """Update sector angles for all vertices in the graph."""

        bool_mask = jnp.sum(get_symmetric_adjacency(self.constants.adj), axis=1) > 0
        indices = jnp.where(bool_mask)[0]  # Numpy faster here...
        if bool_mask.any():
            sectors, neighbors = update_all_sector_angles_vmap(
                self.constants.adj,
                self.constants.vp,
                self.constants.sector_angles,
                self.constants.neighbors,
                self.constants.reference_neighbor,
                indices,
            )
            self.constants = self.constants.replace(sector_angles=sectors, neighbors=neighbors)

    def reset_graph(self):
        """Reset the graph to an empty state."""
        self.constants = reset_graph(self.constants)

    @property
    def full_adjacency(self) -> jnp.ndarray:
        """Returns the full adjacency matrix including undirected boundary edges."""
        return self.constants.adj + self.constants.adj_b

    @property
    def num_vertices(self) -> jnp.ndarray:
        """Returns the number of vertices in the graph."""
        return num_vertices(self.constants.adj)

    def plot_planar_graph(
        self,
        show: bool = True,
        subdir: Optional[Path] = None,
        title: Optional[str] = None,
        full_adjacency: bool = True,
    ) -> None:
        """Plot the graph and save it in your job-dir.

        Params
        ------
        show: bool: Whether to open the plot or not.
        """
        if not jnp.isnan(self.constants.vp).any():
            adj = self.full_adjacency if full_adjacency else self.constants.adj
            if subdir is not None:
                (self.job_dir / subdir).mkdir(exist_ok=True, parents=True)
                title_ = f"{title}.png" if title is not None else "graph.png"
                save_path = self.job_dir / subdir / title_
            else:
                save_path = self.job_dir / "graph.png"
            fig, _ = plot_graph(adj, self.constants.vp, save_path=save_path, save_dpi=300, save_pdf=True)
            if show:
                plt.show()
            plt.close(fig)
        else:
            logger.warning("Cannot plot graph with NaN values in vertex positions.")

    def area_and_perimeter(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute total (unsigned) area and outer perimeter from the triangulated faces.

        Returns
        -------
        (area, perimeter) : Tuple[float, float]
          area      = sum of triangle areas in XY plane (unsigned)
          perimeter = length of boundary
        """
        # Ensure faces/triangulation exist
        return polygon_area_perimeter_from_boundary(self.constants.adj_b, self.constants.vp)

    def make_aabb(self, apply_to_cp=True) -> tuple[jnp.ndarray, jnp.ndarray, Constants, dict[str, jnp.ndarray]]:
        loss, len_diff, constants_new, info = make_cp_rectangle_wrapper(self.constants)
        if apply_to_cp:
            self.constants = constants_new
            self.has_aabb = True
        return loss, len_diff, constants_new, info


@partial(jax.jit, donate_argnums=(0,))
def reset_graph(constants: Constants) -> Constants:
    """Reset the graph to an empty state."""
    constants = constants.replace(
        adj=constants.adj * 0,
        adj_b=constants.adj_b * 0,
        sector_angles=constants.sector_angles * 0 - 1,
        reference_neighbor=constants.reference_neighbor * 0,
        neighbors=constants.neighbors * 0 - 1,
        vp=jnp.nan_to_num(constants.vp * 0),
    )
    constants = constants.replace(
        # Direct the initial edge from vertex 0 → 1 so vertex 1 remains a boundary sink.
        adj=constants.adj.at[(0, 1)].set(1),
        vp=constants.vp.at[1].set([1, 0, 0]),
        root_vertices=constants.root_vertices.at[0].set(0),
    )
    return constants
