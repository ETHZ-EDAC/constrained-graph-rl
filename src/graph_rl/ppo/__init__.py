"""PPO package with lazy exports to avoid import-time cycles."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["PlanarGraphEnv", "ObservationWrapper", "ActionWrapper"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        env_module = import_module("graph_rl.ppo.env")
        return getattr(env_module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
