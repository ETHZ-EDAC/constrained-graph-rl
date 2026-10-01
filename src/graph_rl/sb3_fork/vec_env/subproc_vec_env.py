# Standard library
import multiprocessing as mp
from copy import deepcopy
from typing import Any, Callable, Dict, Optional

# Third-party
import cloudpickle
import gymnasium as gym
import numpy as np
from torch_geometric.data import Data

# First-party
from graph_rl.sb3_fork.vec_env.base_vec_env import BaseVecEnv, VecEnvIndices, VecEnvObs, VecEnvStepReturn


class CloudpickleWrapper:
    """
    Uses cloudpickle to serialize the object for multiprocessing.
    """

    def __init__(self, var: Any):
        self.var = var

    def __getstate__(self) -> Any:
        return cloudpickle.dumps(self.var)

    def __setstate__(self, var: Any) -> None:
        self.var = cloudpickle.loads(var)

    def __call__(self) -> Any:
        return self.var()


def _worker(
    remote: mp.connection.Connection, parent_remote: mp.connection.Connection, env_fn_wrapper: CloudpickleWrapper
) -> None:
    parent_remote.close()
    env = env_fn_wrapper()
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                obs, reward, terminated, truncated, info = env.step(data)
                done = terminated or truncated
                reset_info: dict[str, Any] = {}
                if done:
                    info = dict(info)
                    info["terminal_observation"] = obs
                    obs, reset_info = env.reset()
                remote.send((obs, reward, done, info, reset_info))
            elif cmd == "reset":
                seed, options = data
                if seed is None:
                    obs, reset_info = env.reset()
                else:
                    obs, reset_info = env.reset(seed=seed, options=options)
                remote.send((obs, reset_info))
            elif cmd == "close":
                env.close()
                remote.close()
                break
            elif cmd == "get_spaces":
                remote.send((env.observation_space, env.action_space))
            elif cmd == "get_attr":
                attr_name = data
                if hasattr(env, "get_wrapper_attr"):
                    remote.send(env.get_wrapper_attr(attr_name))
                else:
                    remote.send(getattr(env, attr_name))
            elif cmd == "set_attr":
                attr_name, value = data
                setattr(env, attr_name, value)
                remote.send(None)
            elif cmd == "env_method":
                method_name, method_args, method_kwargs = data
                if hasattr(env, "get_wrapper_attr"):
                    method = env.get_wrapper_attr(method_name)
                else:
                    method = getattr(env, method_name)
                remote.send(method(*method_args, **method_kwargs))
            else:
                raise NotImplementedError(f"Unknown command {cmd}")
    except KeyboardInterrupt:
        pass


class SubprocVecEnv(BaseVecEnv):
    """
    Vectorized environment that runs each environment in its own subprocess.
    This is useful for computationally heavy environments.
    """

    actions: np.ndarray

    def __init__(self, env_fns: list[Callable[[], gym.Env]], start_method: Optional[str] = None):
        self.waiting = False
        self.closed = False

        if start_method is None:
            start_method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"

        print(start_method)
        self.ctx = mp.get_context(start_method)

        self.remotes, self.work_remotes = zip(*[self.ctx.Pipe() for _ in env_fns])
        self.processes = []
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            process = self.ctx.Process(target=_worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            process.daemon = True
            process.start()
            work_remote.close()
            self.processes.append(process)

        self.remotes[0].send(("get_spaces", None))
        observation_space, action_space = self.remotes[0].recv()
        super().__init__(len(env_fns), observation_space, action_space)

        self.buf_obs: Dict[int, Data] = {}
        self.buf_dones = np.zeros((self.num_envs,), dtype=bool)
        self.buf_rews = np.zeros((self.num_envs,), dtype=np.float32)
        self.buf_infos: list[dict[str, Any]] = [{} for _ in range(self.num_envs)]

        try:
            self.metadata = self.get_attr("metadata")[0]
        except Exception:
            self.metadata = {}

        self._options: list[dict[str, Any]] = [{} for _ in range(self.num_envs)]

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        self.waiting = True

    def step_wait(self) -> VecEnvStepReturn:
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False

        for env_idx, (obs, reward, done, info, reset_info) in enumerate(results):
            self.buf_rews[env_idx] = reward
            self.buf_dones[env_idx] = done
            self.buf_infos[env_idx] = info
            if done:
                self.reset_infos[env_idx] = reset_info
            self._save_obs(env_idx, obs)

        return self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones), deepcopy(self.buf_infos)

    def reset(self) -> VecEnvObs:
        for env_idx, remote in enumerate(self.remotes):
            remote.send(("reset", (self._seeds[env_idx], self._options[env_idx])))
        results = [remote.recv() for remote in self.remotes]
        for env_idx, (obs, reset_info) in enumerate(results):
            self.reset_infos[env_idx] = reset_info
            self._save_obs(env_idx, obs)
        self._reset_seeds()
        self._reset_options()
        return self._obs_from_buf()

    def close(self) -> None:
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(("close", None))
        for process in self.processes:
            process.join()
        self.closed = True

    def _save_obs(self, env_idx: int, obs: Data) -> None:
        self.buf_obs[env_idx] = obs

    def _obs_from_buf(self) -> VecEnvObs:
        return deepcopy(self.buf_obs)

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> list[Any]:
        indices = self._get_indices(indices)
        for env_idx in indices:
            self.remotes[env_idx].send(("get_attr", attr_name))
        return [self.remotes[env_idx].recv() for env_idx in indices]

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        indices = self._get_indices(indices)
        for env_idx in indices:
            self.remotes[env_idx].send(("set_attr", (attr_name, value)))
        for env_idx in indices:
            self.remotes[env_idx].recv()

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> list[Any]:
        indices = self._get_indices(indices)
        for env_idx in indices:
            self.remotes[env_idx].send(("env_method", (method_name, method_args, method_kwargs)))
        return [self.remotes[env_idx].recv() for env_idx in indices]

    def __del__(self) -> None:
        if not self.closed:
            self.close()
