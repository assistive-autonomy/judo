# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

"""MuJoCo rollout backend for parallel trajectory simulation."""

import time
from copy import deepcopy

import numpy as np
from mujoco import MjData, MjModel
from mujoco.rollout import Rollout

from judo.utils.rollout_backend import RolloutBackend


def make_model_data_pairs(model: MjModel, num_pairs: int) -> tuple[list[MjModel], list[MjData]]:
    """Create model/data pairs for mujoco threaded rollout."""
    models = [deepcopy(model) for _ in range(num_pairs)]
    datas = [MjData(m) for m in models]
    return models, datas


class MJRolloutBackend(RolloutBackend):
    """Backend for conducting multithreaded rollouts using standard MuJoCo.

    Uses mujoco.rollout for direct physics simulation with controls.
    """

    def __init__(
        self,
        model: MjModel,
        num_threads: int,
    ) -> None:
        """Initialize the rollout backend.

        Args:
            model: MuJoCo model for the scene.
            num_threads: Number of parallel rollout threads.
        """
        self.num_threads = num_threads
        self.model = model

        self._models, self._datas = make_model_data_pairs(model, num_threads)
        self._rollout_obj = Rollout(nthread=num_threads)

    def rollout(
        self,
        x0: np.ndarray,
        controls: np.ndarray,
        last_policy_output: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Conduct parallel rollouts.

        Args:
            x0: Initial state, shape (nq+nv,). Will be tiled to num_threads internally.
            controls: Control inputs, shape (num_threads, num_timesteps, nu).
            last_policy_output: Unused. Accepted for interface compatibility.

        Returns:
            Tuple of:
                - states: Rolled out states, shape (num_threads, num_timesteps, nq+nv)
                - sensors: Sensor readings, shape (num_threads, num_timesteps, nsensor)
                - policy_outputs: Always None for this backend.
        """
        if x0.ndim == 1:
            x0 = np.tile(x0, (self.num_threads, 1))

        nq = self._models[0].nq
        nv = self._models[0].nv
        nu = self._models[0].nu

        # Prepend time to batched x0
        full_states = np.concatenate([time.time() * np.ones((len(self._models), 1)), x0], axis=-1)

        assert full_states.shape[-1] == nq + nv + 1
        assert full_states.ndim == 2
        assert controls.ndim == 3
        assert controls.shape[-1] == nu
        assert controls.shape[0] == full_states.shape[0]

        _states, _sensors = self._rollout_obj.rollout(self._models, self._datas, full_states, controls)

        out_states = np.array(_states)[..., 1:]  # Remove time from state
        out_sensors = np.array(_sensors)
        return out_states, out_sensors, None

    def update(self, num_threads: int) -> None:
        """Update the number of threads.

        Recreates internal state (model/data pairs) for new thread count.

        Args:
            num_threads: New number of parallel threads.
        """
        self.num_threads = num_threads
        self._rollout_obj.close()
        self._models, self._datas = make_model_data_pairs(self.model, num_threads)
        self._rollout_obj = Rollout(nthread=num_threads)
