# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

"""MuJoCo Simulation with direct actuator control."""

import numpy as np
from mujoco import mj_step
from omegaconf import DictConfig

from judo.simulation.base import Simulation


class MJSimulation(Simulation):
    """MuJoCo simulation with direct actuator control.

    Applies controls directly to MuJoCo actuators and steps
    the physics simulation forward.
    """

    def __init__(
        self,
        init_task: str = "spot_base",
        task_registration_cfg: DictConfig | None = None,
    ) -> None:
        """Initialize the MuJoCo simulation.

        Args:
            init_task: Name of the task to initialize.
            task_registration_cfg: Optional task registration configuration.
        """
        super().__init__(init_task=init_task, task_registration_cfg=task_registration_cfg)

    def step(self, command: np.ndarray) -> None:
        """Step the simulation forward.

        Args:
            command: Control command for this timestep.
        """
        if self.paused:
            return

        command = self.task.task_to_sim_ctrl(command)
        self.task.data.ctrl[:] = command[: self.task.model.nu]
        self.task.pre_sim_step()
        mj_step(self.task.sim_model, self.task.data)
        self.task.post_sim_step()
