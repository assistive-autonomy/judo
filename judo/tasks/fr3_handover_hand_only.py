# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

from dataclasses import dataclass, field
from typing import Any, Literal

import mujoco
import numpy as np

from judo import MODEL_PATH
from judo.gui import slider
from judo.tasks.base import Task, TaskConfig
from judo.utils.fields import np_1d_field
from judo.utils.math_utils import min_max_reg, quat_mul, rpy_to_quat

XML_PATH = str(MODEL_PATH / "xml/fr3_handover_hand_only.xml")
# For hand only
QPOS_HOME = np.array(
    [
        0.7, 0, 0.05, 1, 0, 0, 0,  # object pose
        0, 0, 0, 0, 0, 0,  # arm vel
        0.04, 0.04,  # gripper pos, equality constrained
    ]
)  # fmt: skip
CTRL_HOME = np.array([0, 0, 0, 0, 0, 0, 0.04])  # fmt: skip


@slider("w_pos", 0.0, 10.0, 0.01)
@slider("w_vel", 0.0, 10.0, 0.01)
@slider("w_ee_quat", 0.0, 10.0, 0.01)
@slider("w_gripper_pos", 0.0, 1.0, 0.001)
@slider("w_gripper_vel", 0.0, 1.0, 0.01)
@dataclass
class PrimitiveWeights:
    w_pos: float = 1.0
    w_vel: float = 0.0  # 2.0
    w_ee_quat: float = 4.0
    w_gripper_pos: float = 10.0
    w_gripper_vel: float = 0.0


@slider("w_upright", 0.0, 10.0, 0.01)
@slider("w_coll", 0.0, 10.0, 0.01)
@slider("w_qvel", 0.0, 10.0, 0.01)
@slider("w_open", 0.0, 10.0, 0.01)
@dataclass
class GlobalConfig:
    """Global reward configuration for the FR3 handover task."""

    w_upright: float = 0.25
    w_coll: float = 0.1
    w_qvel: float = 0.005
    w_open: float = 2.0


@slider("gripper_pos", 0.0, 0.04, 0.001)
@slider("gripper_vel", 0.0, 0.1, 0.001)
@dataclass
class FR3HandoverHandOnlyConfig(TaskConfig):
    # reward weights

    primitive_weights: PrimitiveWeights = field(default_factory=PrimitiveWeights)
    global_weights: GlobalConfig = field(default_factory=GlobalConfig)

    goal_pose: np.ndarray = np_1d_field(
        np.array([0.6, 0.4, 0.5, 0.0, 0.0, 0.0]),
        names=["x", "y", "z", "r", "p", "y"],
        mins=[0.4, -1.0, 0.01, -3.14, -3.14, -3.14],
        maxs=[1.0, 1.0, 1.0, 3.14, 3.14, 3.14],
        steps=[0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
        vis_name="human_hand_pose",
        xyz_vis_indices=[0, 1, 2],
        xyz_vis_defaults=[0.0, 0.0, 0.0],
        rpy_vis_indices=[3, 4, 5],
        rpy_vis_defaults=[0.0, 0.0, 0.0],
    )

    gripper_pos: float = 0.04  # gripper open position = 0.04, closed position = 0.0
    gripper_vel: float = 0.0  # gripper velocity


class FR3HandoverHandOnly(Task[FR3HandoverHandOnlyConfig]):
    """Defines the FR3 handover task (hand-only)."""

    name: str = "fr3_handover_hand_only"
    config_t: type[FR3HandoverHandOnlyConfig] = FR3HandoverHandOnlyConfig

    def __init__(self, model_path: str = XML_PATH, sim_model_path: str | None = None) -> None:
        """Initializes the LEAP cube rotation task."""
        super().__init__(model_path=model_path, sim_model_path=sim_model_path)
        self.qpos_home = QPOS_HOME.copy()
        self.reset_command = CTRL_HOME.copy()

        ## states
        # object pose and velocity
        self.obj_pos_adr = self.get_joint_position_start_index("object_joint")
        self.obj_pos_slice = slice(self.obj_pos_adr, self.obj_pos_adr + 3)
        self.obj_quat_adr = self.obj_pos_adr + 3
        self.obj_quat_slice = slice(self.obj_quat_adr, self.obj_quat_adr + 4)
        obj_vel_adr = self.get_joint_velocity_start_index("object_joint")
        self.obj_vel_slice = slice(obj_vel_adr, obj_vel_adr + 3)
        obj_angvel_adr = obj_vel_adr + 3
        self.obj_angvel_slice = slice(obj_angvel_adr, obj_angvel_adr + 3)

        # gripper position and velocity
        gripper_pos_adr = self.get_joint_position_start_index("finger_joint1")
        self.gripper_pos_slice = slice(gripper_pos_adr, gripper_pos_adr + 2)  # 2 dofs for the gripper
        gripper_vel_adr = self.get_joint_velocity_start_index("finger_joint1")
        self.gripper_vel_slice = slice(gripper_vel_adr, gripper_vel_adr + 2)  # 2 dofs for the gripper

        ## sensors
        # end-effector pose and velocity
        self.ee_pos_adr = self.get_sensor_start_index("ee_pos")
        self.ee_pos_slice = slice(self.ee_pos_adr, self.ee_pos_adr + 3)
        self.ee_quat_adr = self.get_sensor_start_index("ee_quat")
        self.ee_quat_slice = slice(self.ee_quat_adr, self.ee_quat_adr + 4)
        self.ee_linvel_adr = self.get_sensor_start_index("ee_linvel")
        self.ee_linvel_slice = slice(self.ee_linvel_adr, self.ee_linvel_adr + 3)
        self.ee_angvel_adr = self.get_sensor_start_index("ee_angvel")
        self.ee_angvel_slice = slice(self.ee_angvel_adr, self.ee_angvel_adr + 3)

        # distances
        self.left_finger_obj_adr = self.get_sensor_start_index("left_finger_obj")
        self.right_finger_obj_adr = self.get_sensor_start_index("right_finger_obj")
        self.left_finger_table_adr = self.get_sensor_start_index("left_finger_table")
        self.right_finger_table_adr = self.get_sensor_start_index("right_finger_table")
        self.obj_table_adr = self.get_sensor_start_index("obj_table")

        self.grasp_site_adr = self.get_sensor_start_index("trace_grasp_site")

        ## metadata that stores the current phase of the task
        self._data = mujoco.MjData(self.model)  # used for computing hypothetical sensor data

        self.reset()

    def in_goal_xy(self, curr_state: np.ndarray) -> np.ndarray:
        """Checks if the object is somewhere in the tube above the goal position of radius r.

        Args:
            curr_state: The current state value. Shape=(nq + nv,).
            config: The task configuration.

        Returns:
            in_goal: A bool indicating whether the object is in the goal region. Shape=(,).
        """
        obj_pos = curr_state[self.obj_pos_adr : self.obj_pos_adr + 3]  # (3,)
        dist = np.linalg.norm(obj_pos - self.config.goal_pose[:3])
        in_goal = dist <= self.config.goal_radius
        return in_goal

    def check_sensor_dists(
        self,
        sensors: np.ndarray,
        pair: Literal["left_finger_obj", "right_finger_obj", "left_finger_table", "right_finger_table", "obj_table"],
    ) -> np.ndarray:
        """Computes the distance between a specified pair of bodies.

        Args:
            sensors: The sensor values. Shape=(num_rollouts, T, total_sensor_dim).
            pair: The pair of bodies to check contact for. One of "left_finger_obj", "right_finger_obj", or "obj_table".

        Returns:
            dist: An array indicating the distance between the specified pair. Shape=(num_rollouts, T).
        """
        if pair == "left_finger_obj":
            i = self.left_finger_obj_adr
        elif pair == "right_finger_obj":
            i = self.right_finger_obj_adr
        elif pair == "left_finger_table":
            i = self.left_finger_table_adr
        elif pair == "right_finger_table":
            i = self.right_finger_table_adr
        elif pair == "obj_table":
            i = self.obj_table_adr
        else:
            raise ValueError(
                f"Invalid pair: {pair}. Must be one of 'left_finger_obj', 'right_finger_obj', or 'obj_table'."
            )
        dist = sensors[:, :, i]
        return dist

    # def pre_rollout(self, curr_state: np.ndarray) -> None:
    #     pass

    def reg_norm_cost(
        self, input: np.ndarray, target: np.ndarray, weight: float, min_val: float = -1.0, max_val: float = 1.0
    ) -> np.ndarray:
        return (
            np.linalg.norm(min_max_reg(input, min_val, max_val) - min_max_reg(target, min_val, max_val), axis=-1).sum(
                axis=-1
            )
            * weight
        )

    def steering_cost(self, states: np.ndarray, sensors: np.ndarray) -> np.ndarray:
        """Computes the LLM-generated steering cost for the FR3 handover task.

        Returns:
            total_cost: The total cost for the rollout. Shape=(num_rollouts,).
        """
        # Compute cost terms (sum over time per rollout). All shapes are (num_rollouts, T)
        ee_pos_cost = self.reg_norm_cost(
            sensors[..., self.ee_pos_slice], self.config.goal_pose[:3], self.config.primitive_weights.w_pos, -5.0, 5.0
        )

        ee_quat_err = np.abs(
            np.sum(
                (
                    sensors[..., self.ee_quat_slice]
                    * quat_mul(rpy_to_quat(self.config.goal_pose[3:]), np.array([0.0, -1.0, 0.0, 0.0]))
                ),
                axis=-1,
            )
        )
        ee_quat_cost = (2.0 * min_max_reg(np.arccos(np.clip(ee_quat_err, 0.0, 1.0)), 0.0, np.pi / 2.0)).sum(
            axis=-1
        ) * self.config.primitive_weights.w_ee_quat

        ee_linvel_cost = self.reg_norm_cost(
            sensors[..., self.ee_linvel_slice],
            np.zeros_like(sensors[..., self.ee_linvel_slice]),
            self.config.primitive_weights.w_vel,
            -1.0,
            1.0,
        )
        ee_angvel_cost = self.reg_norm_cost(
            sensors[..., self.ee_angvel_slice],
            np.zeros_like(sensors[..., self.ee_angvel_slice]),
            self.config.primitive_weights.w_vel,
            -1.0,
            1.0,
        )

        gripper_pos_cost = self.reg_norm_cost(
            states[..., self.gripper_pos_slice],
            self.config.gripper_pos,
            self.config.primitive_weights.w_gripper_pos,
            0.0,
            0.04,
        )
        gripper_vel_cost = self.reg_norm_cost(
            states[..., self.gripper_vel_slice],
            self.config.gripper_vel,
            self.config.primitive_weights.w_gripper_vel,
            -1.0,
            1.0,
        )
        # print(states[0, :20, self.gripper_pos_slice])

        total_cost = ee_pos_cost + ee_quat_cost + ee_linvel_cost + ee_angvel_cost + gripper_pos_cost + gripper_vel_cost
        return total_cost

    def global_cost(self, states: np.ndarray, sensors: np.ndarray) -> np.ndarray:
        """Computes the global cost for the FR3 handover task.

        Returns:
            total_cost: The total cost for the rollout. Shape=(num_rollouts,).
        """
        # TODO implement cost terms for
        # control effort and smoothness
        # joint (position, velocity and acceleration) limits
        # joint space singularities
        # collision avoidance (self and environment)

        # # querying states
        # obj_pos = states[..., self.obj_pos_slice]  # (num_rollouts, T, 3)
        # # arm_pos = states[..., self.arm_pos_slice]  # (num_rollouts, T, 8)
        # z_obj = states[..., self.obj_pos_adr + 2]  # (num_rollouts, T)
        # qvel = states[..., self.model.nq : self.model.nq + self.model.nv]  # (num_rollouts, T, nv)
        # qvel_norm = np.linalg.norm(qvel, axis=-1)  # (num_rollouts, T)
        # gripper_pos = arm_pos[..., -1]  # (num_rollouts, T)

        # # querying sensors
        # left_finger_table_dist = self.check_sensor_dists(sensors, "left_finger_table")  # noqa: F841
        # right_finger_table_dist = self.check_sensor_dists(sensors, "right_finger_table")  # noqa: F841
        # obj_table_dist = self.check_sensor_dists(sensors, "obj_table")  # noqa: F841

        # grasp_site_pos = sensors[..., self.grasp_site_adr : self.grasp_site_adr + 3]  # (num_rollouts, T, 3)
        # # ee_pos = sensors[..., self.ee_pos_slice]  # (num_rollouts, T, 3)
        # # ee_quat = sensors[..., self.ee_quat_slice]  # (num_rollouts, T, 4)
        # # ee_linvel = sensors[..., self.ee_linvel_slice]  # (num_rollouts, T, 3)
        # # ee_angvel = sensors[..., self.ee_angvel_slice]  # (num_rollouts, T, 3)

        # # distances and errors
        # # q_arm_goal = QPOS_HOME[self.arm_pos_slice]  # (9,)
        # grasp_dist = ((grasp_site_pos - obj_pos) ** 2).sum(-1)  # (num_rollouts, T)
        # pick_height_err = (z_obj - self.config.pick_height) ** 2  # (num_rollouts, T)
        # obj_goal_pos_dist = np.linalg.norm(obj_pos - self.config.goal_pose[:3], axis=-1)  # (num_rollouts, T)
        # # home_dist = np.linalg.norm(arm_pos - q_arm_goal, axis=-1)  # (num_rollouts, T)

        # # contact checks
        # left_finger_touching = left_finger_table_dist <= 0.0  # (num_rollouts, T)
        # right_finger_touching = right_finger_table_dist <= 0.0  # (num_rollouts, T)
        # hand_touching = left_finger_touching | right_finger_touching

        # w_upright = self.config.global_weights.w_upright
        # w_coll = self.config.global_weights.w_coll
        # w_qvel = self.config.global_weights.w_qvel
        # w_open = self.config.global_weights.w_open

        # rew_coll = (1 - hand_touching).sum(axis=-1)  # (num_rollouts,)
        # time_decay = np.linspace(1.0, 0.0, states.shape[1])  # decay the velocity penalty over time
        # rew_qvel = -(time_decay * qvel_norm).sum(axis=-1)
        # rew_open = -((gripper_pos - 0.04) ** 2).sum(axis=-1)  # encourage the gripper to be open

        # rewards = w_upright * rew_upright + w_coll * rew_coll + w_qvel * rew_qvel + w_open * rew_open

        return np.zeros(states.shape[0])

    def reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        system_metadata: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Wrapper function that uses the pre-implemented costs functions and cost functions generated by the LLM."""
        rewards = -self.steering_cost(states, sensors) - self.global_cost(states, sensors)
        # print(rewards.shape)

        return rewards

    def reset(self) -> None:
        """Resets the model to a default state with random goal."""
        self.data.qpos[:] = self.qpos_home
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = self.reset_command
        mujoco.mj_forward(self.model, self.data)
