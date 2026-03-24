# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

from judo.simulation.base import Simulation
from judo.simulation.mj_simulation import MJSimulation
from judo.simulation.policy_mj_simulation import PolicyMJSimulation


def _get_policy_mj_simulation() -> type[MJSimulation]:
    from judo.simulation.policy_mj_simulation import PolicyMJSimulation

    return PolicyMJSimulation


_simulation_registry = {
    "mujoco": lambda: MJSimulation,
    "mujoco_policy": _get_policy_mj_simulation,
}


def get_simulation_backend(simulation_backend: str) -> type:
    """Get the simulation class for a given backend."""
    if simulation_backend not in _simulation_registry:
        raise KeyError(f"Unknown simulation backend: {simulation_backend!r}")
    return _simulation_registry[simulation_backend]()


__all__ = [
    "Simulation",
    "MJSimulation",
    "PolicyMJSimulation",
    "get_simulation_backend",
]
