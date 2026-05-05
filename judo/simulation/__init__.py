# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

from judo.simulation.base import Simulation
from judo.simulation.hierarchical_mj_simulation import HierarchicalMJSimulation
from judo.simulation.mj_simulation import MJSimulation

DEFAULT_SIMULATION_BACKEND_REGISTRY: dict[str, type[Simulation]] = {
    "mujoco": MJSimulation,
    "mujoco_hierarchical": HierarchicalMJSimulation,
}


def get_simulation_backend(simulation_backend: str) -> type:
    """Get the simulation class for a given backend.

    Args:
        simulation_backend: Name of the simulation backend to get.

    Returns:
        The simulation class for the given backend.
    """
    if simulation_backend not in DEFAULT_SIMULATION_BACKEND_REGISTRY:
        raise KeyError(f"Unknown simulation backend: {simulation_backend!r}")
    return DEFAULT_SIMULATION_BACKEND_REGISTRY[simulation_backend]


__all__ = [
    "Simulation",
    "MJSimulation",
    "HierarchicalMJSimulation",
    "DEFAULT_SIMULATION_BACKEND_REGISTRY",
    "get_simulation_backend",
]
