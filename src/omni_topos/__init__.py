"""OmniTopos: Universal Topology Engine — Cosmological emergence via persistent homology."""

from __future__ import annotations

from omni_topos._types import (
    BettiSignature,
    GodTensorResult,
    TopologicalState,
    TopologyPhase,
)
from omni_topos.faraday import GodTensorEngine
from omni_topos.hamilton import HamiltonNBody
from omni_topos.phase_manager import PhaseManager
from omni_topos.simulation import BigBangSimulation

# Re-export simulation.main for backwards compatibility with old CLI entry-point
from omni_topos.simulation import main as main  # noqa: A001

__all__ = [
    "BigBangSimulation",
    "BettiSignature",
    "GodTensorEngine",
    "GodTensorResult",
    "HamiltonNBody",
    "PhaseManager",
    "TopologicalState",
    "TopologyPhase",
    "main",
]