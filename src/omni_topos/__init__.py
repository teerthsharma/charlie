"""
OmniTopos: Universal Topology Engine
Simulating Cosmological Emergence via Persistent Homology Phase Space

Invented by Teerth Sharma | May 2026
github.com/teerthsharma/omni-topos

Six topology engines unified:
  FARADAY    → God Tensor (E↔H coupling, spectral fixed point)
  HAMLITON   → N-Body tensor (multi-body coupling, tensor product Hilbert space)
  LAMBDA-TOPO→ Topological memory (ripser barcodes, FAISS store)
  EPSILON-CLI→ Stochastic resonance (noise-driven emergence, POVM)
  AETHER-LINK→ Measurement apparatus (POVM state collapse)
  HOLLOW     → EM manifold (FDFD cavity, wave superposition, GC guards)
"""

from omni_topos._types import TopologyPhase, BettiSignature, TopologicalState, GodTensorResult
from omni_topos.faraday import GodTensorEngine
from omni_topos.hamilton import HamiltonNBody
from omni_topos.lambda_topo import TopologicalMemory
from omni_topos.epsilon import StochasticBang
from omni_topos.phase_manager import PhaseManager
from omni_topos.simulation import BigBangSimulation

__all__ = [
    "TopologyPhase",
    "BettiSignature",
    "TopologicalState",
    "GodTensorResult",
    "GodTensorEngine",
    "HamiltonNBody",
    "TopologicalMemory",
    "StochasticBang",
    "PhaseManager",
    "BigBangSimulation",
]