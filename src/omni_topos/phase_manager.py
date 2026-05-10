"""Phase Manager — manages topological phase evolution: VACUUM → H0_ONLY → H0H1 → H0H1H2 → GOD_FIXED."""

from __future__ import annotations

import structlog

import numpy as np

from omni_topos._types import BettiSignature, TopologicalState, TopologyPhase
from omni_topos.epsilon import StochasticBang
from omni_topos.faraday import GodTensorEngine
from omni_topos.lambda_topo import TopologicalMemory

log = structlog.get_logger(__name__)


class PhaseManager:
    """Manages the topological phase evolution of the cosmological simulation.

    The phase lifecycle follows a strict hierarchy:
        VACUUM (∅) → H0_ONLY (existence) → H0H1 (quantum flux)
                    → H0H1H2 (gravitational voids) → GOD_FIXED (attractor)

    Each phase transition is driven by a combination of:
    - StochasticBang (vacuum fluctuations, POVM collapse)
    - GodTensorEngine (spectral fixed-point convergence)
    - TopologicalMemory (phase history tracking and similarity retrieval)

    Attributes:
        god_tensor: The FARADAY engine for fixed-point computation
        memory: The LAMBDA-TOPO engine for state storage
        current_phase: The current TopologyPhase
    """

    def __init__(self) -> None:
        self.god_tensor = GodTensorEngine(latent_dim=16)
        self.memory = TopologicalMemory(latent_dim=16)
        self.current_phase: TopologyPhase = TopologyPhase.VACUUM

    def detect_phase(self, betti: BettiSignature) -> TopologyPhase:
        """Classify a BettiSignature into the correct TopologyPhase.

        Detection rules (in order of precedence):
        - b2 > 0 → H0H1H2 (gravitational voids have formed)
        - b1 > 0 → H0H1 (loops/wave paths exist)
        - b0 > 0 → H0_ONLY (existence, but no structure)
        - all zero → VACUUM

        Args:
            betti: The BettiSignature to classify

        Returns:
            The detected TopologyPhase
        """
        total = betti.total_features
        if total == 0:
            return TopologyPhase.VACUUM
        elif betti.b2 > 0:
            return TopologyPhase.H0H1H2
        elif betti.b1 > 0:
            return TopologyPhase.H0H1
        else:
            return TopologyPhase.H0_ONLY

    def evolve_phase(
        self,
        state: TopologicalState,
        rng: np.random.Generator,
    ) -> TopologicalState:
        """Advance the cosmological state by one simulation step.

        Combines stochastic POVM collapse, God Tensor fixed-point
        application, and phase detection to evolve the state.

        Steps:
        1. Sample POVM collapse from current betti signature
        2. If vacuum, sample vacuum fluctuation instead
        3. Detect resulting phase
        4. Apply God Tensor to Hilbert signature (if initialized)
        5. Compute residual and entropy
        6. Store in topological memory

        Args:
            state: Current TopologicalState
            rng: Numpy random generator

        Returns:
            Evolved TopologicalState for the next simulation step
        """
        noise = abs(rng.standard_normal() * max(state.entropy, 0.01) / 10)
        stoch = StochasticBang(noise_amplitude=noise)
        new_betti = stoch.povm_collapse(state.betti, noise, rng)
        if new_betti.is_vacuum:
            new_betti = stoch.vacuum_fluctuation(rng)
        new_phase = self.detect_phase(new_betti)
        if self.god_tensor.T is not None:
            new_sig = self.god_tensor.apply(state.hilbert_signature)
            new_sig = new_sig / np.linalg.norm(new_sig)
            _, residual = self.god_tensor.power_iteration(new_sig, iters=100)
        else:
            new_sig = state.hilbert_signature
            residual = None
        new_entropy = max(1, new_betti.total_features)
        evolved = TopologicalState(
            phase=new_phase,
            betti=new_betti,
            hilbert_signature=new_sig,
            barcode=state.barcode,
            entropy=float(np.log(new_entropy)),
            residual=residual,
            age=state.age + 1,
        )
        self.memory.store(evolved)
        self.current_phase = new_phase
        return evolved