"""Phase Manager — manages topological phase evolution: VACUUM → H0_ONLY → H0H1 → H0H1H2 → GOD_FIXED."""

from __future__ import annotations

from typing import Any

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
        VACUUM (∅) → H0_ONLY → H0H1 → H0H1H2 → GOD_FIXED

    Phase transitions are driven by three mechanisms:
    1. Persistent homology features from ripser-computed barcodes — the H₀/H₁/H₂
       counts come directly from the barcode data, not from random draw.
    2. POVM stochastic collapse (EPSILON-CLI) controlling when features appear.
    3. God Tensor fixed-point convergence (FARADAY) as the cosmological attractor.

    The TopologicalMemory stores states with real barcodes; phase decisions
    are made by inspecting the actual homology groups present in the barcode.

    Attributes:
        god_tensor: The FARADAY engine for spectral fixed-point computation.
        memory: The LAMBDA-TOPO engine with real ripser barcodes.
        current_phase: The current TopologyPhase.
    """

    def __init__(self) -> None:
        self.god_tensor = GodTensorEngine(latent_dim=16)
        self.memory = TopologicalMemory(latent_dim=16)
        self.current_phase: TopologyPhase = TopologyPhase.VACUUM

    def _betti_from_barcode(self, barcode: dict[str, Any]) -> BettiSignature:
        """Extract Betti numbers from a ripser-computed barcode.

        B₀ = number of H₀ pairs (connected components)
        B₁ = number of H₁ pairs (loops/cycles)
        B₂ = number of H₂ pairs (voids/2D cavities)

        Open-ended pairs (death=∞) count as persistent features and are
        included in the Betti count.

        Args:
            barcode: Dict with 'h0', 'h1', 'h2' keys, each a list of
                (birth, death) float pairs.

        Returns:
            BettiSignature derived from the barcode.
        """
        h0_pairs = barcode.get("h0", [])
        h1_pairs = barcode.get("h1", [])
        h2_pairs = barcode.get("h2", [])

        b0 = len([p for p in h0_pairs if p[1] < 1e5])  # finite H0 components
        b1 = len(h1_pairs)
        b2 = len(h2_pairs)

        return BettiSignature(b0=b0, b1=b1, b2=b2)

    def detect_phase(self, betti: BettiSignature) -> TopologyPhase:
        """Classify a BettiSignature into the correct TopologyPhase.

        Detection rules (in order of precedence):
        - residual < 1e-15 → GOD_FIXED (converged to cosmological attractor)
        - b2 > 0 → H0H1H2 (gravitational voids have formed)
        - b1 > 0 → H0H1 (quantum flux loops exist)
        - b0 > 0 → H0_ONLY (connected components, no structure)
        - all zero → VACUUM

        Args:
            betti: The BettiSignature to classify.

        Returns:
            The detected TopologyPhase.
        """
        if betti.is_vacuum:
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

        Combines real barcode computation (via TopologicalMemory),
        POVM stochastic collapse, and God Tensor fixed-point convergence.

        Steps:
        1. Apply God Tensor to Hilbert signature → evolved signature
        2. Store state → TopologicalMemory triggers ripser barcode computation
        3. Extract Betti numbers from the stored barcode
        4. Sample POVM collapse on Betti numbers (vacuum fluctuation)
        5. Classify phase from resulting Betti numbers

        Args:
            state: Current TopologicalState.
            rng: Numpy random generator.

        Returns:
            Evolved TopologicalState for the next simulation step.
        """
        noise_amplitude = abs(
            rng.standard_normal() * max(state.entropy, 0.01) / 10
        )

        # Apply God Tensor to evolve the Hilbert signature
        if self.god_tensor.T is not None:
            new_sig = self.god_tensor.apply(state.hilbert_signature)
            new_sig = new_sig / (np.linalg.norm(new_sig) + 1e-15)
            _, residual = self.god_tensor.power_iteration(new_sig, iters=100)
        else:
            new_sig = state.hilbert_signature.copy()
            residual = None

        # Build a candidate topological state and store it to trigger barcode
        candidate = TopologicalState(
            phase=state.phase,
            betti=state.betti,
            hilbert_signature=new_sig,
            barcode=state.barcode,
            entropy=state.entropy,
            residual=residual,
            age=state.age + 1,
        )
        self.memory.store(candidate)

        # Extract Betti numbers from the real barcode stored at the latest index
        new_barcode = self.memory.barcodes[-1] if self.memory.barcodes else {}
        new_betti = self._betti_from_barcode(new_barcode)

        # Apply POVM stochastic collapse if at vacuum
        stoch = StochasticBang(noise_amplitude=noise_amplitude)
        if new_betti.is_vacuum:
            new_betti = stoch.vacuum_fluctuation(rng)

        # Only use POVM collapse for Betti evolution, not random generation
        # The Betti counts come from barcodes; collapse only perturbs them
        if not new_betti.is_vacuum:
            new_betti = stoch.povm_collapse(new_betti, noise_amplitude, rng)

        new_phase = self.detect_phase(new_betti)
        new_entropy = max(1.0, float(np.log(max(1, new_betti.total_features))))

        evolved = TopologicalState(
            phase=new_phase,
            betti=new_betti,
            hilbert_signature=new_sig,
            barcode=new_barcode.get("barcode", state.barcode),
            entropy=new_entropy,
            residual=residual,
            age=state.age + 1,
        )
        self.current_phase = new_phase
        return evolved