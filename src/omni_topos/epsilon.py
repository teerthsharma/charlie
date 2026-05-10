"""EPSILON-CLI Engine — Stochastic Bang: noise-driven emergence, POVM state collapse."""

from __future__ import annotations

import numpy as np

from omni_topos._types import BettiSignature


class StochasticBang:
    """Big Bang as stochastic topological bifurcation.

    Models the origin of the universe as a noise-driven phase transition
    from the vacuum state (∅) into existence. Noise at the Planck scale
    (1.22e-32 GeV) causes spontaneous symmetry breaking, generating
    topological structure where none existed.

    This engine captures the key physical intuition:
    - Vacuum fluctuations are Poisson-distributed around the Planck density
    - POVM (Positive Operator-Valued Measure) collapse drives phase transitions
    - The Born rule governs the probability of outcome states

    Attributes:
        PLANCK_ENERGY_DENSITY: 1.22e-32 GeV/cm³ — critical energy density
            for vacuum decay in semiclassical gravity
        noise_amplitude: Override for the noise amplitude (defaults to Planck density)
    """

    PLANCK_ENERGY_DENSITY = 1.22e-32  # GeV

    def __init__(self, noise_amplitude: float | None = None) -> None:
        self.noise_amplitude = noise_amplitude or self.PLANCK_ENERGY_DENSITY

    def vacuum_fluctuation(self, rng: np.random.Generator) -> BettiSignature:
        """Sample a vacuum fluctuation from Planck-scale noise.

        Models the spontaneous emergence of topology from quantum foam.
        If the fluctuation is below threshold (0.01 Planck units), the
        vacuum remains stable (all zeros). Otherwise, Betti numbers
        are sampled proportional to the noise amplitude.

        Args:
            rng: Numpy random generator for reproducibility

        Returns:
            BettiSignature — either vacuum (b0=b1=b2=0) or a small
            topological fluctuation with b0 ≥ 1
        """
        normalized_noise = rng.poisson(self.noise_amplitude * 1e10) / 1e10
        if normalized_noise < 0.01:
            return BettiSignature.vacuum()
        b0 = max(1, int(normalized_noise * 10))
        b1 = max(0, int(normalized_noise * 3))
        b2 = max(0, int(normalized_noise * 1))
        return BettiSignature(b0=b0, b1=b1, b2=b2)

    def povm_collapse(
        self,
        state: BettiSignature,
        noise: float,
        rng: np.random.Generator,
    ) -> BettiSignature:
        """Apply POVM collapse to a BettiSignature.

        The POVM (Positive Operator-Valued Measure) models the effect
        of a measurement on the cosmological state. Three outcomes:

        0 — Vacuum collapse: reset to vacuum (ring the bell, start over)
        1 — Growth: add features proportional to noise amplitude
        2 — Decay: lose features (some topology collapses)

        Outcome probabilities are perturbed by Gaussian noise to model
        the fundamental indeterminacy of quantum measurement.

        Args:
            state: Current BettiSignature
            noise: Noise amplitude (scale factor for feature changes)
            rng: Numpy random generator

        Returns:
            New BettiSignature after POVM collapse
        """
        probs = np.array([0.3, 0.5, 0.2]) + rng.standard_normal(3) * 0.1
        probs = np.abs(probs)
        probs = probs / probs.sum()
        outcome = rng.choice(3, p=probs)
        if outcome == 0:
            return BettiSignature.vacuum()
        elif outcome == 1:
            new_b0 = state.b0 + max(0, int(noise * 5))
            new_b1 = state.b1 + max(0, int(noise * 2))
            return BettiSignature(b0=new_b0, b1=new_b1, b2=state.b2)
        else:
            new_b0 = max(1, state.b0 - max(0, int(noise * 3)))
            return BettiSignature(b0=new_b0, b1=state.b1, b2=state.b2)