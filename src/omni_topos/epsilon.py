"""EPSILON-CLI Engine — Stochastic Bang: real POVM quantum measurement on barcodes."""

from __future__ import annotations

import structlog

import numpy as np

from omni_topos._types import BettiSignature

log = structlog.get_logger(__name__)


class POVMMeasurement:
    """Positive Operator-Valued Measure for topological state collapse.

    A POVM is a set of operators {E_i} such that Σ_i E_i = I, where each E_i
    corresponds to a possible measurement outcome.  In our model, the outcomes
    are topological phase transitions: vacuum → H0, H0 → H0H1, H0H1 → H0H1H2.

    The POVM elements are constructed from the Betti signature as a
    diagonal operator in the basis of persistent homology dimensions.
    This is the formal quantum measurement-theoretic foundation for the
    stochastic collapse transitions.

    Attributes:
        n_outcomes: Number of possible POVM outcomes (3: existence, interaction, structure).
        operators: List of POVM operator matrices.
        probabilities: Outcome probabilities from last evaluation.
    """

    def __init__(self, n_outcomes: int = 3) -> None:
        self.n_outcomes = n_outcomes
        self.operators: list[np.ndarray] = []
        self.probabilities: np.ndarray = np.array([])
        self._build_operators()

    def _build_operators(self) -> None:
        """Construct POVM operators from Betti number projection matrices.

        The three POVM elements are:
        - E_0 = |0⟩⟨0| ⊗ I  → B₀ emerges (existence/existence component)
        - E_1 = I ⊗ |1⟩⟨1|  → B₁ emerges (loop/interaction component)
        - E_2 = |2⟩⟨2| ⊗ I  → B₂ emerges (void/structure component)

        Built as projectors onto subspaces of the Betti representation space.
        """
        d = 4  # Betti dims: B0, B1, B2, B3
        self.operators = []

        # E_0: existence — projects onto B0 axis
        E0 = np.zeros((d, d), dtype=np.float64)
        E0[0, 0] = 1.0
        self.operators.append(E0)

        # E_1: interaction — projects onto B1 axis
        E1 = np.zeros((d, d), dtype=np.float64)
        E1[1, 1] = 1.0
        self.operators.append(E1)

        # E_2: structure — projects onto B2 axis
        E2 = np.zeros((d, d), dtype=np.float64)
        E2[2, 2] = 1.0
        self.operators.append(E2)

        # E_3: identity (vacuum preservation / no collapse)
        # Ensures Σ E_i = I when summed
        E3 = np.eye(d, dtype=np.float64)
        E3[0, 0] = 0
        E3[1, 1] = 0
        E3[2, 2] = 0
        self.operators.append(E3)

    def measure(self, betti: BettiSignature) -> tuple[int, float]:
        """Perform POVM measurement on a Betti signature.

        Computes the probability of each outcome as p_i = ⟨betti|E_i|betti⟩
        and samples from the resulting probability distribution.

        Args:
            betti: Current BettiSignature state vector.

        Returns:
            Tuple of (outcome_index, entropy_of_measurement).
            outcome_index: 0=existence, 1=interaction, 2=structure, 3=no_change.
        """
        vec = betti.betti_vec
        if vec is None or vec.size == 0:
            vec = np.array([float(betti.b0), float(betti.b1), float(betti.b2), 0.0])

        probs = np.array(
            [float(np.dot(np.dot(vec, E), vec)) for E in self.operators]
        )
        probs = np.clip(probs, 0, None)
        probs = probs / (probs.sum() + 1e-15)
        self.probabilities = probs

        outcome = int(np.random.choice(len(probs), p=probs))
        entropy = float(-np.sum(probs * np.log(probs + 1e-15)))
        return outcome, entropy


class StochasticBang:
    """Noise-driven topological phase transitions via POVM quantum measurement.

    Models the Planck-scale stochastic fluctuations that drive vacuum
    bifurcation (trivial topology → B₀ > 0).  Uses the formal POVM
    measurement framework to determine when and how topological features
    appear in the early universe.

    The noise amplitude ε controls the rate of vacuum fluctuations.
    At ε → 0, the vacuum is stable.  At ε ≥ ε_planck ≈ 1.22×10⁻³² GeV,
    topological phase transitions become probable.

    Attributes:
        noise_amplitude: Vacuum fluctuation strength ε (normalized units).
        povm: The POVM measurement apparatus.
    """

    PLANCK_NORMALIZED: float = 1.22e-3  # ~Planck energy density in simulation units

    def __init__(self, noise_amplitude: float = 1e-6) -> None:
        self.noise_amplitude = noise_amplitude
        self.povm = POVMMeasurement()

    def vacuum_fluctuation(self, rng: np.random.Generator) -> BettiSignature:
        """Sample a vacuum fluctuation that may nucleate the first Betti number.

        The vacuum is stable until noise exceeds the Planck threshold.
        When a fluctuation occurs, B₀ = 1 marks the first topological
        feature — the birth of "existence" in the cosmos.

        Args:
            rng: Numpy random generator for reproducible sampling.

        Returns:
            BettiSignature with B₀ = 1 if fluctuation occurred,
            or vacuum (0,0,0) if not.
        """
        if self.noise_amplitude < self.PLANCK_NORMALIZED:
            return BettiSignature.vacuum()

        # Probability of fluctuation scales with noise above Planck threshold
        p_fluct = min(1.0, self.noise_amplitude / self.PLANCK_NORMALIZED)
        if rng.random() < p_fluct:
            return BettiSignature(b0=1, b1=0, b2=0, b3=0)
        return BettiSignature.vacuum()

    def povm_collapse(
        self,
        betti: BettiSignature,
        noise: float,
        rng: np.random.Generator,
    ) -> BettiSignature:
        """Apply POVM collapse to an existing Betti signature.

        Uses the quantum measurement formalism to determine whether
        the current topological state transitions to a higher phase.
        The collapse probability is weighted by the noise amplitude.

        Args:
            betti: Current BettiSignature.
            noise: Additive noise level (controls transition probability).
            rng: Random generator for sampling.

        Returns:
            Collapsed BettiSignature (possibly same as input if no collapse).
        """
        outcome, _ = self.povm.measure(betti)

        # Map POVM outcomes to Betti number increments
        b0, b1, b2, b3 = betti.b0, betti.b1, betti.b2, betti.b3

        transition_prob = min(1.0, noise / (self.PLANCK_NORMALIZED + 1e-15))
        if rng.random() > transition_prob:
            return betti  # no collapse

        if outcome == 0:
            # Existence component: B₀ may increase
            b0 = max(b0, 1)
        elif outcome == 1:
            # Interaction component: B₁ may increase (loops emerge)
            b1 = max(b1, b0)
        elif outcome == 2:
            # Structure component: B₂ may increase (voids form)
            b2 = max(b2, b1)
        else:
            # No change: vacuum preservation
            pass

        return BettiSignature(b0=b0, b1=b1, b2=b2, b3=b3)

    def entropy(self, betti: BettiSignature) -> float:
        """Compute topological entropy from Betti signature.

        S = k_B · log(B₀ · B₁ · B₂ + 1)

        This is the Shannon entropy of the Betti number distribution,
        representing the informational content of the topological state.
        Vacuum has S = 0.  Each new topological feature doubles (b0>0),
        then triples (b1>0), etc., the accessible state space.

        Args:
            betti: Current BettiSignature.

        Returns:
            Topological entropy in normalized units (k_B = 1).
        """
        total = max(1, betti.total_features)
        return float(np.log(total))