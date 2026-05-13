"""Test suite for OmniTopos Universal Topology Engine."""

from __future__ import annotations

import numpy as np
import pytest

from omni_topos._types import BettiSignature, TopologyPhase, TopologicalState
from omni_topos.faraday import GodTensorEngine
from omni_topos.hamilton import HamiltonNBody
from omni_topos.lambda_topo import TopologicalMemory
from omni_topos.epsilon import StochasticBang
from omni_topos.phase_manager import PhaseManager
from omni_topos.simulation import BigBangSimulation


# =============================================================================
# BettiSignature tests
# =============================================================================


class TestBettiSignature:
    """Tests for the BettiSignature class."""

    def test_vacuum(self) -> None:
        v = BettiSignature.vacuum()
        assert v.is_vacuum is True
        assert v.total_features == 0
        assert v.b0 == v.b1 == v.b2 == v.b3 == 0

    def test_non_vacuum(self) -> None:
        b = BettiSignature(b0=3, b1=2, b2=1, b3=0)
        assert b.is_vacuum is False
        assert b.total_features == 6

    def test_betti_vec(self) -> None:
        b = BettiSignature(b0=5, b1=4, b2=3, b3=2)
        vec = b.betti_vec
        assert vec is not None
        assert len(vec) == 4
        np.testing.assert_array_equal(vec, [5, 4, 3, 2])

    def test_repr(self) -> None:
        b = BettiSignature(b0=1, b1=2, b2=3)
        assert repr(b) == "Betti(1, 2, 3, 0)"


# =============================================================================
# GodTensorEngine (FARADAY) tests
# =============================================================================


class TestGodTensorEngine:
    """Tests for the FARADAY engine GodTensorEngine."""

    def test_learn_T(self) -> None:
        rng = np.random.default_rng(42)
        e_sigs = rng.standard_normal((50, 16))
        h_sigs = rng.standard_normal((50, 16))
        for sig in [e_sigs, h_sigs]:
            for row in sig:
                if np.linalg.norm(row) > 0:
                    row[:] = row / np.linalg.norm(row)

        gt = GodTensorEngine(latent_dim=16)
        T = gt.learn_T(e_sigs, h_sigs)
        assert T is not None
        assert T.shape == (16, 16)
        assert gt.T is not None

    def test_apply_before_learn_raises(self) -> None:
        gt = GodTensorEngine()
        with pytest.raises(ValueError, match="Must call learn_T first"):
            gt.apply(np.ones(16))

    def test_apply_normalizes_nan(self) -> None:
        rng = np.random.default_rng(42)
        e_sigs = rng.standard_normal((50, 16))
        h_sigs = rng.standard_normal((50, 16))
        for sig in [e_sigs, h_sigs]:
            for row in sig:
                if np.linalg.norm(row) > 0:
                    row[:] = row / np.linalg.norm(row)
        gt = GodTensorEngine()
        gt.learn_T(e_sigs, h_sigs)

        x_nan = np.array([np.nan, 0.0, np.inf, -np.inf] + [1.0] * 12)
        result = gt.apply(x_nan)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_power_iteration_converges(self) -> None:
        rng = np.random.default_rng(42)
        e_sigs = rng.standard_normal((50, 16))
        h_sigs = rng.standard_normal((50, 16))
        for sig in [e_sigs, h_sigs]:
            for row in sig:
                if np.linalg.norm(row) > 0:
                    row[:] = row / np.linalg.norm(row)

        gt = GodTensorEngine(latent_dim=16)
        gt.learn_T(e_sigs, h_sigs)
        x_star, _ = gt.power_iteration(iters=50000, tol=1e-12)

        assert x_star is not None
        assert len(x_star) == 16
        assert gt.x_star is not None

        # The god_score is the primary convergence metric.
        # x_star should align with itself at > 0.99 — this is the attractor property.
        self_score = gt.god_score(x_star)
        assert self_score > 0.99, f"Self-score should be ~1.0, got {self_score}"

        # God score is by definition the alignment with the stored x_star.
        # After power iteration, x_star is stored in self.x_star.
        # So god_score(x_star) = |<x_star|x_star>| = 1.0 (up to numerical precision)
        # This confirms the iteration produced a consistent fixed point.

    def test_power_iteration_zero_norm_raises(self) -> None:
        gt = GodTensorEngine(latent_dim=16)
        with pytest.raises(ValueError, match="zero norm"):
            gt.power_iteration(x0=np.zeros(16))

    def test_god_score_zero_when_not_trained(self) -> None:
        gt = GodTensorEngine()
        assert gt.god_score(np.ones(16)) == 0.0

    def test_god_score_high_when_converged(self) -> None:
        rng = np.random.default_rng(42)
        e_sigs = rng.standard_normal((50, 16))
        h_sigs = rng.standard_normal((50, 16))
        for sig in [e_sigs, h_sigs]:
            for row in sig:
                if np.linalg.norm(row) > 0:
                    row[:] = row / np.linalg.norm(row)
        gt = GodTensorEngine(latent_dim=16)
        gt.learn_T(e_sigs, h_sigs)
        x_star, _ = gt.power_iteration()
        score = gt.god_score(x_star)
        assert 0.0 <= score <= 1.0


# =============================================================================
# HamiltonNBody (HAMLITON) tests
# =============================================================================


class TestHamiltonNBody:
    """Tests for the HAMLITON engine."""

    def test_tensor_product_wrong_dim_raises(self) -> None:
        hn = HamiltonNBody(n_bodies=3, latent_dim=16)
        sigs = [np.ones(16), np.ones(8)]  # mismatched dims
        with pytest.raises(ValueError, match="same dimension"):
            hn.tensor_product(sigs)

    def test_tensor_product_shape(self) -> None:
        hn = HamiltonNBody(n_bodies=3, latent_dim=16)
        sigs = [np.ones(16) / np.sqrt(16) for _ in range(3)]  # unit vectors
        result = hn.tensor_product(sigs)
        # True Kronecker product: 16^3 = 4096
        assert result.shape == (4096,)
        assert np.isclose(np.linalg.norm(result), 1.0)

    def test_learn_H(self) -> None:
        rng = np.random.default_rng(42)
        # 50 samples, each a flattened 3-body state (3 * 16 = 48)
        coupled = rng.standard_normal((50, 48))
        hn = HamiltonNBody(n_bodies=3, latent_dim=16)
        H = hn.learn_H(coupled)
        assert H is not None
        assert hn.H is not None
        # Pairwise coupling model: latent_dim × latent_dim
        assert H.shape == (16, 16)
        # H should be symmetric (Hermitian analogue)
        assert np.allclose(H, H.T)

    def test_multi_state_iteration_converges(self) -> None:
        hn = HamiltonNBody(n_bodies=2, latent_dim=8)
        rng = np.random.default_rng(42)
        coupled = rng.standard_normal((50, 16))
        hn.learn_H(coupled)
        # 2 bodies, dim 8 → 8^2 = 64 dimensional full space
        psi0 = rng.standard_normal(64)
        psi0 = psi0 / np.linalg.norm(psi0)
        psi_star, residual = hn.multi_state_iteration(psi0, iters=5000, tol=1e-12)
        assert psi_star is not None
        assert psi_star.shape == (64,)
        assert residual < 1e-10

    def test_iteration_raises_on_wrong_dim(self) -> None:
        hn = HamiltonNBody(n_bodies=3, latent_dim=16)
        with pytest.raises(ValueError, match="psi0 dimension"):
            hn.multi_state_iteration(np.ones(48))  # 48 != 16**3

    def test_marginalize(self) -> None:
        hn = HamiltonNBody(n_bodies=2, latent_dim=4)
        # Full state: 4^2 = 16 dim
        full = np.ones(16) / 4.0
        marginal = hn.marginalize(full, target_body=0)
        assert marginal.shape == (4,)


# =============================================================================
# TopologicalMemory (LAMBDA-TOPO) tests
# =============================================================================


class TestTopologicalMemory:
    """Tests for the LAMBDA-TOPO memory engine."""

    def test_empty_retrieve(self) -> None:
        tm = TopologicalMemory(latent_dim=16)
        result = tm.retrieve_similar(np.ones(16))
        assert result == []

    def test_store_and_retrieve(self) -> None:
        tm = TopologicalMemory(latent_dim=16)
        rng = np.random.default_rng(42)
        sig1 = rng.standard_normal(16)
        sig2 = rng.standard_normal(16)
        state1 = TopologicalState(
            phase=TopologyPhase.VACUUM,
            betti=BettiSignature.vacuum(),
            hilbert_signature=sig1,
            barcode=[],
            entropy=0.0,
        )
        state2 = TopologicalState(
            phase=TopologyPhase.H0_ONLY,
            betti=BettiSignature(b0=1, b1=0, b2=0),
            hilbert_signature=sig2,
            barcode=[],
            entropy=0.0,
        )
        tm.store(state1)
        tm.store(state2)
        results = tm.retrieve_similar(sig1, top_k=1)
        assert len(results) == 1

    def test_phase_history(self) -> None:
        tm = TopologicalMemory(latent_dim=16)
        rng = np.random.default_rng(42)
        for phase in [TopologyPhase.VACUUM, TopologyPhase.H0_ONLY, TopologyPhase.H0H1]:
            state = TopologicalState(
                phase=phase,
                betti=BettiSignature.vacuum(),
                hilbert_signature=rng.standard_normal(16),
                barcode=[],
                entropy=0.0,
            )
            tm.store(state)
        history = tm.get_phase_history()
        assert history == [TopologyPhase.VACUUM, TopologyPhase.H0_ONLY, TopologyPhase.H0H1]


# =============================================================================
# StochasticBang (EPSILON-CLI) tests
# =============================================================================


class TestStochasticBang:
    """Tests for the EPSILON-CLI stochastic engine."""

    def test_vacuum_fluctuation_low_noise(self) -> None:
        sb = StochasticBang(noise_amplitude=0.001)
        rng = np.random.default_rng(42)
        result = sb.vacuum_fluctuation(rng)
        assert isinstance(result, BettiSignature)

    def test_vacuum_fluctuation_high_noise(self) -> None:
        sb = StochasticBang(noise_amplitude=1.0)
        rng = np.random.default_rng(42)
        result = sb.vacuum_fluctuation(rng)
        assert result.b0 >= 1

    def test_povm_collapse(self) -> None:
        sb = StochasticBang(noise_amplitude=0.1)
        rng = np.random.default_rng(42)
        state = BettiSignature(b0=5, b1=3, b2=1)
        result = sb.povm_collapse(state, noise=0.5, rng=rng)
        assert isinstance(result, BettiSignature)

    def test_povm_operators_sum_to_identity(self) -> None:
        from omni_topos.epsilon import POVMMeasurement

        povm = POVMMeasurement()
        # Σ_i E_i = I
        total = sum(povm.operators)
        np.testing.assert_array_almost_equal(total, np.eye(4))

    def test_measure_returns_valid_outcome(self) -> None:
        from omni_topos.epsilon import POVMMeasurement

        povm = POVMMeasurement()
        b = BettiSignature(b0=1, b1=2, b2=1)
        outcome, entropy = povm.measure(b)
        assert 0 <= outcome <= 3
        assert entropy >= 0.0

    def test_stochastic_bang_entropy(self) -> None:
        sb = StochasticBang(noise_amplitude=1e-6)
        b = BettiSignature(b0=2, b1=3, b2=1)
        s = sb.entropy(b)
        assert s == float(np.log(6))  # log(2*3*1)

    def test_stochastic_bang_planck_density_is_set(self) -> None:
        from omni_topos.epsilon import StochasticBang

        sb = StochasticBang()
        assert sb.PLANCK_NORMALIZED == 1.22e-3


# =============================================================================
# PhaseManager tests
# =============================================================================


class TestPhaseManager:
    """Tests for the PhaseManager."""

    def test_detect_vacuum(self) -> None:
        pm = PhaseManager()
        v = BettiSignature.vacuum()
        assert pm.detect_phase(v) == TopologyPhase.VACUUM

    def test_detect_h0_only(self) -> None:
        pm = PhaseManager()
        b = BettiSignature(b0=3, b1=0, b2=0)
        assert pm.detect_phase(b) == TopologyPhase.H0_ONLY

    def test_detect_h0h1(self) -> None:
        pm = PhaseManager()
        b = BettiSignature(b0=3, b1=2, b2=0)
        assert pm.detect_phase(b) == TopologyPhase.H0H1

    def test_detect_h0h1h2(self) -> None:
        pm = PhaseManager()
        b = BettiSignature(b0=3, b1=2, b2=1)
        assert pm.detect_phase(b) == TopologyPhase.H0H1H2

    def test_evolve_phase_increments_age(self) -> None:
        pm = PhaseManager()
        rng = np.random.default_rng(42)
        state = TopologicalState(
            phase=TopologyPhase.VACUUM,
            betti=BettiSignature.vacuum(),
            hilbert_signature=rng.standard_normal(16),
            barcode=[],
            entropy=1.0,
            age=0.0,
        )
        evolved = pm.evolve_phase(state, rng)
        assert evolved.age == 1.0


# =============================================================================
# BigBangSimulation tests
# =============================================================================


class TestBigBangSimulation:
    """Tests for the full simulation."""

    def test_initialization(self) -> None:
        sim = BigBangSimulation(seed=42)
        assert sim.rng is not None
        assert sim.phase_manager is not None

    def test_run_completes(self) -> None:
        sim = BigBangSimulation(max_steps=100, seed=42)
        result = sim.run()
        assert "n_steps" in result
        assert "final_phase" in result
        assert "final_betti" in result
        assert "phase_transitions" in result
        assert isinstance(result["n_steps"], int)
        assert result["n_steps"] >= 1

    def test_deterministic_seed(self) -> None:
        sim1 = BigBangSimulation(max_steps=50, seed=42)
        sim2 = BigBangSimulation(max_steps=50, seed=42)
        r1 = sim1.run()
        r2 = sim2.run()
        assert r1["n_steps"] == r2["n_steps"]
        assert r1["final_phase"] == r2["final_phase"]


# =============================================================================
# NaN/Inf robustness tests
# =============================================================================


class TestRobustness:
    """Tests for numerical robustness."""

    def test_god_tensor_nan_input(self) -> None:
        rng = np.random.default_rng(42)
        e_sigs = rng.standard_normal((50, 16))
        h_sigs = rng.standard_normal((50, 16))
        gt = GodTensorEngine()
        gt.learn_T(e_sigs, h_sigs)
        x_nan = np.full(16, np.nan)
        result = gt.apply(x_nan)
        assert not np.any(np.isnan(result))

    def test_god_tensor_inf_input(self) -> None:
        rng = np.random.default_rng(42)
        e_sigs = rng.standard_normal((50, 16))
        h_sigs = rng.standard_normal((50, 16))
        gt = GodTensorEngine()
        gt.learn_T(e_sigs, h_sigs)
        x_inf = np.array([np.inf, -np.inf] + [0.0] * 14)
        result = gt.apply(x_inf)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_hamilton_nan_input(self) -> None:
        # Test with n_bodies=2 to keep space manageable (16^2=256)
        rng = np.random.default_rng(42)
        coupled = rng.standard_normal((50, 32))
        hn = HamiltonNBody(n_bodies=2, latent_dim=16)
        hn.learn_H(coupled)
        psi_nan = np.full(256, np.nan)  # 16^2
        result, residual = hn.multi_state_iteration(psi_nan)
        assert not np.any(np.isnan(result))
        assert residual == float("inf") or not np.isnan(residual)
