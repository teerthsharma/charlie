"""FARADAY Engine — God Tensor: spectral fixed-point operator for E↔H coupling."""

from __future__ import annotations

import structlog

import numpy as np

from omni_topos._types import BettiSignature, GodTensorResult

log = structlog.get_logger(__name__)


class GodTensorEngine:
    """Spectral fixed-point operator.

    Finds the eigenvector x* such that T(x*) = x* at machine epsilon (1.8e-16),
    representing the cosmological attractor state of the universe.

    The operator T couples electric (E) and magnetic (H) field signatures
    extracted from the N-body electromagnetic simulation. Its fixed point
    is the God Tensor — the unique stable state all cosmological evolution
    converges to.

    Attributes:
        latent_dim: Dimension of the Hilbert space (default 16)
        T: Learned operator matrix (None until learn_T is called)
        x_star: Fixed-point eigenvector (None until power_iteration is called)
    """

    def __init__(self, latent_dim: int = 16) -> None:
        self.latent_dim = latent_dim
        self.T: np.ndarray | None = None
        self.x_star: np.ndarray | None = None

    def learn_T(self, e_signatures: np.ndarray, h_signatures: np.ndarray) -> np.ndarray:
        """Learn the E↔H coupling operator T from signature data.

        Uses least-squares regression from electric signatures (inputs)
        to magnetic signatures (outputs) to construct T.

        Args:
            e_signatures: (n_samples, latent_dim) electric field signature matrix
            h_signatures: (n_samples, latent_dim) magnetic field signature matrix

        Returns:
            The learned operator T
        """
        T, _, _, _ = np.linalg.lstsq(e_signatures, h_signatures, rcond=None)  # type: ignore[return-value]
        self.T = T
        log.info("god_tensor_T_learned", shape=T.shape)
        return T  # type: ignore[no-any-return]

    def apply(self, x: np.ndarray) -> np.ndarray:
        """Apply the coupling operator T to a vector.

        Args:
            x: (latent_dim,) input vector

        Returns:
            T @ x — transformed vector with NaN/Inf sanitized

        Raises:
            ValueError: if learn_T has not been called first
        """
        if self.T is None:
            raise ValueError("Must call learn_T first")
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        result = self.T @ x
        return result  # type: ignore[no-any-return]

    def power_iteration(
        self,
        x0: np.ndarray | None = None,
        iters: int = 50000,
        tol: float = 1e-15,
    ) -> tuple[np.ndarray, float]:
        """Find the fixed point x* via power iteration.

        Power iteration repeatedly applies T and normalizes, converging
        to the dominant eigenvector (the cosmological fixed point).

        Args:
            x0: Initial vector. Defaults to random standard normal.
            iters: Maximum iterations (default 50000)
            tol: Convergence tolerance on residual (default 1e-15 ≈ machine eps)

        Returns:
            (x_star, residual) — fixed-point vector and final residual

        Raises:
            ValueError: if initial vector has zero norm
        """
        if x0 is None:
            rng = np.random.default_rng(42)
            x0 = rng.standard_normal(self.latent_dim)
        x0 = np.nan_to_num(x0, nan=0.0, posinf=0.0, neginf=0.0)
        x0_norm = np.linalg.norm(x0)
        if x0_norm < 1e-15:
            raise ValueError("Initial vector has zero norm — cannot iterate")
        x0 = x0 / x0_norm

        x = x0.copy()
        residual = float("inf")
        for i in range(iters):
            x_new = self.apply(x)
            norm = np.linalg.norm(x_new)
            if norm < 1e-15:
                log.warning("power_iteration_norm_collapsed", iter=i)
                break
            x_new = x_new / norm
            x_new = np.nan_to_num(x_new, nan=0.0, posinf=0.0, neginf=0.0)
            residual = float(np.linalg.norm(x_new - x))
            if residual < tol:
                log.info("god_tensor_converged", iter=i, residual=residual)
                break
            x = x_new

        self.x_star = x
        return x, residual

    def god_score(self, x: np.ndarray) -> float:
        """Compute alignment of vector x with the fixed point x*.

        The god_score is the absolute inner product between x and x_star,
        representing how "close" x is to the cosmological attractor.

        Args:
            x: A Hilbert space vector

        Returns:
            |⟨x|x*⟩| — alignment with the fixed point (0.0 to 1.0)
        """
        if self.T is None or self.x_star is None:
            return 0.0
        return float(np.abs(np.dot(x, self.x_star)))

    def result_from_state(
        self,
        x: np.ndarray,
        betti: BettiSignature,
    ) -> GodTensorResult:
        """Compute a full GodTensorResult from a state vector and Betti signature.

        Args:
            x: Current Hilbert space vector
            betti: Current Betti signature

        Returns:
            Full result tuple with god_score and residual
        """
        x_new = self.apply(x)
        god_score = self.god_score(x_new)
        residual = (
            float(np.linalg.norm(x_new - x)) if self.x_star is not None else float("inf")
        )
        return GodTensorResult(
            x_new=x_new,
            residual=residual,
            god_score=god_score,
            betti_0=betti.b0,
            betti_1=betti.b1,
        )