"""HAMLITON Engine — N-Body tensor: tensor product Hilbert space ⊗ᵢ ψᵢ."""

from __future__ import annotations

import structlog

import numpy as np

log = structlog.get_logger(__name__)


class HamiltonNBody:
    """N-body extension of the God Tensor.

    Extends the single-body spectral operator to multi-body systems
    where each body occupies a tensor product factor of the full
    Hilbert space: H_total = ⊗ᵢ Hᵢ

    In cosmological terms, this models the gravitational coupling
    between N bodies in the universe — each body's field contribution
    is computed jointly in the tensor product space.

    Attributes:
        n_bodies: Number of coupled bodies
        latent_dim: Dimension of each body's Hilbert space factor
        H: Learned Hamiltonian matrix (None until learn_H is called)
    """

    def __init__(self, n_bodies: int, latent_dim: int = 16) -> None:
        self.n_bodies = n_bodies
        self.latent_dim = latent_dim
        self.H: np.ndarray | None = None

    def tensor_product_signatures(self, signatures: list[np.ndarray]) -> np.ndarray:
        """Project N individual signatures into the tensor product Hilbert space.

        Given N body signatures, computes their tensor product as a flattened
        vector in the combined Hilbert space of dimension latent_dim^N.

        Args:
            signatures: List of N (latent_dim,) arrays, one per body

        Returns:
            Flattened (n_bodies * latent_dim,) coupled signature vector,
            L2-normalized to unit length

        Raises:
            ValueError: if the number of signatures does not equal n_bodies
        """
        if len(signatures) != self.n_bodies:
            raise ValueError(f"Expected {self.n_bodies} signatures")
        stacked = np.stack(signatures, axis=0)
        coupled = stacked.flatten()
        coupled = coupled / np.linalg.norm(coupled)
        return coupled  # type: ignore[no-any-return]

    def learn_H(self, coupled_signatures: np.ndarray) -> np.ndarray:
        """Learn the Hamiltonian H from coupled signature data.

        Uses least-squares regression of the coupled signatures onto
        themselves to learn the interaction Hamiltonian.

        Args:
            coupled_signatures: (n_samples, n_bodies * latent_dim) matrix
                of coupled N-body signatures

        Returns:
            The learned Hamiltonian matrix H
        """
        H, _, _, _ = np.linalg.lstsq(coupled_signatures, coupled_signatures, rcond=None)  # type: ignore[return-value]
        self.H = H
        log.info("hamilton_H_learned", shape=H.shape)
        return H  # type: ignore[no-any-return]

    def multi_state_iteration(
        self,
        psi0: np.ndarray,
        iters: int = 1000,
        tol: float = 1e-12,
    ) -> tuple[np.ndarray, float]:
        """Iterate a multi-body state toward the Hamiltonian fixed point.

        Repeatedly applies the Hamiltonian H to the state vector,
        normalizing at each step, until convergence or max iterations.

        Args:
            psi0: Initial state vector in the coupled Hilbert space
            iters: Maximum iterations (default 1000)
            tol: Convergence tolerance on residual (default 1e-12)

        Returns:
            (psi, residual) — converged state and final residual

        Raises:
            ValueError: if learn_H has not been called first
        """
        if self.H is None:
            raise ValueError("Must call learn_H first")
        psi = psi0.copy()
        psi = np.nan_to_num(psi, nan=0.0, posinf=0.0, neginf=0.0)
        residual = float("inf")
        try:
            for i in range(iters):
                psi_new = self.H @ psi
                norm = np.linalg.norm(psi_new)
                if norm < 1e-15:
                    break
                psi_new = psi_new / norm
                psi_new = np.nan_to_num(psi_new, nan=0.0, posinf=0.0, neginf=0.0)
                residual = float(np.linalg.norm(psi_new - psi))
                if residual < tol:
                    log.info("hamilton_converged", iter=i, residual=residual)
                    break
                psi = psi_new
        except Exception:
            log.warning("hamilton_iteration_error", psi_norm=np.linalg.norm(psi))
            raise
        return psi, float(residual)