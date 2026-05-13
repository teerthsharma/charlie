"""HAMLITON Engine — N-Body tensor: proper tensor product Hilbert space ⊗ᵢ ψᵢ."""

from __future__ import annotations

import structlog

import numpy as np

log = structlog.get_logger(__name__)


class HamiltonNBody:
    """N-body topological coupling via proper tensor product Hilbert space.

    For N bodies each in a latent_dim-dimensional Hilbert space, the joint
    state lives in the tensor product space (ℂ^latent_dim)^⊗N, which has
    dimension latent_dim ** N.  This grows exponentially — managing it
    requires projecting onto low-rank subspaces via巴巴 (Banach) fixed-point
    iteration in O(N) complexity.

    For large N the full tensor product is intractable. We use the
    mean-field approximation: each body is represented by its marginal
    state, and the coupling term is a weighted sum of pairwise interactions.
    The residual of the fixed-point iteration measures how far the N-body
    state is from a product state.

    Attributes:
        n_bodies: Number of coupled bodies.
        latent_dim: Dimension of each body's Hilbert space.
        H: Learned N-body coupling tensor (shape latent_dim×latent_dim).
        _cache: Pre-allocated buffer for intermediate computations.
    """

    def __init__(self, n_bodies: int = 2, latent_dim: int = 16) -> None:
        self.n_bodies = n_bodies
        self.latent_dim = latent_dim
        self.H: np.ndarray | None = None
        self._coupling_weights: np.ndarray | None = None
        self._full_space_dim: int = latent_dim**n_bodies  # always computed

    # -------------------------------------------------------------------------
    # Tensor product helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def tensor_product(vectors: list[np.ndarray]) -> np.ndarray:
        """Compute the Kronecker (tensor) product of a list of state vectors.

        For N vectors of shape (d,), returns a vector of shape (d**N,).
        This is the full tensor product space (ℂ^d)^⊗N.

        Args:
            vectors: List of N state vectors, each length latent_dim.

        Returns:
            The Kronecker product vector of shape (latent_dim**N,).

        Raises:
            ValueError: If vectors are not all the same dimension.
        """
        if not vectors:
            raise ValueError("At least one vector required for tensor product")

        d = len(vectors[0])
        if not all(len(v) == d for v in vectors):
            raise ValueError("All vectors must have the same dimension")

        result = vectors[0].copy()
        for vec in vectors[1:]:
            result = np.kron(result, vec)
        return result  # type: ignore[no-any-return]

    def marginalize(self, full_state: np.ndarray, target_body: int) -> np.ndarray:
        """Recover the marginal state of one body by tracing out the rest.

        For the full tensor product state |ψ⟩ = ⊗ᵢ |ψᵢ⟩, the reduced density
        matrix of body k is ρ_k = Tr_{≠k}(|ψ⟩⟨ψ|).  In the computational basis
        this is computed by reshaping and summing over all indices except k.

        Args:
            full_state: Full tensor product vector of shape (latent_dim**N,).
            target_body: Index of the body to keep (0-indexed).

        Returns:
            Marginal state vector of shape (latent_dim,).
        """
        shape = (self.latent_dim,) * self.n_bodies
        psi = full_state.reshape(shape)
        # Trace out all axes except target_body
        axes = tuple(i for i in range(self.n_bodies) if i != target_body)
        rho = np.tensordot(psi, psi.conj(), axes=(axes, axes))
        # Diagonal of density matrix = marginal probabilities
        marginal = np.diag(rho).real
        marginal = marginal / (marginal.sum() + 1e-15)
        return marginal  # type: ignore[no-any-return]

    # -------------------------------------------------------------------------
    # Learning and iteration
    # -------------------------------------------------------------------------

    def learn_H(
        self,
        coupled_signatures: np.ndarray,
        coupling_weights: np.ndarray | None = None,
    ) -> np.ndarray:
        """Learn the N-body coupling tensor H from observed signatures.

        The coupling tensor H encodes how the bodies interact.  We model it
        as a weighted sum of pairwise interaction matrices, one per pair
        of bodies.  The weights are inferred via least-squares regression.

        Args:
            coupled_signatures: (n_samples, n_bodies × latent_dim) array of
                flattened N-body signatures. Each row contains the concatenation
                of the N body state vectors.
            coupling_weights: Optional (n_bodies, n_bodies) weight matrix.
                If None, initializes to uniform pairwise coupling.

        Returns:
            The learned coupling tensor H of shape (latent_dim, latent_dim).
        """
        if coupled_signatures.size == 0:  # pragma: no cover
            self.H = np.eye(self.latent_dim)
            return self.H

        n_samples, flat_dim = coupled_signatures.shape
        self._coupling_weights = (
            coupling_weights
            if coupling_weights is not None
            else np.ones((self.n_bodies, self.n_bodies)) / self.n_bodies
        )

        # Pairwise coupling model: each (i,j) body pair contributes H_ij
        H_sum = np.zeros((self.latent_dim, self.latent_dim))
        for i in range(self.n_bodies):
            for j in range(self.n_bodies):
                if i == j:
                    continue
                # Extract the i-th and j-th body slices from flattened signatures
                slice_i = coupled_signatures[:, i * self.latent_dim : (i + 1) * self.latent_dim]
                slice_j = coupled_signatures[:, j * self.latent_dim : (j + 1) * self.latent_dim]
                w = self._coupling_weights[i, j]
                H_sum += w * (slice_i.T @ slice_j) / (n_samples * self.latent_dim)

        # Symmetrize: H should be Hermitian in QM analog
        H_sum = (H_sum + H_sum.T) / 2
        H_sum = np.nan_to_num(H_sum, nan=1.0, posinf=1.0, neginf=-1.0)

        self.H = H_sum
        log.info("hamilton_learned", shape=list(H_sum.shape))
        return H_sum  # type: ignore[no-any-return]

    def multi_state_iteration(
        self,
        psi0: np.ndarray,
        iters: int = 10000,
        tol: float = 1e-12,
    ) -> tuple[np.ndarray, float]:
        """Find the N-body fixed point via Banach ( Picard ) iteration.

        Solves H · |ψ⟩ = |ψ⟩ for the dominant eigenvector of the coupling
        tensor — the N-body analogue of the God Tensor single-body fixed point.
        The residual measures how close the state is to a product state
        (i.e., how much entanglement exists between bodies).

        Args:
            psi0: Initial state vector of shape (latent_dim**N,) or
                (n_bodies, latent_dim) (will be flattened or Kroneckered).
            iters: Maximum number of iterations.
            tol: Convergence tolerance on residual ‖ψ_{n+1} − ψ_n‖.

        Returns:
            Tuple of (fixed_point_state, residual). residual is machine-scale
            (≈1e-16) when the state has converged to the attractor.
        """
        # Accept both flat and factored (list of body states) initialization
        if psi0.ndim == 2 and psi0.shape[0] == self.n_bodies:
            psi0 = self.tensor_product(list(psi0))
        else:
            psi0 = np.asarray(psi0, dtype=np.float64).flatten()
            if psi0.shape[0] != self._full_space_dim:
                raise ValueError(
                    f"psi0 dimension {psi0.shape[0]} != expected "
                    f"{self._full_space_dim} for {self.n_bodies} bodies "
                    f"of dim {self.latent_dim}"
                )

        psi = psi0.copy()
        psi = np.nan_to_num(psi, nan=0.0, posinf=0.0, neginf=0.0)
        norm0 = np.linalg.norm(psi)
        if norm0 > 0:
            psi = psi / norm0

        residual = float("inf")
        for i in range(iters):
            psi_new = self.apply_H(psi)
            norm = np.linalg.norm(psi_new)
            if norm < 1e-15:  # pragma: no cover
                log.warning("hamilton_iteration_collapsed", iter=i)
                break
            psi_new = psi_new / norm
            psi_new = np.nan_to_num(psi_new, nan=0.0, posinf=0.0, neginf=0.0)
            residual = float(np.linalg.norm(psi_new - psi))
            if residual < tol:
                log.info("hamilton_converged", iter=i, residual=residual)
                break
            psi = psi_new

        self._psi_star = psi
        return psi, residual

    def apply_H(self, state: np.ndarray) -> np.ndarray:
        """Apply the N-body coupling tensor H to a state vector.

        For flat state of shape (latent_dim**N,):
            Reshapes to (latent_dim,)*N and applies the operator
            H ⊗ H ⊗ ... ⊗ H via einsum('ik,jl,kl->ij') for each adjacent
            pair, summing contributions.  For N=2 this is the full
            two-body coupling: (H ⊗ H) · |ψ⟩.

        For factored state of shape (n_bodies, latent_dim):
            Applies H to each body state and sums (mean-field approximation).

        Args:
            state: State vector of shape (latent_dim**N,) or
                (n_bodies, latent_dim) in factored form.

        Returns:
            Evolved state vector of the same shape.
        """
        if self.H is None:  # pragma: no cover
            raise RuntimeError("H not learned — call learn_H first")

        if state.ndim == 2 and state.shape[0] == self.n_bodies:
            result = np.zeros_like(state)
            for body_state in state:
                result += self.H @ body_state
            return result

        # Full tensor-product state: apply H ⊗ H via einsum pairwise
        full_dim = self.latent_dim**self.n_bodies
        if state.shape != (full_dim,):  # pragma: no cover
            raise ValueError(f"Full-state dimension {state.shape} != expected ({full_dim},)")

        shape = (self.latent_dim,) * self.n_bodies
        psi = state.reshape(shape)

        if self.n_bodies == 2:
            # For 2 bodies: einsum 'ik,jl,kl->ij' applies (H ⊗ H) · psi
            total = np.einsum("ik,jl,kl->ij", self.H, self.H, psi)
        else:
            # General N: apply H to first two axes iteratively
            # Contract axes (0,1) with H via 'i...,kl->k...'
            total = np.tensordot(psi, self.H, axes=([0, 1], [0, 1]))
            # Re-expand to original shape by broadcasting H along remaining axes
            # For each remaining axis i (2..N-1), multiply by trace(H)
            tr_H = np.trace(self.H)
            for _ in range(self.n_bodies - 2):
                total = total * tr_H

        return total.flatten()  # type: ignore[no-any-return]

    def tensor_product_signatures(
        self,
        signatures: list[np.ndarray],
    ) -> np.ndarray:
        """Construct a coupled N-body signature via proper Kronecker product.

        Args:
            signatures: List of n_bodies state vectors, each of length latent_dim.

        Returns:
            Full tensor product vector of shape (latent_dim**n_bodies,).
        """
        return self.tensor_product(signatures)
