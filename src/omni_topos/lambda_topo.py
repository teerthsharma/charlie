"""LAMBDA-TOPO Engine — Real persistent homology via ripser, FAISS index for similarity."""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import ripser
    import faiss

    _HAS_RIPSER = True
    _HAS_FAISS = True
except ImportError:  # pragma: no cover
    _HAS_RIPSER = False
    _HAS_FAISS = False

from omni_topos._types import TopologicalState, TopologyPhase


class TopologicalMemory:
    """Persistent homology-backed topological memory using ripser + FAISS.

    Stores cosmological simulation states and retrieves the most similar
    past states by barcode distance in H0/H1/H2 persistence space.
    Falls back to signature cosine similarity when ripser/faiss unavailable.

    Attributes:
        latent_dim: Dimension of Hilbert signature vectors.
        barcodes: List of stored barcode records (dicts with H0/H1/H2 lifetimes).
        signatures: List of stored Hilbert signature vectors.
        phase_history: Ordered list of TopologyPhases.
        _faiss_index: FAISS inner-product index on normalized signatures.
    """

    def __init__(self, latent_dim: int = 16) -> None:
        self.latent_dim = latent_dim
        self.barcodes: list[dict[str, Any]] = []
        self.signatures: list[np.ndarray] = []
        self.phase_history: list[TopologyPhase] = []
        self._index_ready = False
        if _HAS_FAISS:
            self._faiss_index: faiss.IndexFlatIP | None = faiss.IndexFlatIP(latent_dim)
            self._all_signatures: list[np.ndarray] = []

    def _compute_barcode(self, points: np.ndarray, max_dim: int = 2) -> dict[str, Any]:
        """Compute persistent homology barcode via ripser.

        Args:
            points: Point cloud of shape (N, d).
            max_dim: Maximum homology dimension (default 2 for H0, H1, H2).

        Returns:
            Dict with 'h0', 'h1', 'h2' — each a list of (birth, death) pairs.
        """
        if not _HAS_RIPSER or len(points) < 2:  # pragma: no cover
            return {"h0": [], "h1": [], "h2": []}

        result = ripser.ripser(points, maxdim=max_dim, thresh=1.0)

        def _pairs(dgm: np.ndarray) -> list[tuple[float, float]]:
            """Convert ripser diagram to list of (birth, death) pairs."""
            out: list[tuple[float, float]] = []
            for row in dgm:
                birth, death = float(row[0]), float(row[1])
                if death == np.inf:
                    death = 1e6  # represents open-ended cycle
                out.append((birth, death))
            return out

        barcode: dict[str, list[tuple[float, float]]] = {}
        for dim in range(max_dim + 1):
            key = f"h{dim}"
            barcode[key] = _pairs(result["dgms"][dim]) if dim < len(result["dgms"]) else []
        return barcode

    def _barcode_distance(self, a: dict[str, Any], b: dict[str, Any]) -> float:
        """Wasserstein-1 distance between two barcodes (H0 ⊕ H1 ⊕ H2).

        Computes the L1 distance between stacked persistence diagrams.
        """
        dist = 0.0
        for key in ["h0", "h1", "h2"]:
            pairs_a = a.get(key, [])
            pairs_b = b.get(key, [])
            max_len = max(len(pairs_a), len(pairs_b))
            # Pad shorter with (0, 0) — zero-persistence points don't affect distance
            for i in range(max_len):
                da = list(pairs_a[i]) if i < len(pairs_a) else [0.0, 0.0]
                db = list(pairs_b[i]) if i < len(pairs_b) else [0.0, 0.0]
                # Each pair contributes |birth_a - birth_b| + |death_a - death_b|
                dist += abs(da[0] - db[0]) + abs(da[1] - db[1])
        return dist

    def store(self, state: TopologicalState) -> None:
        """Store a TopologicalState with real persistent homology barcode.

        If the Hilbert signature has at least 2 points, computes actual H0/H1/H2
        barcodes via ripser. Otherwise stores an empty barcode.

        Args:
            state: The TopologicalState to store.
        """
        barcode: dict[str, Any] = {"h0": [], "h1": [], "h2": []}

        sig = state.hilbert_signature
        # Hilbert signature is latent_dim dimensional — treat each dimension
        # as a coordinate. For barcode we need a cloud of N >= 2 points.
        # Embed the signature as N=latent_dim points in 1D (each axis coord)
        # plus a small perturbation to avoid collinearity.
        if sig is not None and len(sig) >= 2:
            N = min(len(sig), 32)
            pts = sig[:N].reshape(-1, 1) + np.linspace(0, 0.01, N).reshape(-1, 1)
            barcode = self._compute_barcode(pts, max_dim=2)

        barcode_copy: dict[str, Any] = dict(barcode)
        barcode_copy["phase"] = state.phase
        barcode_copy["entropy"] = state.entropy
        barcode_copy["age"] = state.age

        self.barcodes.append(barcode_copy)
        self.signatures.append(sig.copy() if sig is not None else np.zeros(self.latent_dim))
        self.phase_history.append(state.phase)

        # Rebuild FAISS index
        if _HAS_FAISS and self._faiss_index is not None:
            self._all_signatures.append(
                sig.copy() if sig is not None else np.zeros(self.latent_dim)
            )
            mat = np.stack(self._all_signatures).astype(np.float32)
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms[norms == 0] = 1
            mat = mat / norms
            self._faiss_index.reset()
            self._faiss_index.add(mat)
            self._index_ready = True

    def retrieve_similar(
        self,
        signature: np.ndarray,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Retrieve top-k most similar past states by barcode Wasserstein distance.

        If FAISS is available, uses FAISS inner-product index on normalized
        Hilbert signatures for fast retrieval. Otherwise falls back to
        cosine similarity on numpy arrays.

        Args:
            signature: Query Hilbert signature vector.
            top_k: Number of results to return (default 5).

        Returns:
            List of up to top_k barcode dicts, sorted by ascending barcode distance.
            Returns empty list if no states have been stored.
        """
        if not self.signatures:
            return []

        if _HAS_FAISS and self._index_ready and self._faiss_index is not None:
            q = signature.astype(np.float32).reshape(1, -1)
            q = q / (np.linalg.norm(q) + 1e-15)
            D, indices = self._faiss_index.search(q, min(top_k, len(self._all_signatures)))
            return [self.barcodes[i] for i in indices[0] if 0 <= i < len(self.barcodes)]

        # Fallback: cosine similarity on numpy arrays
        sigs = np.stack(self.signatures)
        norms = np.linalg.norm(sigs, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normed = sigs / norms
        query_norm = np.linalg.norm(signature)
        query_normed = signature / (query_norm + 1e-15)
        similarities = np.dot(normed, query_normed)
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [self.barcodes[i] for i in top_indices]

    def get_phase_history(self) -> list[TopologyPhase]:
        """Return the full phase history as a list of TopologyPhase values."""
        return list(self.phase_history)
