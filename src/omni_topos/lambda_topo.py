"""LAMBDA-TOPO Engine — Topological memory: ripser barcodes, FAISS-style similarity store."""

from __future__ import annotations

from typing import Any

import numpy as np

from omni_topos._types import TopologicalState, TopologyPhase


class TopologicalMemory:
    """FAISS-style index for Betti signatures and phase history.

    Stores cosmological simulation states and retrieves the most similar
    past states by Hilbert signature cosine similarity. Used to track
    phase evolution and detect topological cycles in the simulation.

    Attributes:
        latent_dim: Dimension of Hilbert signature vectors
        barcodes: List of stored barcode records (dicts)
        signatures: List of stored Hilbert signature vectors
        phase_history: Ordered list of TopologyPhases
    """

    def __init__(self, latent_dim: int = 16) -> None:
        self.latent_dim = latent_dim
        self.barcodes: list[dict[str, Any]] = []
        self.signatures: list[np.ndarray] = []
        self.phase_history: list[TopologyPhase] = []

    def store(self, state: TopologicalState) -> None:
        """Store a TopologicalState in the memory index.

        Makes a deep copy of the state's barcode to ensure the stored
        record is stable even if the state object is modified later.

        Args:
            state: The TopologicalState to store
        """
        barcode_copy: list[Any] = list(state.barcode) if state.barcode else []
        self.barcodes.append({
            "phase": state.phase,
            "betti": state.betti.betti_vec.copy(),
            "barcode": barcode_copy,
            "entropy": state.entropy,
            "age": state.age,
        })
        self.signatures.append(state.hilbert_signature.copy())
        self.phase_history.append(state.phase)

    def retrieve_similar(
        self,
        signature: np.ndarray,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Retrieve the top-k most similar past states by cosine similarity.

        Uses normalized dot product as the similarity metric. States with
        zero-norm signatures are handled gracefully.

        Args:
            signature: Query Hilbert signature vector
            top_k: Number of results to return (default 5)

        Returns:
            List of up to top_k barcode dicts, sorted by descending similarity.
            Returns empty list if no states have been stored.
        """
        if not self.signatures:
            return []
        sigs = np.stack(self.signatures)
        norms = np.linalg.norm(sigs, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normed = sigs / norms
        query_norm = np.linalg.norm(signature)
        query_normed = signature / query_norm if query_norm > 1e-15 else signature
        similarities = np.dot(normed, query_normed)
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [self.barcodes[i] for i in top_indices]

    def get_phase_history(self) -> list[TopologyPhase]:
        """Return the full phase history as a list of TopologyPhase values."""
        return list(self.phase_history)