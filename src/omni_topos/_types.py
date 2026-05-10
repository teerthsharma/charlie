"""Type definitions shared across all OmniTopos modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypeAlias

import numpy as np


# Teerth's convention: dict[str, Any] over TypedDict
ModeData: TypeAlias = dict[str, Any]
BettiState: TypeAlias = tuple[int, int, int, int]


class TopologyPhase(Enum):
    """Topological phase of the cosmological state."""

    VACUUM = "vacuum"
    H0_ONLY = "h0_only"
    H0H1 = "h0h1"
    H0H1H2 = "h0h1h2"
    GOD_FIXED = "god_fixed"


@dataclass
class BettiSignature:
    """Betti numbers as cosmological state variables.

    Betti numbers count connected components (b0), loops (b1),
    voids (b2), and higher-dimensional topology (b3) in a simplicial complex.
    In this model, they encode the topological structure of the universe.

    Attributes:
        b0: Connected components — number of disjoint spatial regions
        b1: Holes/loops — closed spatial cycles (electromagnetic wave paths)
        b2: Voids — cavities enclosed by 2-spheres (gravitational wells)
        b3: Higher-order topology — not used in current phase model
        betti_vec: Computed numpy vector of [b0, b1, b2, b3]
    """

    b0: int
    b1: int
    b2: int
    b3: int = 0
    betti_vec: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.float64),
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "betti_vec",
            np.array([self.b0, self.b1, self.b2, self.b3], dtype=np.float64),
        )

    @classmethod
    def vacuum(cls) -> "BettiSignature":
        """Return the vacuum Betti signature (all zeros)."""
        return cls(b0=0, b1=0, b2=0, b3=0)

    @property
    def is_vacuum(self) -> bool:
        """Return True if this signature represents the vacuum state."""
        return bool(self.betti_vec.sum() == 0)

    @property
    def total_features(self) -> int:
        """Total number of topological features across all Betti numbers."""
        return int(self.betti_vec.sum())

    def __repr__(self) -> str:
        return f"Betti({self.b0}, {self.b1}, {self.b2}, {self.b3})"


@dataclass
class TopologicalState:
    """Full topological state of the cosmological simulation.

    Attributes:
        phase: Current TopologyPhase
        betti: Betti signature encoding topological structure
        hilbert_signature: Quantum Hilbert space representation vector
        barcode: Persistent homology barcode of the current filtration
        entropy: Topological entropy (log of total features)
        residual: God Tensor residual (None if not yet converged)
        age: Simulation step count
    """

    phase: TopologyPhase
    betti: BettiSignature
    hilbert_signature: np.ndarray
    barcode: list[Any]
    entropy: float
    residual: float | None = None
    age: float = 0.0


@dataclass
class GodTensorResult:
    """Result from a God Tensor computation.

    Attributes:
        x_new: Next Hilbert space vector after applying T
        residual: Fixed-point residual ||x_new - x||
        god_score: Inner product alignment with fixed point x*
        betti_0: b0 component of the resulting Betti signature
        betti_1: b1 component of the resulting Betti signature
    """

    x_new: np.ndarray
    residual: float
    god_score: float
    betti_0: int
    betti_1: int