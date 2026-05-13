"""Big Bang Simulation — full cosmological simulation from vacuum to God Tensor fixed point."""

from __future__ import annotations

import structlog
from typing import Any

import numpy as np

from omni_topos._types import BettiSignature, TopologicalState, TopologyPhase
from omni_topos.phase_manager import PhaseManager

log = structlog.get_logger(__name__)


class BigBangSimulation:
    """Full cosmological simulation from vacuum to God Tensor fixed point.

    This is the top-level orchestrator. It initializes the God Tensor from
    random signature data, then runs the phase manager iteratively until
    either:
    - The simulation reaches the target phase (default: H0H1H2), OR
    - The God Tensor residual drops below 1e-15 (machine epsilon), OR
    - The maximum number of steps is reached

    The simulation tracks a full timeline of TopologicalStates for analysis
    of phase transition dynamics.

    Attributes:
        initial_topology: Starting phase name (default "vacuum")
        final_topology: Target phase name (default "h0h1h2")
        max_steps: Maximum simulation iterations (default 10000)
        seed: Random seed for reproducibility (default 42)
        rng: Numpy random generator
        phase_manager: PhaseManager managing phase evolution
    """

    def __init__(
        self,
        initial_topology: str = "vacuum",
        final_topology: str = "h0h1h2",
        max_steps: int = 10000,
        seed: int = 42,
    ) -> None:
        self.initial_topology = initial_topology
        self.final_topology = final_topology
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.phase_manager = PhaseManager()
        self._initialize_god_tensor()

    def _initialize_god_tensor(self) -> None:
        """Initialize the God Tensor from randomly generated signatures.

        Generates 50 pairs of (electric, magnetic) signatures as
        random unit vectors in R^latent_dim, then learns T and computes
        the fixed point via power iteration.
        """
        n_samples = 50
        e_sigs = self.rng.standard_normal((n_samples, 16))
        h_sigs = self.rng.standard_normal((n_samples, 16))
        for sig in [e_sigs, h_sigs]:
            for row in sig:
                row[:] = row / np.linalg.norm(row)
        self.phase_manager.god_tensor.learn_T(e_sigs, h_sigs)
        x_star, residual = self.phase_manager.god_tensor.power_iteration()
        log.info("god_tensor_initialized", residual=residual)

    def run(self) -> dict[str, Any]:
        """Execute the full cosmological simulation.

        Returns:
            Dictionary with:
                n_steps: Total simulation steps run
                final_phase: The final TopologyPhase reached
                final_betti: String representation of final BettiSignature
                final_entropy: Final topological entropy value
                final_residual: God Tensor residual at end (None if not converged)
                phase_transitions: Number of phase transitions in timeline
                timeline: List[TopologicalState] of all states
        """
        initial_sig = self.rng.standard_normal(16)
        initial_sig = initial_sig / np.linalg.norm(initial_sig)
        state = TopologicalState(
            phase=TopologyPhase.VACUUM,
            betti=BettiSignature.vacuum(),
            hilbert_signature=initial_sig,
            barcode=[],
            entropy=0.0,
            age=0.0,
        )
        timeline: list[TopologicalState] = [state]
        for step in range(self.max_steps):
            state = self.phase_manager.evolve_phase(state, self.rng)
            timeline.append(state)
            if state.residual is not None and state.residual < 1e-15:
                log.info("simulation_converged", step=step, residual=state.residual)
                break
            target_reached = (
                self.final_topology == "h0h1h2" and state.phase == TopologyPhase.H0H1H2
            ) or (state.residual is not None and state.residual < 1e-15)
            if target_reached:
                log.info("target_phase_reached", step=step, phase=state.phase.value)
                break

        final_state = timeline[-1]
        transitions = sum(
            1 for i in range(1, len(timeline)) if timeline[i].phase != timeline[i - 1].phase
        )
        return {
            "n_steps": len(timeline),
            "final_phase": final_state.phase.value,
            "final_betti": repr(final_state.betti),
            "final_entropy": final_state.entropy,
            "final_residual": final_state.residual,
            "phase_transitions": transitions,
            "timeline": timeline,
        }


# =============================================================================
# CLI
# =============================================================================


def main() -> None:
    """CLI entry point for OmniTopos simulation."""
    import argparse

    parser = argparse.ArgumentParser(description="OmniTopos: Cosmological Topology Simulator")
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Run big bang simulation",
    )
    parser.add_argument(
        "--phases",
        default="all",
        help="Target phases",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10000,
        help="Max simulation steps",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--god-check",
        action="store_true",
        help="Verify God Tensor convergence",
    )
    args = parser.parse_args()

    if args.god_check:
        from omni_topos.faraday import GodTensorEngine

        gt = GodTensorEngine()
        rng = np.random.default_rng(args.seed)
        e_sigs = rng.standard_normal((50, 16))
        h_sigs = rng.standard_normal((50, 16))
        for sig in [e_sigs, h_sigs]:
            for row in sig:
                row[:] = row / np.linalg.norm(row)
        gt.learn_T(e_sigs, h_sigs)
        x_star, residual = gt.power_iteration()
        print(f"God Tensor residual: {residual:.3e}")
        print(f"Machine epsilon: 1.8e-16  |  Converged: {residual < 1e-15}")

    elif args.simulate:
        sim = BigBangSimulation(
            initial_topology="vacuum",
            final_topology=args.phases,
            max_steps=args.steps,
            seed=args.seed,
        )
        result = sim.run()
        print("Simulation complete:")
        print(f"  Steps: {result['n_steps']}")
        print(f"  Final phase: {result['final_phase']}")
        print(f"  Final Betti: {result['final_betti']}")
        print(f"  Final entropy: {result['final_entropy']:.4f}")
        print(f"  Final residual: {result['final_residual']}")
        print(f"  Phase transitions: {result['phase_transitions']}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
