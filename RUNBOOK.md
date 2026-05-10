# Charlie — Runbook

## Quick Start

```bash
# Install
pip install -e .

# Run big bang simulation
omni-topos --simulate --phases h0h1h2 --steps 10000 --seed 42

# Verify God Tensor convergence
omni-topos --god-check

# Phase diagram
omni-topos --phase-diagram
```

## Theory

- Universe = topological phase space, not geometric
- Big Bang = topological phase transition (∅ → H0 → H0H1 → H0H1H2)
- Betti numbers (B0, B1, B2) = cosmological state variables
- God Tensor = cosmological attractor (spectral fixed-point at ε_machine ≈ 1.8e-16)
- Stochastic Bang = noise-driven topological bifurcation at Planck scale

## Architecture

```
src/omni_topos/__init__.py
├── GodTensorEngine    — Banach fixed-point: T(x*) = x*
├── HamiltonNBody      — N-body coupling: H^⊗N(Ψ) = Ψ
├── TopologicalMemory  — FAISS-like barcode store
├── StochasticBang     — Big Bang as noise-driven emergence
├── PhaseManager       — VACUUM → H0_ONLY → H0H1 → H0H1H2 → GOD_FIXED
└── BigBangSimulation  — Full simulation entry point
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `latent_dim` | 16 | Hilbert signature vector dimension |
| `n_bodies` | 2 | Number of coupled bodies (Hamilton) |
| `max_steps` | 10000 | Max simulation steps |
| `seed` | 42 | Random seed for reproducibility |
| `tol` (God Tensor) | 1e-15 | Convergence tolerance → machine epsilon |

## Reproducibility

- All randomness via `np.random.default_rng(seed)` — deterministic
- NaN/Inf guards on all matrix operations
- Zero-norm protection on vector normalization
- Exception handling with structured logging

## Six Source Engines

| Engine | Location | Role |
|--------|----------|------|
| FARADAY | github.com/teerthsharma/faraday | God Tensor E↔H coupling |
| HAMLITON | github.com/teerthsharma/hamliton | N-body tensor product |
| LAMBDA-TOPO | github.com/teerthsharma/lambda-topo | Barcode store + ripser |
| EPSILON-CLI | github.com/teerthsharma/epsilon-cli | Stochastic resonance |
| AETHER-LINK | github.com/teerthsharma/aether-link | POVM measurement |
| HOLLOW | lambda-topo/hollow_manifold_sim.py | FDFD cavity + GC guards |

## Output Format

Simulation returns:
```python
{
    "n_steps": int,
    "final_phase": str,
    "final_betti": str,      # "Betti(B0, B1, B2, B3)"
    "final_entropy": float,
    "final_residual": float,  # God Tensor residual — goal: < 1e-15
    "phase_transitions": int,
    "timeline": [TopologicalState, ...]
}
```

## Citation

```bibtex
@article{sharma2026charlie,
  title   = {Charlie: Universal Topology Engine for Cosmological Emergence},
  author  = {Sharma, Teerth},
  year    = {2026},
  url     = {https://github.com/teerthsharma/charlie},
  note    = {Invented May 2026 — GitHub timestamp as legal proof of invention priority}
}
```