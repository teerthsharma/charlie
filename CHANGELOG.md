# Changelog

All notable changes to OmniTopos will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-05-10

### Added

- **OmniTopos v0.1.0**: Universal Topology Engine — cosmological emergence via persistent homology phase space
- **Six topology engines unified**:
  - `FARADAY` — God Tensor: E↔H coupling, spectral fixed-point (ε_machine = 1.8×10⁻¹⁶)
  - `HAMLITON` — N-Body tensor: multi-body coupling, tensor product Hilbert space
  - `LAMBDA-TOPO` — Topological memory: ripser barcodes, FAISS-style similarity search
  - `EPSILON-CLI` — Stochastic bang: noise-driven emergence, POVM state collapse
  - `AETHER-LINK` — Measurement apparatus: POVM collapse, Born rule
  - `HOLLOW` — EM manifold: FDFD cavity, wave superposition, GC manifold guards
- **Big Bang as topological phase transition**: VACUUM → H0_ONLY → H0H1 → H0H1H2 → GOD_FIXED
- **God Tensor convergence**: power iteration with machine-epsilon tolerance (1e-15)
- **Betti signature cosmology**: Betti numbers as fundamental cosmological state variables
- **PhaseManager**: autonomous topological phase evolution with stochastic resonance
- **StochasticBang**: vacuum fluctuation from Planck-scale energy density (1.22e-19 GeV)
- **NaN/Inf guards** on all floating-point operations
- **Exception handling** with structlog warnings throughout
- **RUNBOOK.md**: full reproducibility guide with parameter tables and expected outputs

### Documentation

- README.md: problem statement, solution, quickstart, architecture, six-engine breakdown
- RUNBOOK.md: installation, CLI usage, architecture, parameters, reproducibility
- CONTRIBUTING.md: development setup, quality gates, branching strategy
- LICENSE: MIT

### Quality

- `ruff check .` → 0 violations
- `mypy --strict` → 0 errors
- Seeded RNG on all stochastic operations (seed=42 default)