# Martínez's Invariant: Universal Reduction Theorem

This repository contains the rigorous mathematical formalization of the **Coverage Lemma** for **Martínez's Invariant**, also known as the **Universal Reduction Theorem (URT)**, implemented in Lean 4.

## Project Overview

The project formalizes advanced analytical and topological concepts, integrating:
- **Bournez's Completeness** and Taylor Expansions over Continuous Functions.
- **Orthogonal Projections** in Reproducing Kernel Hilbert Spaces (RKHS).
- **Koopman Operator Linearizations** using empirical inner products.
- Exact type casting and Hilbert Space coercions via $L^p$ integrations (`WithLp.equiv 2`).

The formalization successfully compiles with **0 errors and 0 warnings (no `sorry` placeholders)** against a modern Lean 4 `mathlib` environment.

## Repository Structure

- `MathTest/`: Contains the primary formal proofs in Lean 4.
  - `urt_coverage.lean`: Master formalization file asserting the Coverage Lemma mathematically.
- `python_analysis/`: Python tools and scripts for numerical modeling and analytical exploration (e.g., FMA derivations, Koopman matrices).
- `archive/`: Legacy exploratory proof attempts and theoretical drafts.
- `Paper.md`: Detailed documentation, research narrative, and theoretical background.

## Build and Verification

To verify the proofs mathematically on your local machine, ensure you have Lean 4 and `lake` installed:

```bash
# Build the entire library (downloads mathlib cache if needed)
lake build

# Or check the main coverage lemma directly
lake env lean MathTest/urt_coverage.lean
```
