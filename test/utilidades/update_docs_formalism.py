import re

files_to_update = ["Paper.md", "Poema.md", "Poema-manual.md"]

for filename in files_to_update:
    try:
        with open(filename, "r") as f:
            content = f.read()

        # Update URT
        content = re.sub(
            r"There exists a constructive morphism \\Phi_AC such that \\Phi_AC\(f\) is represented as a finite composition of FMA\/GEMM operations\.",
            r"There exists a constructive morphism $\Phi_{AC}$, formally tracked by `verify_urt_koopman_bound` at runtime, which bounds the computable domain mapped to FMA/GEMM operations. Non-linear Koopman reductions empirically demonstrate mathematically bounded truncation errors $\delta(d)$.",
            content
        )
        
        # Add new chapter on formal verification if not exist
        if "## Formal Verification Reality (Runtime Invariants)" not in content:
            content += """
## Formal Verification Reality (Runtime Invariants)
This framework shifts theoretical conjectures into hardened runtime observables via the `poema.formal_verification` module:
- **1.1. Universal Reduction Theorem (URT):** Acknowledged as an open research bounding problem for non-analytic functions. Evaluated at runtime computing $\delta(d) = \|\Phi_d(f) - f\|$.
- **1.2. FMA Conservation Law:** Monitored as a strict structural AST property ($E(f) = E(\Phi(f))$) rather than a universally assumed Noether-like symmetry.
- **1.3. Functorial Composition:** Measured dynamically ($\Phi(f \circ g) - (\Phi(f) \circ \Phi(g))$) to bound divergences in mixed Koopman-polynomial trees.
- **1.4. $\alpha(f)$ Index Discrepancies:** Combinatorial, spectral, and geometric derivations are quantified simultaneously to log convergence instead of asserting immediate unifications.
- **1.5. Inexact Reversibility:** $\Phi^{-1}$ reconstruction exactness verified primarily over polynomial scopes; Koopman reconstructions are actively restricted and bounded via explicit error tolerances.
"""
        with open(filename, "w") as f:
            f.write(content)
            print(f"Updated {filename}")
    except Exception as e:
        print(f"Failed {filename}: {e}")

