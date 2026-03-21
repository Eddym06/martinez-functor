import re

with open('Paper.md', 'r', encoding='utf-8') as f:
    content = f.read()

new_section = r"""
## 5. The FMA Residual: Intrinsic Self-Correction and Reversibility ($\Phi^{-1}$)

A foundational breakthrough of Martínez's Invariant is the realization that hardware rounding errors (induced by imperfect silicon precision, e.g., FP32 or FP8) are not "external bugs" needing software patches. Rather, under the Functor $\Phi$, they represent *computational energy leaks* that can be intrinsically recaptured by the algebraic topology itself.

### 5.1. The Intrinsic Residue in FMA
In physical hardware, an FMA operation computes $a \times b + c$ and rounds the result at the end. Mathematically, there exists an exact residue $r$ such that:
$$a \times b + c = \text{rounded\_result} + r$$

To make this natural to the functor, the sequence of GEMM matrices is not computed as a straight line, but as a recursive helix. When decomposing a function, the functor generates two parallel flows within the weights:
1. **Primary Flow:** Computes the principal approximation of the function.
2. **Invariance Flow:** Utilizes the properties of the FMA itself to calculate the exact residue $r$ of the previous step, folding it into the inner product of the next step.

### 5.2. Geometrically Clean Horner's Method
In traditional software, Horner's algorithm for polynomials evaluates as $y_{i} = x \cdot y_{i-1} + a_{n-i}$.
By recognizing the error as topological energy, the functor transforms this recursion. The error of iteration $i-1$ is mathematically absorbed into the coefficient $a_{n-i}$ of iteration $i$. The implementation error "dilutes" directly into the structure of the polynomial matrix representation itself, ensuring numerical stability by design, rather than through additive overhead.

### 5.3. Pre-compensated Reversibility as Control ($\Phi^{-1}$)
Using the reversibility defined in Section 4.3, the functor projects the function to a GEMM. Before execution, the functor predicts the catastrophic loss of the silicon hardware:
1. It projects the symbolic function in exact precision ($\mathbb{R}$, modeled as FP64).
2. It collapses the weights locally to predict the hardware FP32/FP8 rounding loss.
3. The functor inherently calculates the exact difference (the "noise" of the silicon) and adjusts the matrix weights $W$ *prior to execution*.

The result is a GEMM matrix whose weights are physically "pre-compensated" with the exact errors the hardware is destined to make. 

**Conclusion of the Mechanised Functor:**
Because the correction lives strictly inside the parameters of the matrix, there is **Zero Runtime Overhead**. The functor analyzes silicon "noise", adapts the calculus morphology, and processes predicting errors to mathematically cancel them out. It transforms standard GPU operations into a fundamentally perfect computational machine under the scope of its precision boundary.

## 6. Real-World Applications and Engineering
"""

content = content.replace("## 5. Real-World Applications and Engineering", new_section)

with open('Paper.md', 'w', encoding='utf-8') as f:
    f.write(content)
