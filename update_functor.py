import re

with open("martinez/functor.py", "r") as f:
    content = f.read()

methods_to_add = r"""
    def compute_martinez_invariant(self, eigenvalues):
        """
        [Evolution 4] Computes the Martínez Invariant α(f) based on the spectral 
        geometry of the Koopman operator. α(f) unifies computational complexity 
        (Kolmogorov entropy) with spectral decay.
        """
        # Form: α(f) = limsup_{j->∞} [ -log|λ_j| / log(j) ]
        if isinstance(eigenvalues, np.ndarray):
            eigenvalues = torch.tensor(eigenvalues)
        
        j_values = torch.arange(1, len(eigenvalues) + 1, dtype=torch.float64)
        spectral_decay = -torch.log(torch.abs(eigenvalues)) / torch.log(j_values + 1e-9)
        alpha = torch.max(spectral_decay).item()
        
        # Enriched cost: δ(d) ~ |λ_{d+1}| 
        delta_d = torch.abs(eigenvalues[-1]).item() if len(eigenvalues) > 0 else 0.0
        
        print(f"[Φ-Functor Enriched] Martínez Invariant α(f): {alpha:.4f}, Truncation Cost δ(d): {delta_d:.1e}")
        return alpha, delta_d

    def stratified_execute(self, x, selectors, strata_functions, device="cuda"):
        """
        [Evolution 3] Stratified Category Execution (Comp_strat -> GEMM_δ x Bool)
        Executes piecewise topological manifolds continuously without breaking 
        FMA compatibility. Evaluates constructible sheaves using Heaviside selectors.
        
        selectors: list of boolean masks or functions returning boolean masks.
        strata_functions: list of pre-reduced matrix weights corresponding to each S_i.
        """
        x = x.to(device)
        result = torch.zeros_like(x).to(device)
        
        print("[Φ-Functor Stratified] Merging GEMM_δ ⊕ GEMM_δ via logical Bool sheaf selectors...")
        
        for condition, W_strata in zip(selectors, strata_functions):
            # Evaluate the selector (H) to isolate the manifold S_i
            mask = condition(x) if callable(condition) else condition
            mask = mask.to(device).to(x.dtype)
            
            # W_strata is assumed pre-compensated α(f_i) structure matrices
            W_strata = W_strata.to(device)
            branch_result = torch.matmul(x, W_strata)
            
            # Multiply by Heaviside limits (mask) logically
            result += mask * branch_result
            
        return result
"""

# Insert these methods right before "def execute"
content = content.replace('    def execute(self, input_tensor, device="cuda"):', methods_to_add + '\n    def execute(self, input_tensor, device="cuda"):')

with open("martinez/functor.py", "w") as f:
    f.write(content)
