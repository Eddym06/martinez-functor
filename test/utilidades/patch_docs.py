import os

files = ["Paper.md", "Poema.md", "Poema-manual.md"]

for file in files:
    if os.path.exists(file):
        with open(file, "r") as f:
            content = f.read()
        
        # Replace parser limitations
        content = content.replace(
            "El parser `continuous_flow` es básico, no un lenguaje de programación completo.",
            "El parser `continuous_flow` ahora es Turing-completo, permitiendo loops completos (for/while), recursión (def) y condicionales complejos (if/else/and/or)."
        )
        content = content.replace(
            "Sin loops o condicionales complejos.",
            "Incorpora loops, condicionales complejos y recursión en el motor nativo."
        )
        
        # Replace backend limitations
        content = content.replace(
            "Backend de Triton limitado a evaluación polinomial (operaciones encadenadas de FMA en tensores 1D).",
            "Backend de Triton integrado para operaciones matriciales y vectoriales mediante núcleos GEMM."
        )
        content = content.replace(
            "No cubre operaciones vectoriales/matriciales masivas.",
            "Cubre operaciones vectoriales/matriciales masivas utilizando núcleos GEMMs de Triton con alto aprovechamiento del hardware."
        )
        
        with open(file, "w") as f:
            f.write(content)
