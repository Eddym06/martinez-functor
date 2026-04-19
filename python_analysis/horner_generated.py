# AUTO-GENERADO desde MathTest/HornerExtract.lean - NO MODIFICAR MANUALMENTE
# Teorema: hornerFMASeq_correct

def horner_fma_seq(coeffs):
    """Genera secuencia FMA de Horner como lista (w, b)."""
    n = len(coeffs)
    if n <= 1:
        return []
    # coeffs = [a0, a1, ..., an], secuencia usa [a_{n-1}, ..., a0]
    return [(1.0, float(c)) for c in reversed(coeffs[:-1])]

def horner_seed(coeffs):
    return float(coeffs[-1]) if coeffs else 0.0

__lean_source__ = "MathTest/HornerExtract.lean"
__theorem__ = "hornerFMASeq_correct"
