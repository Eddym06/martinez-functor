# koopman_fma.py
import sympy as sp
import numpy as np

def explicit_koopman_lifting():
    """
    Demuestra cómo una dinámica no lineal se eleva a un espacio lineal puro 
    para conservar la operación FMA perfecta sin depender de W(y).
    
    Sistema No Lineal Original:
    x1' = mu * x1
    x2' = lambda * (x2 - x1^2)  <-- No Lineal, rompe la forma FMA pura W*y + b
    
    Observables de Koopman (Lifting):
    z1 = x1
    z2 = x2
    z3 = x1^2
    """
    mu, lam = sp.symbols('mu lambda')
    x1, x2 = sp.symbols('x1 x2')
    
    # Derivadas clásicas
    dx1 = mu * x1
    dx2 = lam * (x2 - x1**2)
    
    print("1. SISTEMA ORIGINAL (No Lineal)")
    print(f"  x1' = {dx1}")
    print(f"  x2' = {dx2}  <-- El término x1^2 requiere super-FMA\n")
    
    # Operador de Koopman en el observable subyacente z3 = x1^2
    z1, z2, z3 = sp.symbols('z1 z2 z3')
    
    # dx1^2 / dt = 2 * x1 * dx1 = 2 * x1 * (mu * x1) = 2 * mu * x1^2 = 2 * mu * z3
    dz1 = mu * z1
    dz2 = lam * z2 - lam * z3
    dz3 = 2 * mu * z3
    
    # Matriz del Operador Estático de Koopman
    K = sp.Matrix([
        [mu, 0,    0],
        [0,  lam, -lam],
        [0,  0,    2*mu]
    ])
    Z = sp.Matrix([z1, z2, z3])
    
    dZ = K * Z
    
    print("2. ESPACIO DE KOOPMAN (Lifting Estructural)")
    print(f"  Matriz Oculta (K) =\n{sp.pretty(K)}\n")
    print(f"  Vector Z' = K * Z =\n{sp.pretty(dZ)}\n")
    
    print("CONCLUSIÓN AXIOM-1:")
    print("Al elevar la dimensión a través de los observables, la no-linealidad")
    print("desaparece. El sistema vuelve a ser evaluable exacta y puramente")
    print("con llamadas de hardware Tensor Core FMA [y_next = K * y_prev + 0]!")

if __name__ == "__main__":
    print("="*60)
    print("AXIOM-1: SOLUCIÓN NO LINEAL VIA OPERADORES DE KOOPMAN")
    print("="*60)
    explicit_koopman_lifting()
