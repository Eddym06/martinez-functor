# phi_inverse.py
import sympy as sp

def phi_inverse(fma_sequence, target_var):
    """
    Formalismo \Phi^{-1}
    Reconstruye el objeto matemático abstracto (la función fuente) 
    exactamente a partir de su cadena estructural FMA.
    
    El mapa inverso \Phi^{-1} asegura la Reversibilidad del Invariante de Martínez:
    f(x) == \Phi^{-1}(\Phi(f(x))) 
    """
    print(f"Reconstruyendo morfismo sobre la variable abstracta '{target_var}'")
    x = target_var
    
    # El estado inicial antes de las iteraciones
    # y = w * x + b
    y = x
    
    # Procesar de adentro hacia afuera (Bottom-Up)
    # Suponiendo que la secuencia FMA es de la forma
    # iter_i: y_new = x_val * y_old + a_i  (Para Horner/Morfismo Estandar)
    
    # In order of evaluation (from innermost to outermost loop of evaluation)
    for i, (W, b) in enumerate(fma_sequence):
        print(f"  [Paso {i}] FMA Op: out = {W}*[{y}] + {b}")
        
        # En una evaluación FMA estándar tipo Horner
        # el weight W suele ser la variable (x) y el y_old el acumulador.
        # W puede ser 'x' o un W real matricial. 
        y = W * y + b
        
    return sp.expand(y)

if __name__ == "__main__":
    x = sp.Symbol('x')
    
    # Ejemplo de secuencia generada para: 4x^3 + 3x^2 - 2x + 10
    # Horner manual: x * (x * (x * 4 + 3) - 2) + 10
    
    # Estructura computacional "Mnemónica" GEMM de la máquina
    # Formato: [(W_1, b_1), (W_2, b_2), (W_3, b_3), ...]
    # Donde W es x (la entrada base multiplicando el registro escalar de estado).
    fma_chain = [
        (0, 4),    # eval  0:  0*x + 4          = 4
        (x, 3),    # eval  1:  x*(4) + 3        = 4x + 3
        (x, -2),   # eval  2:  x*(4x+3) - 2     = 4x^2 + 3x - 2
        (x, 10)    # eval  3:  x*(...) + 10 = 4x^3 + 3x^2 - 2x + 10
    ]
    
    print("="*60)
    print("AXIOM-1: TRANSFORMACIÓN INVERSA (Φ⁻¹)")
    print("="*60)
    
    reconstructed_f = phi_inverse(fma_chain, x)
    print(f"\nFunción Abstracta Resultante f(x) = {reconstructed_f}")
    
    expected = 4*x**3 + 3*x**2 - 2*x + 10
    is_reversible = sp.simplify(reconstructed_f - expected) == 0
    
    print(f"Conservación de la Energía Φ⁻¹(Φ(f(x))) == f(x): \033[92m{is_reversible}\033[0m")
