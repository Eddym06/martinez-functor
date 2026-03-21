import torch
from martinez import MartinezFunctor

def test_api():
    print("--- Probando Martinez Functor API ---")
    # Inicializando el objeto
    phi = MartinezFunctor(target_dim_d=512, precision="fp32")
    
    # Probando la reducción simbólica en los tres caminos
    print("\n1. Testing Path Routing:")
    r1 = phi.reduce("5*x**3 - 2*x + 1")
    r2 = phi.reduce("sin(exp(x))")
    r3 = phi.reduce("def my_ode_system(x): pass")
    
    # Probando la ejecución del tensor (en CPU para evitar fallos si no hay CUDA local)
    print("\n2. Testing Execution:")
    input_tensor = torch.randn(1, 10, 512)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output = phi.execute(input_tensor, device=device)
    
    print(f"\nResultado de la ejecución. Shape del tensor output: {output.shape}")
    assert output.shape == (1, 10, 512), "Shape mismatch!"
    print("\n✅ API Funciona Correctamente.")

if __name__ == '__main__':
    test_api()
