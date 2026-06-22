"""
FRN — Functorial Nova Network (Red de Nova-Functorial)
========================================================
Arquitectura Nova-nativa que compone functores ANOVA(2).

  Input ──→ [N₁ N₂ N₃ ... Nₖ] ──→ Crossing ──→ Output
              ↑ neurons in parallel     ↑ ANOVA(2) sobre outputs

Capa 1: K neuronas en PARALELO, cada una con distintos pares
        (diversidad vía correlation_threshold escalonado).
Capa 2 (Crossing): ANOVA(2) sobre los OUTPUTS de la capa 1.
        Captura interacciones entre neuronas.

Propiedades:
  - Sin backprop, sin SGD
  - Interpretable: cada neurona tiene sentido
  - Compone functores: ANOVA(2) ∘ ANOVA(2)
  - Maneja caos: cada neurona ve el input completo
  - Robusto a ruido: la redundancia paralela filtra outliers
"""

import numpy as np, time
from .nova_phi_neuron import NovaPhiNeuron, NovaPrediction


class FunctorialNovaNetwork:
    """FRN: Red de Nova-Functorial.

    Capa horizontal: K neuronas ANOVA(2) en paralelo.
    Capa crossing: 1 neurona ANOVA(2) sobre [out_N1 | out_N2 | ... | out_NK].
    """

    def __init__(self, name: str, n_input: int, n_output: int,
                 n_parallel: int = 6,
                 max_degree: int = 2,
                 max_pairs_per: int = 80,
                 crossing_pairs: int = 120,
                 l2_lambda: float = 0.1):
        self.name = name
        self.n_input = n_input
        self.n_output = n_output
        self.n_parallel = n_parallel

        # ── Capa horizontal: K neuronas en paralelo ──
        self.horizontal: list[NovaPhiNeuron] = []
        for k in range(n_parallel):
            # Diversidad: cada neurona ve diferentes pares
            ct = 0.02 + k * 0.015  # correlation_threshold creciente
            neuron = NovaPhiNeuron(
                name=f"{name}_H{k}",
                n_input=n_input,
                n_output=8,  # feature bottleneck
                max_degree=max_degree,
                max_pairs=max_pairs_per,
                l2_lambda=l2_lambda,
                correlation_threshold=ct,
            )
            self.horizontal.append(neuron)

        # ── Capa crossing: ANOVA(2) sobre outputs horizontales ──
        crossing_input_dim = n_parallel * 8  # K neuronas × 8 features cada una
        self.crossing = NovaPhiNeuron(
            name=f"{name}_X",
            n_input=crossing_input_dim,
            n_output=n_output,
            max_degree=2,
            max_pairs=crossing_pairs,
            l2_lambda=l2_lambda,
            correlation_threshold=0.03,
        )

    def _forward_horizontal(self, X: np.ndarray) -> np.ndarray:
        """Evaluar todas las neuronas horizontales en batch."""
        outputs = []
        for neuron in self.horizontal:
            B = neuron._eval_basis(X)
            preds = np.array([neuron._predict_one(B[i]) for i in range(len(X))])
            # ReLU para no-linealidad entre capas
            outputs.append(np.maximum(0, preds))
        return np.hstack(outputs)  # (n_samples, n_parallel * 8)

    def fit(self, X: np.ndarray, Y: np.ndarray, verbose: bool = True):
        X = np.asarray(X, np.float64)
        Y = np.asarray(Y, np.float64)
        t0 = time.perf_counter()

        # ── Fase 1: Entrenar capa horizontal en paralelo ──
        if verbose:
            print(f"🧬 FRN '{self.name}': {self.n_parallel} neurons × {X.shape[1]}→8")
        for k, neuron in enumerate(self.horizontal):
            # Target autoencoder (SVD) para capas ocultas
            n_comp = min(8, min(X.shape) - 1, 32)
            try:
                U, s, Vt = np.linalg.svd(X, full_matrices=False)
                targets = np.zeros((len(X), 8))
                for ch in range(min(8, len(s))):
                    targets[:, ch] = U[:, ch] * s[ch]
                stds = np.std(targets, axis=0) + 1e-8
                targets = targets / stds[None, :]
            except np.linalg.LinAlgError:
                targets = np.random.randn(len(X), 8) * 0.1
            neuron.fit(X, targets)
            if verbose:
                t = time.perf_counter() - t0
                print(f"  H{k}: ε={neuron.epsilon_mu:.4f} p={len(neuron._pairs)} [{t:.1f}s]")

        # ── Fase 2: Entrenar capa crossing ──
        H = self._forward_horizontal(X)
        if verbose:
            print(f"  Crossing: {H.shape[1]}→{self.n_output}")
        self.crossing.fit(H, Y)
        if verbose:
            t = time.perf_counter() - t0
            print(f"  X: ε={self.crossing.epsilon_mu:.4f} p={len(self.crossing._pairs)} [{t:.1f}s]")

    def predict(self, x: np.ndarray) -> NovaPrediction:
        x = np.atleast_2d(np.asarray(x, np.float64))
        H = self._forward_horizontal(x)
        return self.crossing.predict(H[0])

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        return self.predict(x).mean

    def summary(self) -> str:
        return (f"FRN('{self.name}') {self.n_input}→[{self.n_parallel}×8]→{self.n_output} | "
                f"total_neurons={self.n_parallel+1} | ANOVA²∘ANOVA²")
