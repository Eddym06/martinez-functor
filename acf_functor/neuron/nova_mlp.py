"""
NovaMLP — MLP nativo con neuronas Nova.

Cada capa oculta es un AUTOENCODER (SVD targets).
La última capa es supervisada.
Entre capas: ReLU.
"""

import numpy as np
from .nova_phi_neuron import NovaPhiNeuron, NovaPrediction


class NovaMLP:
    def __init__(self, layer_dims, max_degree=2, max_pairs=60, l2_lambda=0.1):
        """
        Args:
            layer_dims: [n_input, hidden1, hidden2, ..., n_output]
        """
        self.layer_dims = list(layer_dims)
        self.neurons = []
        for i in range(len(layer_dims) - 1):
            n_in = layer_dims[i]
            n_out = layer_dims[i + 1]
            pairs = min(max_pairs, n_in * (n_in - 1) // 2, 200)
            neuron = NovaPhiNeuron(
                name=f"nmlp_L{i}",
                n_input=n_in,
                n_output=n_out,
                max_degree=max_degree,
                max_pairs=pairs,
                l2_lambda=l2_lambda,
            )
            self.neurons.append(neuron)

    @staticmethod
    def _svd_targets(X, n_components):
        """Generar targets vía SVD parcial (mismo método que NovaConv2D)."""
        n_comp = min(n_components, min(X.shape) - 1, 64)
        try:
            U, s, Vt = np.linalg.svd(X, full_matrices=False)
            targets = np.zeros((len(X), n_components))
            for ch in range(min(n_components, len(s))):
                targets[:, ch] = U[:, ch] * s[ch]
        except np.linalg.LinAlgError:
            targets = np.random.randn(len(X), n_components) * 0.1
        stds = np.std(targets, axis=0) + 1e-8
        return targets / stds[None, :]

    def _predict_batch(self, neuron, X):
        """Evaluar neurona en batch."""
        B = neuron._eval_basis(X)
        return np.array([neuron._predict_one(B[i]) for i in range(len(X))])

    def fit(self, X, Y, verbose=True):
        X = np.asarray(X, np.float64)
        Y = np.asarray(Y, np.float64)
        H = X.copy()
        t0 = __import__('time').perf_counter()

        for i, neuron in enumerate(self.neurons[:-1]):
            # Capa oculta: autoencoder con SVD targets
            targets = self._svd_targets(H, neuron.n_output)
            neuron.fit(H, targets)
            # Forward con ReLU
            H = np.maximum(0, self._predict_batch(neuron, H))
            if verbose:
                t = __import__('time').perf_counter() - t0
                print(f"  L{i}: {neuron.n_input}→{neuron.n_output} "
                      f"[{t:.1f}s] ε={neuron.epsilon_mu:.4f}")

        # Capa final: supervisada
        self.neurons[-1].fit(H, Y)
        if verbose:
            t = __import__('time').perf_counter() - t0
            print(f"  L{len(self.neurons)-1}: {self.neurons[-1].n_input}"
                  f"→{self.neurons[-1].n_output} [{t:.1f}s]")

    def predict(self, x):
        x = np.atleast_2d(np.asarray(x, np.float64))
        h = x.copy()
        for neuron in self.neurons[:-1]:
            h = np.maximum(0, self._predict_batch(neuron, h))
        return self.neurons[-1].predict(h[0])

    def evaluate(self, x):
        return self.predict(x).mean

    def summary(self):
        dims = ' → '.join(str(d) for d in self.layer_dims)
        return f"NovaMLP({dims}) | {len(self.neurons)} layers"
