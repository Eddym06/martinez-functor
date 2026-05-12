"""
Spectral Preprocessor Module
=============================

This module contains the SpectralPreprocessor class which performs PCA preprocessing
with optional whitening and provides safe access to the underlying PCA model.

Capabilities:
  - PCA projection with optional whitening (variance normalization)
  - Inverse transform (reconstruction from latent codes)
  - Explained variance ratio and effective rank computation
  - Robust fallbacks for rank-deficient or degenerate data
  - Compatible with CCDEngine (exposes `pca` attribute, `get_explained_variance()`)

WHITENING GUIDANCE:
  - whiten=False (default): preserves metric (ρ ≈ 1.0). Recommended for
    downstream diffusion geometry which relies on distance fidelity.
  - whiten=True: normalizes each component to unit variance. Use when
    downstream methods expect isotropic inputs (e.g., some clustering).
"""

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import warnings


class SpectralPreprocessor:
    """
    Spectral Preprocessor that applies PCA followed by optional scaling.

    This class provides safe access to the underlying PCA model through
    the `pca_model` property and `get_explained_variance()` method.

    Args:
        n_components (int): Number of components to keep
        whiten (bool, optional): Whether to whiten the data. Defaults to False.
    """

    def __init__(self, n_components: int, whiten: bool = False):
        self.n_components = n_components
        self.whiten = whiten
        self._scaler = StandardScaler() if whiten else None
        self._pca = None
        self._fitted = False

        # Precomputed transform state
        self._mean: np.ndarray = None
        self._Vt: np.ndarray = None
        self._scale: np.ndarray = None

    # ── Public aliases for compatibility with ccd_engine.py ────────────────

    @property
    def pca(self):
        """Alias for pca_model — used by ccd_engine.py introspection."""
        return self.pca_model

    @property
    def pca_model(self):
        """Get the underlying PCA model."""
        if not self._fitted:
            raise RuntimeError("Preprocessor has not been fitted yet")
        return self._pca

    def get_explained_variance(self):
        """Get the explained variance of the PCA model."""
        return self.pca_model.explained_variance_

    # ── Core API ──────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray) -> "SpectralPreprocessor":
        """
        Fit the preprocessor to the data.

        Args:
            X (np.ndarray): Input data of shape (n, d)

        Returns:
            SpectralPreprocessor: The fitted preprocessor
        """
        n, d = X.shape
        n_comp = min(self.n_components, min(n, d))

        if n_comp < 1:
            raise ValueError(
                f"Cannot fit PCA with n_components={self.n_components} "
                f"on data with shape ({n}, {d})"
            )

        self._pca = PCA(n_components=n_comp, whiten=self.whiten)
        self._pca.fit(X)

        # Extract transform state for inverse_transform
        self._mean = self._pca.mean_
        self._Vt = self._pca.components_
        ev = self._pca.explained_variance_
        self._scale = np.sqrt(np.maximum(ev, 1e-12))

        if self.whiten:
            Z = self._pca.transform(X)
            self._scaler.fit(Z)

        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data using the fitted preprocessor.

        Args:
            X (np.ndarray): Input data of shape (n, d)

        Returns:
            np.ndarray: Transformed data of shape (n, n_components)
        """
        if not self._fitted:
            raise RuntimeError("Preprocessor has not been fitted yet")

        Z = (X - self._mean) @ self._Vt.T

        if self.whiten:
            Z = Z / (self._scale + 1e-8)
            Z = self._scaler.transform(Z)
        return Z

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit and transform data in a single step.

        Args:
            X (np.ndarray): Input data of shape (n, d)

        Returns:
            np.ndarray: Transformed data of shape (n, n_components)
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        """
        Reconstruct approximate X from whitened PCA code Z.

        Args:
            Z (np.ndarray): Latent codes of shape (n, n_components)

        Returns:
            np.ndarray: Reconstructed data of shape (n, d)
        """
        if not self._fitted:
            raise RuntimeError("Preprocessor has not been fitted yet")

        if self.whiten:
            Z = self._scaler.inverse_transform(Z)
            Z_unscaled = Z * self._scale
        else:
            Z_unscaled = Z

        return Z_unscaled @ self._Vt + self._mean

    # ── Diagnostic properties ─────────────────────────────────────────────

    @property
    def explained_variance_ratio(self) -> np.ndarray:
        """Fraction of total variance explained by each PCA component."""
        s2 = self._scale ** 2
        return s2 / (s2.sum() + 1e-12)

    @property
    def effective_rank(self) -> int:
        """
        Smallest number of components explaining ≥ 95% of variance.

        Returns at least 1, at most n_components.
        """
        cumsum = np.cumsum(self.explained_variance_ratio)
        rank = int(np.searchsorted(cumsum, 0.95)) + 1
        return max(1, min(rank, self._Vt.shape[0]))

    @property
    def condition_number(self) -> float:
        """
        Condition number of the PCA transform (max eigenvalue / min eigenvalue).

        High values indicate near-singular data; low values indicate well-conditioned.
        """
        if not self._fitted:
            raise RuntimeError("Preprocessor has not been fitted yet")
        ev = self._scale ** 2
        return float(ev[0] / (ev[-1] + 1e-12))

    @property
    def singular_values(self) -> np.ndarray:
        """Singular values of the training data (sqrt of explained variance * n)."""
        if not self._fitted:
            raise RuntimeError("Preprocessor has not been fitted yet")
        return self._scale * np.sqrt(self._pca.n_samples_)

    # ── Utility ───────────────────────────────────────────────────────────

    def summary(self) -> str:
        """Return a concise text summary of the preprocessor state."""
        if not self._fitted:
            return "SpectralPreprocessor (not fitted)"
        r = self.effective_rank
        evr = self.explained_variance_ratio
        return (
            f"SpectralPreprocessor(n_components={self.n_components}, "
            f"whiten={self.whiten})\n"
            f"  input dim: {self._Vt.shape[1]}  |  "
            f"output dim: {self._Vt.shape[0]}  |  "
            f"effective rank (95%): {r}\n"
            f"  top-3 var ratio: {evr[:min(3, len(evr))].round(3)}\n"
            f"  condition number: {self.condition_number:.1f}"
        )

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "not fitted"
        return f"SpectralPreprocessor(n={self.n_components}, whiten={self.whiten}, {status})"
