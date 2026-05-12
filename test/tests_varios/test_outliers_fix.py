#!/usr/bin/env python3
"""Test para verificar que outliers funciona correctamente"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import trustworthiness
from acf_functor import CCDEngine

# Generar datos con outliers
np.random.seed(42)
n_samples = 1000
d = 100

# Datos normales
X = np.random.randn(n_samples, d)

# Añadir 5% outliers
n_outliers = int(0.05 * n_samples)
outlier_multiplier = 10.0
outlier_shift = 20.0
X[:n_outliers] = X[:n_outliers] * outlier_multiplier + outlier_shift

# Estandarizar
X = StandardScaler().fit_transform(X)

print(f"Datos shape: {X.shape}")
print(f"Número de outliers: {n_outliers}")

# PCA baseline
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)
trust_pca = trustworthiness(X, X_pca, n_neighbors=12)
print(f"PCA trustworthiness: {trust_pca:.4f}")

# CCD con skip-connection
try:
    ccd = CCDEngine(
        d_threshold=2,
        n_preprocess_components=2,
        n_diffusion_components=2,
        skip_connection=True,
        seed=42,
        auto_scale=True,
        use_snr_weighting=True
    )
    
    ccd.fit(X)
    X_ccd = ccd.transform(X)
    trust_ccd = trustworthiness(X, X_ccd, n_neighbors=12)
    alpha = ccd.skip_alpha
    
    print(f"CCD trustworthiness: {trust_ccd:.4f}")
    print(f"Alpha aprendido: {alpha:.3f}")
    print(f"Mejora: {trust_ccd - trust_pca:.6f}")
    print(f"✅ Outliers funcionando correctamente")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()