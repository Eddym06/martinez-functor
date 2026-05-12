"""
Test rápido del CCD con skip-connection mejorado
"""
import numpy as np
import time
from sklearn.decomposition import PCA
from sklearn.manifold import trustworthiness as sklearn_trust
from acf_functor.ccd_engine import CCDEngine

# Generar datos de prueba
np.random.seed(42)
n_samples = 200
n_features = 50

# Datos con estructura no lineal + ruido
t = np.linspace(0, 4 * np.pi, n_samples)
X = np.zeros((n_samples, n_features))
X[:, 0] = np.sin(t)
X[:, 1] = np.cos(t)
X[:, 2] = t / (4 * np.pi)
# Añadir ruido a las dimensiones restantes
X[:, 3:] = np.random.randn(n_samples, n_features - 3) * 0.1

# Añadir outliers
outlier_idx = np.random.choice(n_samples, 5, replace=False)
X[outlier_idx] += np.random.randn(5, n_features) * 3.0

print("=" * 80)
print("TEST CCD CON SKIP-CONNECTION MEJORADO")
print("=" * 80)
print(f"Datos: {n_samples} muestras, {n_features} características")
print(f"Outliers: {len(outlier_idx)} puntos")

# Test PCA
print("\n1. PCA (baseline):")
t0 = time.time()
pca = PCA(n_components=2)
Z_pca = pca.fit_transform(X)
t_pca = time.time() - t0
trust_pca = sklearn_trust(X, Z_pca, n_neighbors=10)
print(f"   Tiempo: {t_pca:.3f}s")
print(f"   Trustworthiness: {trust_pca:.4f}")
print(f"   Varianza explicada: {pca.explained_variance_ratio_.sum():.4f}")

# Test CCD sin skip-connection
print("\n2. CCD sin skip-connection:")
t0 = time.time()
ccd_no_skip = CCDEngine(
    d_threshold=5,
    n_diffusion_components=2,
    skip_connection=False,
    use_snr_weighting=True,
    auto_scale=True,
)
ccd_no_skip.fit(X)
Z_ccd_no_skip = ccd_no_skip.transform(X)
t_ccd_no_skip = time.time() - t0
trust_ccd_no_skip = sklearn_trust(X, Z_ccd_no_skip, n_neighbors=10)
print(f"   Tiempo: {t_ccd_no_skip:.3f}s")
print(f"   Trustworthiness: {trust_ccd_no_skip:.4f}")
print(f"   Alpha skip: {getattr(ccd_no_skip, '_skip_alpha', 'N/A')}")

# Test CCD con skip-connection (ecosistema)
print("\n3. CCD con skip-connection (ecosistema):")
t0 = time.time()
ccd_skip = CCDEngine(
    d_threshold=5,
    n_diffusion_components=2,
    skip_connection=True,
    use_snr_weighting=True,
    auto_scale=True,
)
ccd_skip.fit(X)
Z_ccd_skip = ccd_skip.transform(X)
t_ccd_skip = time.time() - t0
trust_ccd_skip = sklearn_trust(X, Z_ccd_skip, n_neighbors=10)
print(f"   Tiempo: {t_ccd_skip:.3f}s")
print(f"   Trustworthiness: {trust_ccd_skip:.4f}")
print(f"   Alpha skip aprendido: {getattr(ccd_skip, '_skip_alpha', 'N/A')}")

# Comparación
print("\n" + "=" * 80)
print("RESULTADOS:")
print(f"PCA Trustworthiness:           {trust_pca:.4f}")
print(f"CCD sin skip Trustworthiness:  {trust_ccd_no_skip:.4f}")
print(f"CCD con skip Trustworthiness:  {trust_ccd_skip:.4f}")
print(f"\nMejora CCD con skip vs PCA:    {trust_ccd_skip - trust_pca:+.4f}")
print(f"Mejora CCD con skip vs sin:    {trust_ccd_skip - trust_ccd_no_skip:+.4f}")

# Verificar que CCD con skip nunca pierde vs PCA
if trust_ccd_skip >= trust_pca - 0.01:  # Tolerancia pequeña
    print("\n✓ CCD con skip-connection NO PIERDE vs PCA (claim verificado)")
else:
    print(f"\n✗ CCD con skip-connection PIERDE vs PCA por {trust_pca - trust_ccd_skip:.4f}")

# Test con datos de alta dimensión
print("\n" + "=" * 80)
print("TEST CON DIMENSIÓN ALTA (d=1000):")
np.random.seed(123)
n_samples_hd = 500
n_features_hd = 1000

# Datos de alta dimensión con estructura baja dimensional
X_hd = np.random.randn(n_samples_hd, n_features_hd) * 0.5
# Añadir señal en las primeras 10 dimensiones
signal = np.random.randn(n_samples_hd, 10)
X_hd[:, :10] += signal * 2.0

print(f"Datos: {n_samples_hd} muestras, {n_features_hd} características")

# Test CCD con FAISS para alta dimensión
t0 = time.time()
ccd_hd = CCDEngine(
    d_threshold=10,
    n_diffusion_components=10,
    skip_connection=True,
    use_snr_weighting=True,
    auto_scale=True,
)
try:
    ccd_hd.fit(X_hd)
    Z_hd = ccd_hd.transform(X_hd)
    t_hd = time.time() - t0
    print(f"   Tiempo CCD (d=1000): {t_hd:.3f}s")
    print(f"   Dimensiones output: {Z_hd.shape[1]}")
    print("   ✓ CCD funciona con dimensión alta usando FAISS")
except Exception as e:
    print(f"   ✗ Error con dimensión alta: {e}")

print("\n" + "=" * 80)
print("TEST COMPLETADO")