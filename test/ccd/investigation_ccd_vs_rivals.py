"""
=============================================================================
INVESTIGACIÓN PROFUNDA v2: CCDEngine vs Rivales del Estado del Arte
Métricas correctas con sklearn.manifold.trustworthiness (implementación oficial)
NOTA: continuity(X,Z) = trustworthiness(Z,X) — matemáticamente probado.
      LCMC = (trust + cont) / 2
=============================================================================
"""
import sys, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
np.random.seed(42)
from sklearn.manifold import TSNE, Isomap, SpectralEmbedding, trustworthiness
from sklearn.decomposition import KernelPCA, PCA
from sklearn.datasets import make_swiss_roll, make_s_curve, make_blobs, load_digits, load_wine, load_breast_cancer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import umap as umap_lib
sys.path.insert(0, "/home/Martínez's Invariant")
from acf_functor.ccd_engine import CCDEngine, SpectralPreprocessor, DiffusionGeometry
RNG = np.random.RandomState(42)

def lcmc_metrics(X_orig, X_emb, k=15, n_sample=200):
    n = len(X_orig)
    ns = min(n_sample, n)
    idx = RNG.choice(n, ns, replace=False)
    Xo, Xe = X_orig[idx], X_emb[idx]
    tw = trustworthiness(Xo, Xe, n_neighbors=k)
    co = trustworthiness(Xe, Xo, n_neighbors=k)
    k_e = min(k, len(Xo)-1)
    nn_o = NearestNeighbors(n_neighbors=k_e+1).fit(Xo).kneighbors(Xo)[1][:,1:]
    nn_e = NearestNeighbors(n_neighbors=k_e+1).fit(Xe).kneighbors(Xe)[1][:,1:]
    pres = float(np.mean([len(set(a.tolist())&set(b.tolist()))/k_e for a,b in zip(nn_o,nn_e)]))
    return tw, co, (tw+co)/2.0, pres

def run_method(name, X, nc, y=None):
    t0 = time.perf_counter()
    n, d = X.shape
    try:
        if name == 'CCDEngine':
            m = CCDEngine(n_diffusion_components=nc, n_langevin_steps=0, fit_decoder=False)
            m.fit(X); Z = m.transform(X)
        elif name == 'PCA':
            Z = PCA(n_components=min(nc, d, n-1)).fit_transform(X)
        elif name == 'KernelPCA':
            Z = KernelPCA(n_components=nc, kernel='rbf').fit_transform(X)
        elif name == 'Isomap':
            Z = Isomap(n_components=nc, n_neighbors=min(15, n-1)).fit_transform(X)
        elif name == 'TSNE':
            Z = TSNE(n_components=min(nc,3), perplexity=min(30, n//4),
                     random_state=42, max_iter=500).fit_transform(X)
        elif name == 'UMAP':
            Z = umap_lib.UMAP(n_components=nc, n_neighbors=min(15, n-2),
                              random_state=42).fit_transform(X)
        elif name == 'SpectralEmb':
            Z = SpectralEmbedding(n_components=nc, n_neighbors=min(10, n-1),
                                  random_state=42).fit_transform(X)
        else:
            raise ValueError(name)
        elapsed = time.perf_counter() - t0
        k_use = min(15, n//10, len(Z)-1)
        tw, co, lc, pr = lcmc_metrics(X, Z, k=max(k_use, 5), n_sample=200)
        sil = np.nan
        if y is not None:
            ns = min(200, n)
            idx2 = RNG.choice(n, ns, replace=False)
            if len(np.unique(y[idx2])) >= 2:
                try: sil = silhouette_score(Z[idx2], y[idx2])
                except: pass
        return dict(trust=tw, cont=co, lcmc=lc, pres=pr, sil=sil, time=elapsed, ok=True)
    except Exception as e:
        return dict(trust=np.nan, cont=np.nan, lcmc=np.nan, pres=np.nan,
                    sil=np.nan, time=time.perf_counter()-t0, ok=False, err=str(e)[:60])

def table(results, title):
    print(f"\n{'─'*70}")
    print(f"  {title}")
    print(f"{'─'*70}")
    print(f"  {'Método':<13} {'Trust':>6} {'Cont':>6} {'LCMC':>6} {'Pres':>6} {'Sil':>6} {'t':>7}")
    print(f"  {'─'*64}")
    valid = [(n, r) for n, r in results.items() if r.get('ok') and not np.isnan(r.get('lcmc', np.nan))]
    valid.sort(key=lambda x: x[1]['lcmc'], reverse=True)
    for i, (name, r) in enumerate(valid):
        medal = ['🥇','🥈','🥉','  ','  ','  ','  '][min(i,6)]
        sil_s = f"{r['sil']:6.3f}" if not np.isnan(r.get('sil', np.nan)) else "   N/A"
        mark = ' ◀' if name == 'CCDEngine' else ''
        print(f"  {medal}{name:<12} {r['trust']:6.4f} {r['cont']:6.4f} {r['lcmc']:6.4f} {r['pres']:6.4f} {sil_s} {r['time']:>6.2f}s{mark}")
    for name, r in results.items():
        if not r.get('ok', True):
            print(f"  ❌ {name}: {r.get('err','?')}")
    names = [n for n,_ in valid]
    if 'CCDEngine' in names:
        pos = names.index('CCDEngine') + 1
        delta = valid[0][1]['lcmc'] - results['CCDEngine']['lcmc']
        print(f"\n  ▶ CCD: {pos}/{len(valid)}  ΔLCMC={delta:+.4f}  {'✅ LÍDER' if pos==1 else '⚠️ '+str(pos)+'º'}")
    return valid

print("="*70)
print("  INVESTIGACIÓN PROFUNDA v2: CCDEngine vs Estado del Arte")
print("="*70)
t_global = time.perf_counter()

# ─── FASE 0: Diagnóstico n_pre ───────────────────────────────────────────────
print("\n" + "="*70)
print("  FASE 0: DIAGNÓSTICO n_preprocess_components (hipótesis crítica)")
print("  floor=40 en n_pre es perjudicial cuando nc<<d ?")
print("="*70)
X_sr0, _ = make_swiss_roll(600, noise=0.02, random_state=42)
X0 = StandardScaler().fit_transform(np.hstack([X_sr0, np.random.randn(600, 50)*0.2]))
print(f"\n  Swiss Roll R^53 → nc=2  (n_pre actual = max(2*3,40)=40)")
print(f"  {'n_pre':>6}  {'LCMC':>8}")
print(f"  {'─'*20}")
best_pre_r = {}
for n_pre in [4, 6, 8, 10, 12, 15, 20, 30, 40]:
    np_eff = min(n_pre, 52)
    prep = SpectralPreprocessor(n_components=np_eff, whiten=False).fit(X0)
    Zp = prep.transform(X0)
    diff = DiffusionGeometry(n_components=2, n_neighbors=20, alpha=1.0).fit(Zp)
    Z = diff.transform(Zp)
    *_, lc, _ = lcmc_metrics(X0, Z, k=10, n_sample=200)
    best_pre_r[np_eff] = lc
    mark = " ← actual" if n_pre == 40 else ""
    print(f"  {np_eff:>6}  {lc:8.4f}{mark}")

# PCA baseline
Z_pca0 = PCA(n_components=2).fit_transform(X0)
*_, lc_pca0, _ = lcmc_metrics(X0, Z_pca0, k=10, n_sample=200)
print(f"  {'PCA(2)':>6}  {lc_pca0:8.4f}  ← baseline lineal")
best_npre = max(best_pre_r, key=best_pre_r.get)
print(f"\n  Óptimo: n_pre={best_npre} → LCMC={best_pre_r[best_npre]:.4f}")
print(f"  Actual: n_pre=40 → LCMC={best_pre_r.get(40,0):.4f}")
print(f"  PCA directa: LCMC={lc_pca0:.4f}")

# ─── FASE 1: Manifolds no lineales ──────────────────────────────────────────
print("\n" + "="*70)
print("  FASE 1: MANIFOLDS NO LINEALES EN ALTA DIMENSIÓN")
print("="*70)
METHODS_F1 = ['CCDEngine', 'PCA', 'KernelPCA', 'Isomap', 'UMAP', 'SpectralEmb']
f1_results = {}
datasets_f1 = [
    ('Swiss Roll R^3 limpio',    make_swiss_roll(600, noise=0.01, random_state=42)[0], 0, 0.0, 2),
    ('Swiss Roll R^15 12ruido',  make_swiss_roll(600, noise=0.02, random_state=42)[0], 12, 0.2, 2),
    ('Swiss Roll R^53 50ruido',  make_swiss_roll(600, noise=0.02, random_state=42)[0], 50, 0.2, 2),
    ('S-Curve  R^53 50ruido',    make_s_curve(600, noise=0.02, random_state=42)[0],    50, 0.2, 2),
]
for ds_name, base, n_noise, ns, nc in datasets_f1:
    if n_noise > 0:
        X = StandardScaler().fit_transform(np.hstack([base, np.random.randn(len(base), n_noise)*ns]))
    else:
        X = StandardScaler().fit_transform(base)
    results = {m: run_method(m, X, nc) for m in METHODS_F1}
    table(results, f"{ds_name}  {X.shape} → {nc}D")
    f1_results[ds_name] = results

# ─── FASE 2: Robustez ruido ──────────────────────────────────────────────────
print("\n" + "="*70)
print("  FASE 2: ROBUSTEZ AL RUIDO (Swiss Roll R^53, nc=2)")
print("="*70)
METHODS_F2 = ['CCDEngine', 'PCA', 'UMAP', 'Isomap']
X_sr2, _ = make_swiss_roll(600, noise=0.01, random_state=42)
print(f"\n  {'scale':>7} {'SNR':>8}  {'CCD':>10} {'PCA':>10} {'UMAP':>10} {'Isomap':>10}  Ganador")
print("  " + "─"*68)
for ns in [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]:
    noise = np.random.randn(600, 50) * ns
    X = StandardScaler().fit_transform(np.hstack([X_sr2, noise]))
    snr = 10*np.log10(np.var(X_sr2)/(ns**2+1e-10)) if ns>0 else np.inf
    row = {m: run_method(m, X, 2)['lcmc'] for m in METHODS_F2}
    winner = max(row, key=lambda k: row[k] if not np.isnan(row[k]) else -1)
    snr_s = f"{snr:8.1f}dB" if np.isfinite(snr) else "      ∞dB"
    vals = "".join(f"  {row[m]:10.4f}" if not np.isnan(row[m]) else "        FAIL" for m in METHODS_F2)
    print(f"  ns={ns:<4} {snr_s}{vals}  {winner}")

# ─── FASE 3: Escalabilidad ──────────────────────────────────────────────────
print("\n" + "="*70)
print("  FASE 3: ESCALABILIDAD")
print("="*70)
METHODS_F3 = ['CCDEngine', 'PCA', 'UMAP']
print("\n  3A. Escalando d (n=500, señal SR en primeras 3 dims)")
print(f"  {'d':>5}  {'CCD t':>8} {'PCA t':>8} {'UMAP t':>8}  {'CCD LCMC':>10} {'UMAP LCMC':>10}  Ganador")
print("  " + "─"*68)
for d in [10, 50, 100, 200, 500]:
    base_d, _ = make_swiss_roll(500, noise=0.02, random_state=42)
    nc_d = d - 3 if d > 3 else 0
    X_d = StandardScaler().fit_transform(
        np.hstack([base_d, np.random.randn(500, nc_d)*0.2]) if nc_d > 0 else base_d)
    rs = {m: run_method(m, X_d, 2) for m in METHODS_F3}
    lcs = {m: rs[m]['lcmc'] for m in METHODS_F3}
    winner = max(lcs, key=lambda k: lcs[k] if not np.isnan(lcs[k]) else -1)
    print(f"  d={d:<4} {rs['CCDEngine']['time']:8.3f}s {rs['PCA']['time']:8.3f}s {rs['UMAP']['time']:8.3f}s"
          f"  {lcs['CCDEngine']:10.4f} {lcs['UMAP']:10.4f}  {winner}")

print("\n  3B. Escalando n (d=53, Swiss Roll + 50 ruido)")
print(f"  {'n':>6}  {'CCD t':>8} {'PCA t':>8} {'UMAP t':>8}  {'CCD LCMC':>10} {'UMAP LCMC':>10}  Ganador")
print("  " + "─"*68)
for n_s in [200, 500, 1000, 2000]:
    base_n, _ = make_swiss_roll(n_s, noise=0.02, random_state=42)
    X_n = StandardScaler().fit_transform(np.hstack([base_n, np.random.randn(n_s, 50)*0.2]))
    rs = {m: run_method(m, X_n, 2) for m in METHODS_F3}
    lcs = {m: rs[m]['lcmc'] for m in METHODS_F3}
    winner = max(lcs, key=lambda k: lcs[k] if not np.isnan(lcs[k]) else -1)
    print(f"  n={n_s:<5} {rs['CCDEngine']['time']:8.3f}s {rs['PCA']['time']:8.3f}s {rs['UMAP']['time']:8.3f}s"
          f"  {lcs['CCDEngine']:10.4f} {lcs['UMAP']:10.4f}  {winner}")

# ─── FASE 4: Datos reales ────────────────────────────────────────────────────
print("\n" + "="*70)
print("  FASE 4: DATASETS REALES")
print("="*70)
METHODS_F4 = ['CCDEngine', 'PCA', 'KernelPCA', 'Isomap', 'UMAP', 'SpectralEmb', 'TSNE']
f4_results = {}
for ds_name, data_fn, nc in [
    ('Digits 64D→10D',   load_digits,        10),
    ('Wine 13D→5D',      load_wine,           5),
    ('BreastCanc 30D→5D',load_breast_cancer,  5),
]:
    data = data_fn()
    X = StandardScaler().fit_transform(data.data)
    y = data.target
    results = {m: run_method(m, X, nc, y=y) for m in METHODS_F4}
    table(results, f"{ds_name}  {X.shape}")
    f4_results[ds_name] = results

# ─── FASE 5: Invarianza afín ─────────────────────────────────────────────────
print("\n" + "="*70)
print("  FASE 5: INVARIANZA AFÍN")
print("="*70)
base5, _ = make_swiss_roll(400, noise=0.02, random_state=42)
X_b5 = np.hstack([base5, np.random.randn(400, 12)*0.15])
METHODS_F5 = ['CCDEngine', 'PCA', 'UMAP']
transforms = {
    'Original':       X_b5,
    'Scaled x10':     X_b5 * 10,
    'Scaled x0.01':   X_b5 * 0.01,
    'Shifted +100':   X_b5 + 100,
    'Rotated':        X_b5 @ np.linalg.qr(np.random.randn(X_b5.shape[1], X_b5.shape[1]))[0],
    'Heterosc':       X_b5 * (1 + np.random.randn(1, X_b5.shape[1])*0.5),
}
print(f"\n  {'Transform':<16}", end='')
for m in METHODS_F5: print(f"  {m:>12}", end='')
print()
print("  " + "─"*58)
scores5 = {m: [] for m in METHODS_F5}
for t_name, X_t in transforms.items():
    X_std = StandardScaler().fit_transform(X_t)
    print(f"  {t_name:<16}", end='')
    for m in METHODS_F5:
        r = run_method(m, X_std, 2)
        v = r['lcmc']
        scores5[m].append(v)
        print(f"  {v:12.4f}" if not np.isnan(v) else "        FAIL", end='')
    print()
print(f"\n  Varianza LCMC:")
for m in METHODS_F5:
    v = np.var([s for s in scores5[m] if not np.isnan(s)])
    print(f"    {m:<13}: var={v:.6f}  {'✅ robusto' if v < 0.001 else '⚠️ inestable'}")

# ─── FASE 6: Fórmula n_pre óptima ───────────────────────────────────────────
print("\n" + "="*70)
print("  FASE 6: FÓRMULA ÓPTIMA PARA n_preprocess_components")
print("="*70)
configs_f6 = [
    ('Swiss Roll R^53  nc=2',  make_swiss_roll(600, noise=0.02, random_state=42)[0], 50, 0.2, 2),
    ('Swiss Roll R^103 nc=2',  make_swiss_roll(600, noise=0.02, random_state=42)[0], 100, 0.2, 2),
    ('S-Curve  R^53    nc=2',  make_s_curve(600,   noise=0.02, random_state=42)[0],  50, 0.2, 2),
    ('Digits   64D    nc=10',  None, 0, 0.0, 10),
]
for cfg_name, base6, n_noise6, ns6, nc6 in configs_f6:
    if base6 is None:
        data6 = load_digits()
        X6 = StandardScaler().fit_transform(data6.data)
    elif n_noise6 > 0:
        X6 = StandardScaler().fit_transform(np.hstack([base6, np.random.randn(len(base6), n_noise6)*ns6]))
    else:
        X6 = StandardScaler().fit_transform(base6)
    d6 = X6.shape[1]
    print(f"\n  {cfg_name}  d={d6}")
    print(f"  {'Fórmula':<28} {'n_pre':>6} {'LCMC':>8}")
    print(f"  {'─'*46}")
    formulas = [
        ('max(nc*3, 40) [ACTUAL]', lambda nc,d: min(d-1, max(nc*3, 40))),
        ('max(nc*4, 15)',          lambda nc,d: min(d-1, max(nc*4, 15))),
        ('max(nc*5, 12)',          lambda nc,d: min(d-1, max(nc*5, 12))),
        ('max(nc*8, 10)',          lambda nc,d: min(d-1, max(nc*8, 10))),
    ]
    for fname, formula in formulas:
        n_pre6 = formula(nc6, d6)
        prep6 = SpectralPreprocessor(n_components=n_pre6, whiten=False).fit(X6)
        Zp6 = prep6.transform(X6)
        diff6 = DiffusionGeometry(n_components=nc6, n_neighbors=20, alpha=1.0).fit(Zp6)
        Z6 = diff6.transform(Zp6)
        k6 = min(15, len(Z6)//10, len(Z6)-1)
        *_, lc6, _ = lcmc_metrics(X6, Z6, k=max(k6,5), n_sample=200)
        print(f"  {fname:<28} {n_pre6:>6} {lc6:8.4f}")
    Z_pca6 = PCA(n_components=min(nc6, d6, len(X6)-1)).fit_transform(X6)
    *_, lc_p6, _ = lcmc_metrics(X6, Z_pca6, k=max(min(15, len(Z_pca6)//10),5), n_sample=200)
    Z_umap6 = umap_lib.UMAP(n_components=nc6, random_state=42).fit_transform(X6)
    *_, lc_u6, _ = lcmc_metrics(X6, Z_umap6, k=max(min(15, len(Z_umap6)//10),5), n_sample=200)
    print(f"  {'PCA (baseline)':<28} {'—':>6} {lc_p6:8.4f}")
    print(f"  {'UMAP (baseline)':<28} {'—':>6} {lc_u6:8.4f}")

# ─── RESUMEN ─────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  RESUMEN EJECUTIVO — VEREDICTO FINAL")
print("="*70)
all_res = {**f1_results, **f4_results}
wins = 0; total = 0
for ds, res in all_res.items():
    if 'CCDEngine' not in res: continue
    ccd_l = res['CCDEngine'].get('lcmc', np.nan)
    if np.isnan(ccd_l): continue
    rivals = [r['lcmc'] for n,r in res.items()
              if n != 'CCDEngine' and not np.isnan(r.get('lcmc', np.nan))]
    if rivals:
        total += 1
        if ccd_l >= max(rivals): wins += 1
pct = wins/total*100 if total>0 else 0
print(f"\n  Victorias 1er lugar: {wins}/{total}  ({pct:.0f}%)")
print(f"  {'✅ COMPETITIVO vs estado del arte' if pct>=50 else '⚠️ Necesita mejoras'}")
print(f"\n  Tiempo total: {time.perf_counter()-t_global:.1f}s")
print("="*70)
