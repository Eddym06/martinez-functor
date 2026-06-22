"""
phi_neuron.py — Neurona Φ v2.0: Poda L1 + Block-Diag RLS + Batch SVD
======================================================================

v2.0: 1) PODA L1 en cada paso RLS → solo k modos activos
      2) BLOCK-DIAG P por grado → O(n·d²) en vez de O(N²)
      3) BATCH SVD init → ridge regression con SVD truncada

Autor: AXIOM-1
"""

import math, time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from .acf_tensorial_neuron import evaluate_multivariate_chebyshev, TensorialMode


@dataclass
class PhiPrediction:
    mean: np.ndarray; std: np.ndarray; variance: np.ndarray
    @property
    def is_reliable(self) -> bool:
        mag = np.linalg.norm(self.mean)
        if mag < 1e-10: return float(np.max(self.std)) < 1.0
        return float(np.max(self.std) / max(mag, 1e-10)) < 2.0
    def confidence_interval(self, dim: int, k: float = 2.0) -> Tuple[float, float]:
        mu, s = float(self.mean[dim]), float(self.std[dim])
        return (mu - k * s, mu + k * s)
    def __repr__(self):
        r = "✓" if self.is_reliable else "⚠"
        return f"Φ({r} μ={self.mean[:3]}..., σ_max={np.max(self.std):.3f})"


class PhiNeuron:
    """Neurona Φ v2.0 — Poda L1 + Block-Diag RLS + Batch SVD."""

    def __init__(self, name: str, n_input: int, n_output: int,
                 max_degree: int = 3, l2_lambda: float = 0.5,
                 l1_lambda: float = 0.01,
                 domain_lo: Optional[np.ndarray] = None,
                 domain_hi: Optional[np.ndarray] = None,
                 spectral_threshold: float = 0.01,
                 alpha_expand_threshold: float = 0.5,
                 correlation_threshold: float = 0.15):
        self.name = name; self.n_input = n_input; self.n_output = n_output
        self.max_degree = max_degree
        self.domain_lo = domain_lo if domain_lo is not None else np.full(n_input, -2.0)
        self.domain_hi = domain_hi if domain_hi is not None else np.full(n_input, 2.0)
        self.spectral_threshold = spectral_threshold
        self.alpha_expand_threshold = alpha_expand_threshold
        self.correlation_threshold = correlation_threshold
        self.l2_lambda = l2_lambda; self.l1_lambda = l1_lambda

        self.current_degree = 1
        self._indices = self._build_indices(degree=1)
        self.n_basis = len(self._indices)
        self._expansion_history: List[Dict] = []
        self._n_samples_seen = 0

        self.C_mu = np.zeros((n_output, self.n_basis))
        self.C_sigma = np.zeros((n_output, self.n_basis))

        self._block_map = self._build_block_map()
        self.P_blocks = [np.eye(len(b)) / l2_lambda for b in self._block_map]
        self.forgetting_factor = 0.995

        self.modes_mu: List[TensorialMode] = []
        self.modes_sigma: List[TensorialMode] = []
        self.epsilon_mu = float("inf"); self.epsilon_sigma = float("inf")
        self.alpha_A = 0.0; self.total_updates = 0
        self._error_window: List[float] = []
        self._l1_active_count = self.n_basis

    # ── Block-diag ────────────────────────────────────────────────

    def _build_block_map(self) -> List[List[int]]:
        blocks = {0: [], 1: [], 2: [], 3: []}
        for k, idx in enumerate(self._indices):
            deg = min(sum(idx), 3); blocks[deg].append(k)
        return [blocks[d] for d in [0, 1, 2, 3] if blocks[d]]

    def _rebuild_block_map(self):
        self._block_map = self._build_block_map()
        self.P_blocks = [np.eye(len(b)) / self.l2_lambda for b in self._block_map]

    def _block_rls_update(self, psi, err_mu, err_sigma):
        for bi, block in enumerate(self._block_map):
            if not block: continue
            Pb = self.P_blocks[bi]; psib = psi[block]; Ppsi = Pb @ psib
            denom = self.forgetting_factor + float(psib @ Ppsi)
            if denom < 1e-16: continue
            kg = Ppsi / denom
            for j in range(self.n_output):
                self.C_mu[j, block] += kg * err_mu[j]
                self.C_sigma[j, block] += kg * err_sigma[j]
            self.P_blocks[bi] = (Pb - np.outer(kg, Ppsi)) / self.forgetting_factor

    # ── L1 ────────────────────────────────────────────────────────

    @staticmethod
    def _soft_threshold(C, lam):
        return np.sign(C) * np.maximum(np.abs(C) - lam, 0.0)

    def _apply_l1_pruning(self):
        self.C_mu = self._soft_threshold(self.C_mu, self.l1_lambda)
        self.C_sigma = self._soft_threshold(self.C_sigma, self.l1_lambda)
        active = np.any(np.abs(self.C_mu) > 1e-16, axis=0) | np.any(np.abs(self.C_sigma) > 1e-16, axis=0)
        self._l1_active_count = int(np.sum(active))

    # ── Batch SVD ─────────────────────────────────────────────────

    def _batch_svd_init(self, X, Y):
        n_use = min(len(X), 800)
        idx = np.random.choice(len(X), n_use, replace=False)
        Xs, Ys = X[idx], Y[idx]
        Phi = np.zeros((n_use, self.n_basis))
        for i in range(n_use):
            Phi[i] = evaluate_multivariate_chebyshev(Xs[i], self._indices, self.domain_lo, self.domain_hi)
        try:
            U, s, Vt = np.linalg.svd(Phi, full_matrices=False)
            s2 = s**2; inv_s = s / (s2 + self.l2_lambda)
            self.C_mu = ((Vt.T * inv_s) @ (U.T @ Ys)).T
            residuals = np.abs(Ys - (self.C_mu @ Phi.T).T)
            self.C_sigma = np.maximum(((Vt.T * inv_s) @ (U.T @ residuals)).T, 0)
            P_full = (Vt.T / (s2 + self.l2_lambda)[:, None]) @ Vt
            self.P_blocks = [P_full[np.ix_(b, b)] for b in self._block_map]
        except np.linalg.LinAlgError:
            pass

    # ── Base ──────────────────────────────────────────────────────

    def _build_indices(self, degree, pairs=None):
        idxs = [tuple([0] * self.n_input)]
        for i in range(self.n_input):
            ix = [0]*self.n_input; ix[i]=1; idxs.append(tuple(ix))
        if degree >= 2:
            if pairs is None:
                for i in range(self.n_input):
                    for j in range(i, self.n_input):
                        ix = [0]*self.n_input; ix[i]+=1
                        if i==j: ix[i]+=1
                        else: ix[j]+=1
                        t = tuple(ix)
                        if t not in idxs: idxs.append(t)
            else:
                for i,j in pairs:
                    ix = [0]*self.n_input; ix[i]+=1
                    if i==j: ix[i]+=1
                    else: ix[j]+=1
                    t = tuple(ix)
                    if t not in idxs: idxs.append(t)
        if degree >= 3 and pairs:
            for i,j in pairs:
                ix = [0]*self.n_input; ix[i]+=1
                ix[j] += 2 if i==j else 1
                t = tuple(ix)
                if t not in idxs: idxs.append(t)
        return idxs

    def _corr_pairs(self, X):
        if len(X) < 5: return []
        corr = np.abs(np.corrcoef(X.T)); pairs = []
        for i in range(self.n_input):
            for j in range(i, self.n_input):
                if i!=j and corr[i,j] > self.correlation_threshold: pairs.append((i,j))
                elif i==j and np.std(X[:,i]) > 0.1: pairs.append((i,i))
        return pairs

    def _try_expand(self, X):
        if self.current_degree >= self.max_degree: return False
        if self.total_updates < 50: return False
        ns = len(X); ms = max(ns//3, self.n_input+1)
        if self.n_basis >= ms: return False
        if self.alpha_A > self.alpha_expand_threshold: return False
        pairs = self._corr_pairs(X)
        if pairs:
            cm = np.abs(np.corrcoef(X.T))
            sc = [(cm[i,j],(i,j)) for i,j in pairs if i!=j]; sc.sort(reverse=True)
            mp = min(len(sc), self.n_input*3)
            pairs = [p for _,p in sc[:mp]]
            for i in range(self.n_input):
                if np.std(X[:,i])>0.15: pairs.append((i,i))
        if not pairs:
            self.current_degree = min(self.current_degree+1, self.max_degree)
            return False
        nd = min(self.current_degree+1, self.max_degree)
        ni = self._build_indices(nd, pairs)
        nn = len(ni)-len(self._indices)
        if nn<=0 or len(ni)>ms: self.current_degree=nd; return False
        on = len(self._indices); oc = self.C_mu.copy(); os_ = self.C_sigma.copy()
        self._indices=ni; self.n_basis=len(ni)
        nc = np.zeros((self.n_output,self.n_basis)); ns_ = np.zeros((self.n_output,self.n_basis))
        nc[:,:on]=oc; ns_[:,:on]=os_; self.C_mu=nc; self.C_sigma=ns_
        self._rebuild_block_map(); self.current_degree=nd
        self._expansion_history.append({"step":self.total_updates,"deg":nd,"before":on,"after":self.n_basis,"new":nn})
        return True

    # ── API ───────────────────────────────────────────────────────

    def observe_and_learn(self, x, y_target):
        x=np.asarray(x,np.float64).ravel(); y=np.asarray(y_target,np.float64).ravel()
        psi = evaluate_multivariate_chebyshev(x, self._indices, self.domain_lo, self.domain_hi)
        mu = self.C_mu @ psi; ae = np.abs(y - mu)
        self._block_rls_update(psi, y-mu, ae - self.C_sigma@psi)
        self.total_updates+=1; self._n_samples_seen+=1
        self._error_window.append(float(np.mean(ae)))
        if len(self._error_window)>100: self._error_window.pop(0)
        if self.total_updates%50==0:
            self._apply_l1_pruning(); self._prune_modes(); self._update_metrics()
        sp = np.maximum(self.C_sigma@psi, 1e-8)
        return PhiPrediction(mean=mu, std=sp, variance=sp**2)

    def predict(self, x):
        x=np.asarray(x,np.float64).ravel()
        psi = evaluate_multivariate_chebyshev(x, self._indices, self.domain_lo, self.domain_hi)
        sp = np.maximum(self.C_sigma@psi, 1e-8)
        return PhiPrediction(mean=self.C_mu@psi, std=sp, variance=sp**2)

    def evaluate(self, x): return self.predict(x).mean

    def fit(self, X, Y, epochs=20, auto_expand=True):
        t0=time.perf_counter(); errors=[]; expansions=[]
        needs_batch = self.n_basis > self.n_input + 5 and len(X) > self.n_basis

        if needs_batch:
            self._batch_svd_init(X, Y)
            self._apply_l1_pruning(); self._prune_modes(); self._update_metrics()
            for ep in range(epochs):
                if auto_expand and self._try_expand(X):
                    expansions.append(ep)
                    self._batch_svd_init(X, Y)
                    self._apply_l1_pruning()
                else:
                    break
            preds = np.array([self.evaluate(x) for x in X[:min(200, len(X))]])
            errors = [float(np.mean((preds - Y[:min(200, len(X))])**2))]
        else:
            for ep in range(epochs):
                idx=np.random.permutation(len(X)); ee=0.0
                for i in idx:
                    p=self.observe_and_learn(X[i],Y[i]); ee+=float(np.mean(np.abs(Y[i]-p.mean)))
                errors.append(ee/len(idx))
                if auto_expand and self._try_expand(X):
                    expansions.append(ep)
                    if self.n_basis > self.n_input + 5 and len(X) > self.n_basis:
                        self._batch_svd_init(X, Y)
                        self._apply_l1_pruning()
                        break  # batch init es óptimo, no seguir con RLS
            self._apply_l1_pruning()

        elapsed=time.perf_counter()-t0
        self._prune_modes()
        return {"time":elapsed,"errors":errors,"final_error":errors[-1],"expansions":expansions,
                "final_degree":self.current_degree,"n_basis":self.n_basis,"l1_active":self._l1_active_count,
                "mu_modes":len(self.modes_mu),"sigma_modes":len(self.modes_sigma),
                "epsilon_mu":self.epsilon_mu,"total_updates":self.total_updates}

    def _prune_modes(self):
        self.modes_mu=[]; self.modes_sigma=[]
        for k,mi in enumerate(self._indices):
            sm=float(np.linalg.norm(self.C_mu[:,k])); ss=float(np.linalg.norm(self.C_sigma[:,k]))
            if sm>self.spectral_threshold: self.modes_mu.append(TensorialMode(multi_index=mi,coefficient=sm,output_vector=self.C_mu[:,k]/max(sm,1e-16)))
            if ss>self.spectral_threshold: self.modes_sigma.append(TensorialMode(multi_index=mi,coefficient=ss,output_vector=self.C_sigma[:,k]/max(ss,1e-16)))
        self.modes_mu.sort(key=lambda m:m.coefficient,reverse=True)
        self.modes_sigma.sort(key=lambda m:m.coefficient,reverse=True)

    def _update_metrics(self):
        if len(self._error_window)>=10: self.epsilon_mu=float(np.mean(self._error_window[-20:]))
        if self.modes_mu:
            sigs=np.array([m.coefficient for m in self.modes_mu]); sigs=sigs[sigs>1e-16]
            if len(sigs)>=2: self.alpha_A=float(max(0.0,-np.polyfit(np.arange(len(sigs)),np.log(sigs),1)[0]))

    def detect_ood(self, x):
        x=np.asarray(x,np.float64).ravel()
        for i in range(self.n_input):
            if x[i]<self.domain_lo[i] or x[i]>self.domain_hi[i]: return True,f"Dim {i}: {x[i]:.3f}"
        p=self.predict(x)
        if not p.is_reliable: return True,f"σ_max={np.max(p.std):.3f}"
        return False,""

    @property
    def self_knowledge(self):
        return {"name":self.name,"n_input":self.n_input,"n_output":self.n_output,
                "degree":self.current_degree,"max_degree":self.max_degree,
                "n_basis":self.n_basis,"l1_active":self._l1_active_count,
                "mu_modes":len(self.modes_mu),"sigma_modes":len(self.modes_sigma),
                "epsilon_mu":self.epsilon_mu,"alpha_A":self.alpha_A,
                "total_updates":self.total_updates,"expansions":len(self._expansion_history)}

    def summary(self):
        sk=self.self_knowledge
        return (f"Φ v2.0('{self.name}') R^{sk['n_input']}→R^{sk['n_output']} | "
                f"gr{sk['degree']}/{sk['max_degree']} | base={sk['n_basis']} | "
                f"L1={sk['l1_active']}/{sk['n_basis']} | μ={sk['mu_modes']} σ={sk['sigma_modes']} | "
                f"ε={sk['epsilon_mu']:.2e} α={sk['alpha_A']:.3f}")
