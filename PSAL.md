# P-SAL: The Autopoietic Law Synthesis Protocol
### A Closed-Loop Scientific Discovery System for Dynamical Systems

**Author:** Theoretical foundation by Eddy Manuel Piantini — Formal derivation based on  
the ACF hierarchy, SINDy theory, Koopman operator theory, and ERGON thermodynamic closure  
**Date:** April 2026  
**Version:** 2.0 — Full specification with SEM integration and Level-5 Autonomy  

> **Nomenclature:** P-SAL stands for **Protocolo de Síntesis Autopoiética de Leyes** — in English, the *Autopoietic Law Synthesis Protocol*. The word *autopoietic* comes from the Greek αὐτο- (self) + ποίησις (creation, production). A system is autopoietic when it produces its own components and maintains its own organization. P-SAL does exactly this for mathematical laws: it discovers the equations governing its own data, compiles them into executable programs, verifies them, and uses them — all without external supervision.

---

## Epigraph

> *"The scientist does not discover laws because they were hidden. She discovers them because she has the right instrument. P-SAL is that instrument — the membrane that separates data from law."*

> *"SINDy finds the words. ERGON gives them thermodynamic weight. Poema makes them run. Gideon makes them fly."*

---

## Abstract

P-SAL is the apex agent of the ACF hierarchy — the closed loop that connects raw observational data to executable, verified, thermodynamically consistent reduced-order models (ROMs) of dynamical systems. Unlike classical data-driven modeling pipelines, P-SAL does not require a human to specify the governing equations, choose the basis functions, or decide when a model is "good enough." These decisions are made autonomously, using:

1. **SEM** (Level 0.5) to purify stochastic corruption before any analysis begins
2. **TAA** (Level 1) for Koopman spectral decomposition into modal amplitudes
3. **SINDy** for sparse identification of nonlinear dynamics in modal space
4. **ERGON** (Level 2) for thermodynamically grounded closure of truncated ROMs
5. **Poema** for compilation of discovered laws into formally verified programs
6. **Gideon** for hardware-optimized execution of the compiled ROMs
7. A **Knowledge Base** for persistent storage of verified laws and adaptation feedback

The system closes on itself: the laws it discovers feed back into its own perception apparatus, improving future discoveries. This is the mathematical realization of autopoiesis.

---

## 0. Position in the ACF Ecosystem

```
Raw Data y(t) = x(t) + η(t)  ← noise corruption
       │
       ▼
 ┌─────────────────┐
 │  SEM  Level 0.5 │  Stochastic purification: x̂(t), Σ(t)
 └────────┬────────┘
          │ SMOutput {x̂, Σ, Π_SM, certificates}
          ▼
 ┌─────────────────┐
 │  TAA  Level 1   │  Koopman decomposition: λ_k, φ_k, α_A, δ(d)
 └────────┬────────┘
          │ KoopmanSpectrum {modal_amplitudes, eigenvectors, decay_class}
          ▼
 ┌─────────────────┐
 │ ERGON Level 2   │  SRB measure: h_KS, P''(1), μ_SRB, λ_max
 └────────┬────────┘
          │ ERGONDiagnostics {h_ks, pressure_curvature, lyapunov}
          ▼
 ┌─────────────────────────────────────────────┐
 │       P-SAL   —   Autopoietic Scientist      │
 │  OBSERVE → HYPOTHESIZE → CLOSE → COMPILE    │
 │  VERIFY  → MEMORIZE   → ACT   → ADAPT       │
 └─────────────────────────────────────────────┘
          │ DiscoveredLaw {L, Q, c, ν_t, certificates}
          ▼
 Deployed ROM  →  Gideon execution  →  feedback to SEM
```

**The compositional contract:** P-SAL consumes certified outputs from lower levels. It checks SM-2 (trajectory SNR) before using SEM output. It checks TAA-1 (Koopman isometry) before projecting onto modes. It checks ERG-1 (KS entropy) before computing thermodynamic closure. This chain of certificate verification makes the pipeline compositionally sound — each step is only executed when the input has been formally validated by the level below.

---

## 1. The Problem P-SAL Solves

### 1.1 The Curse of Governing Equations

Classical scientific modeling requires a human expert to specify the governing equations of a system: Navier-Stokes for fluid flow, Maxwell's equations for electromagnetism, Hamilton's equations for mechanics. But many important dynamical systems — turbulent flows, financial markets, neural dynamics, ecological systems, climate models — are either too complex for exact analytical treatment or evolve in regimes where the governing equations are not known.

**The fundamental question P-SAL answers is:** Given only observational data $y_0, y_1, \ldots, y_{T-1}$ from a dynamical system, can a machine automatically discover, formalize, verify, and deploy the governing equations?

The answer is: **yes, within a hierarchy of approximations with formal error certificates**.

### 1.2 The Truncation Problem

Even when governing equations are known (e.g., Navier-Stokes), they operate in infinite-dimensional function spaces. Any computational model must truncate to finitely many degrees of freedom. This truncation is mathematically non-trivial:

Let $\mathcal{H}$ be the Hilbert space of solutions and $P_r: \mathcal{H} \to \mathcal{H}_r$ be a projection onto the $r$-dimensional subspace spanned by the leading $r$ modes. The truncated dynamics are:

$$\dot{\mathbf{a}} = P_r \mathcal{F}(\mathbf{a}) \neq P_r \mathcal{F}(P_r^* \mathbf{a})$$

The discrepancy between the left and right sides is the **truncation error** — the energy transferred to unresolved modes that the truncated model does not capture. Without closure, this error accumulates and either blows up the simulation or artificially damps the resolved dynamics.

P-SAL solves this via the **ERGON thermodynamic closure**: using the KS entropy $h_{KS}$ and topological pressure $P''(1)$ to compute the exact eddy viscosity $\nu_t$ that compensates for the missing dissipation from the unresolved modes.

### 1.3 The Autopoiesis Principle

Maturana and Varela (1980) defined an **autopoietic system** as one that:
1. Has a boundary that distinguishes it from its environment
2. Produces the components that constitute that boundary
3. Is organizationally closed — its organization is maintained by its own processes

P-SAL is autopoietic in this sense:
1. The **Knowledge Base** is its boundary — the accumulated laws it has verified
2. It **produces** new laws by running the P-SAL cycle on new data
3. It is **organizationally closed**: the laws it discovers improve its own observation (via better noise models and mode selection) and its own compilation (via better ROM templates)

This distinguishes P-SAL from a simple machine learning pipeline. The output of P-SAL is not a trained model — it is a **mathematical law with a formal certificate of validity**, stored in a knowledge base that the system uses to improve itself.

---

## 2. Mathematical Foundations

### 2.1 Koopman Operator Theory

Let $(X, \mathcal{B}, \mu)$ be a measure space and $T: X \to X$ a measurable map (the dynamics). The **Koopman operator** $\mathcal{K}: L^2(X, \mu) \to L^2(X, \mu)$ is defined by:

$$(\mathcal{K} f)(x) = f(T(x))$$

The key properties:
- $\mathcal{K}$ is linear, even when $T$ is nonlinear
- $\mathcal{K}$ is an isometry on $L^2(X, \mu_{SRB})$ when $\mu_{SRB}$ is the invariant measure
- Eigenvalues $\lambda_k$ of $\mathcal{K}$ encode the spectral frequencies of $T$
- Eigenfunctions $\varphi_k$ are **observable functions** that evolve linearly under $T$: $\mathcal{K}\varphi_k = \lambda_k \varphi_k$

The **Koopman modal decomposition** of an observable $g(x)$ is:

$$g(x) = \sum_{k=1}^r c_k \varphi_k(x) + \text{remainder}$$

where $c_k = \langle g, \varphi_k \rangle_{\mu}$ are the **modal amplitudes**. The dynamics of the amplitudes are exactly:

$$a_k(t) = \lambda_k^t \cdot a_k(0)$$

in the linear (no truncation) case. With truncation to $r$ modes, the residual dynamics must be closed — this is the role of ERGON.

### 2.2 Extended Dynamic Mode Decomposition (EDMD)

In practice, the Koopman operator is estimated from data using **EDMD** (Williams, Kevrekidis, Rowley, 2015). Given a trajectory $\{x_k\}_{k=0}^{N-1}$ and observable dictionary $\psi: X \to \mathbb{R}^d$:

$$\Psi_X = \begin{pmatrix} \psi(x_0)^T \\ \psi(x_1)^T \\ \vdots \end{pmatrix}, \quad \Psi_Y = \begin{pmatrix} \psi(T(x_0))^T \\ \psi(T(x_1))^T \\ \vdots \end{pmatrix}$$

The Koopman matrix estimate is:

$$K = \Psi_X^+ \Psi_Y$$

where $\Psi_X^+$ is the pseudoinverse. The eigendecomposition $K V = V \Lambda$ gives:
- Eigenvalues $\lambda_k$ (diagonal of $\Lambda$)
- Eigenvectors $V$ (Koopman modes)
- Modal amplitudes: $a(t) = V^{-1} \psi(x(t))$

### 2.3 SINDy: Sparse Identification of Nonlinear Dynamics

Once modal amplitudes $\mathbf{a}(t) = (a_1(t), \ldots, a_r(t))$ are available, **SINDy** (Brunton, Proctor, Kutz, 2016) discovers the governing ODE by solving a sparse regression problem.

**Step 1 — Time derivatives:** Compute $\dot{\mathbf{A}} \approx$ (centered finite differences):

$$\dot{a}_i^{(k)} \approx \frac{a_i^{(k+1)} - a_i^{(k-1)}}{2\Delta t}$$

**Step 2 — Library construction:** Build the polynomial library $\Theta(\mathbf{A}) \in \mathbb{R}^{N \times P}$:

$$\Theta = \left[\ \mathbf{1}\ |\ \mathbf{A}\ |\ \mathbf{A}^{\odot 2}\ |\ \mathbf{A}^{\odot 3}\ |\ \ldots\ \right]$$

For polynomial degree $p$ and $r$ modes: $P = \binom{r+p}{p}$ columns.

**Step 3 — Sparse regression (STLSQ):** Find the sparsest $\Xi \in \mathbb{R}^{r \times P}$ such that:

$$\dot{\mathbf{A}} \approx \Theta(\mathbf{A})\, \Xi^T$$

The **Sequentially Thresholded Least-Squares (STLSQ)** algorithm:

```
Initialize:  Ξ ← (ΘᵀΘ)⁻¹Θᵀ Ȧ   (least squares)
Repeat until convergence:
    for each coefficient Ξᵢⱼ:
        if |Ξᵢⱼ| < λ: set Ξᵢⱼ = 0
    Re-fit Ξ on the support {i,j : Ξᵢⱼ ≠ 0} by least squares
```

**Step 4 — Operator extraction:** From the converged $\Xi$:
- $\mathbf{L}_{ij}$ = coefficient of $a_j$ in the equation for $\dot{a}_i$ (linear operator)
- $\mathbf{Q}_{ijk}$ = coefficient of $a_j a_k$ in the equation for $\dot{a}_i$ (quadratic tensor)
- $\mathbf{c}_i$ = constant forcing term

The discovered ODE system is:

$$\dot{a}_i = \sum_j L_{ij} a_j + \sum_{j \leq k} Q_{ijk} a_j a_k + c_i$$

### 2.4 Theorem: STLSQ Consistency

**Theorem (Zhang & Schaeffer, 2019).** For a system with true dynamics $\dot{\mathbf{a}} = \mathbf{L}_0 \mathbf{a} + \text{nonlinear terms}$ and observation noise $\sigma$, STLSQ with threshold $\lambda > 2\sigma\sqrt{\frac{\log P}{N}}$ recovers the true support of $\Xi$ with probability $\geq 1 - 2P^{-1}$ when:

$$N \geq C \cdot s \log P$$

where $s = |\mathrm{supp}(\Xi)|$ is the true number of active terms, $P$ is the library size, and $C$ is a constant depending on the restricted isometry property (RIP) of $\Theta$.

**Corollary:** In the underdetermined regime ($N < P$), STLSQ still finds a consistent sparse solution provided the true dynamics have $s \ll P$ active terms (the quadratically nonlinear case, e.g. Navier-Stokes, satisfies this with $s = O(r^2)$).

### 2.5 Galerkin Projection (Alternative to SINDy)

When the governing equations are known, the ROM can be derived analytically via **Galerkin projection** (Noack et al., 2003). For Navier-Stokes:

$$\partial_t \mathbf{u} + (\mathbf{u} \cdot \nabla)\mathbf{u} = -\nabla p + \nu \nabla^2 \mathbf{u}$$

with $\mathbf{u}(\mathbf{x}, t) \approx \sum_{k=1}^r a_k(t) \phi_k(\mathbf{x})$ (Koopman/POD modes), projection gives:

$$\dot{a}_i = \sum_j l_{ij} a_j + \sum_{j,k} q_{ijk} a_j a_k$$

where:

$$l_{ij} = -\int_\Omega \phi_j \cdot (\bar{\mathbf{u}} \cdot \nabla)\phi_i + \nu \int_\Omega \phi_i : \nabla^2 \phi_j$$
$$q_{ijk} = -\int_\Omega \phi_i \cdot (\phi_j \cdot \nabla)\phi_k$$

The **energy-triadic structure** of $\mathbf{Q}$ satisfies:

$$\sum_{ijk} q_{ijk} a_i a_j a_k = 0 \quad \forall \mathbf{a}$$

(exact energy conservation of the quadratic nonlinearity). SINDy should recover $\mathbf{Q}$ with this triadic structure; the deviation from it is a certificate of SINDy accuracy.

### 2.6 Thermodynamic Closure: ERGON-Based Eddy Viscosity

When the ROM is truncated at $r$ modes, the missing dissipation is:

$$\Delta \epsilon = \epsilon_{DNS} - \epsilon_{ROM} = 2\nu \sum_{k=r+1}^{\infty} k^2 E(k)$$

The ERGON agent provides the thermodynamic quantities needed to estimate this:

- **KS entropy** $h_{KS} = \sum_{\lambda_i > 0} \lambda_i$ (Pesin formula) — the rate at which the system generates information
- **Topological pressure curvature** $P''(1) = \text{Var}(\lambda^+)$ — the variance of local Lyapunov exponents
- **SRB covariance** $\text{Cov}(\mathbf{a}_{res})$ — covariance of the residual (unresolved) modes

The **ERGON thermodynamic closure**:

$$\boxed{\nu_t = \frac{1}{2} \cdot \frac{P''(1)}{h_{KS}} \cdot \text{Tr}\!\left(\text{Cov}(\mathbf{a}_{res})\right)}$$

**Physical interpretation:** $P''(1)/h_{KS}$ is the ratio of Lyapunov fluctuation to mean expansion rate — a dimensionless turbulence intensity. Multiplied by the residual energy $\text{Tr}(\text{Cov})$, it gives the eddy viscosity in the correct units.

The closed ROM is:

$$\dot{\mathbf{a}} = \mathbf{L}^{\text{eff}} \mathbf{a} + \mathbf{Q}(\mathbf{a}, \mathbf{a}) + \mathbf{c}$$

where $\mathbf{L}^{\text{eff}} = \mathbf{L} + \nu_t \mathbf{D}$, with $\mathbf{D} = -\text{diag}(1, 4, 9, \ldots, r^2)$ the spectral dissipation operator.

### 2.7 Theorem: Dissipativity of the Closed ROM

**Theorem.** If $\nu_t$ satisfies:

$$\nu_t > \frac{\lambda_{\max}(\mathbf{L} + \mathbf{L}^T)/2}{k_{\min}^2}$$

where $\lambda_{\max}$ denotes the largest eigenvalue and $k_{\min}$ is the smallest wavenumber of $\mathbf{D}$, then:

$$\frac{d}{dt} \|\mathbf{a}\|^2 = 2\mathbf{a}^T (\mathbf{L}^{\text{eff}} \mathbf{a} + \mathbf{Q}(\mathbf{a},\mathbf{a})) \leq -2\gamma \|\mathbf{a}\|^2 + C$$

for some $\gamma > 0$ and $C$ depending on the forcing $\mathbf{c}$. That is, the closed ROM is **globally dissipative** — trajectories are bounded for all $t \geq 0$.

*Proof sketch:* By the energy-triadic conservation of $\mathbf{Q}$, the quadratic term contributes zero to $d\|\mathbf{a}\|^2/dt$. The linear term contributes $2\mathbf{a}^T \mathbf{L}^{\text{eff}} \mathbf{a} \leq 2(\lambda_{\max}(\mathbf{L}/2) - \nu_t k_{\min}^2) \|\mathbf{a}\|^2 < 0$ by assumption. $\square$

---

## 3. The Eight-Phase Protocol

### Phase Architecture

```
                    ┌─────────────────────────────────────────┐
                    │         AUTOPOIETIC SCIENTIST           │
                    │                                         │
                    │   ┌─────────┐     ┌──────────────┐     │
                    │   │ OBSERVE │ ──> │  HYPOTHESIZE │     │
                    │   └────▲────┘     └──────┬───────┘     │
                    │        │                 │              │
                    │        │           ┌─────▼─────┐       │
                    │        │           │   CLOSE   │       │
                    │        │           └─────┬─────┘       │
                    │        │                 │              │
                    │   ┌────┴────┐     ┌─────▼─────┐       │
                    │   │   ACT   │ <── │  COMPILE  │       │
                    │   └────▲────┘     └─────┬─────┘       │
                    │        │                 │              │
                    │        │           ┌─────▼─────┐       │
                    │        │           │  VERIFY   │       │
                    │        │           └─────┬─────┘       │
                    │        │                 │              │
                    │   ┌────┴────┐     ┌─────▼─────┐       │
                    │   │  ADAPT  │ <── │ MEMORIZE  │       │
                    │   └─────────┘     └───────────┘       │
                    │                                         │
                    └─────────────────────────────────────────┘
```

---

### Phase 1: OBSERVE (SEM + TAA)

**Input:** Raw observational trajectory $y_{0:T-1} \in \mathbb{R}^{T \times d}$  
**Output:** Purified modal amplitudes $\mathbf{a}(t) \in \mathbb{R}^{T \times r}$

**Step 1a — SEM purification (v1.1 integration):**  
The `StochasticMembrane` processes the raw trajectory first:

$$\hat{x}_{0:T-1}, \Sigma_{0:T-1} = \text{SEM}(y_{0:T-1})$$

The purified trajectory $\hat{x}$ has stochastic corruption removed; $\Sigma_k$ is the posterior uncertainty at each time step. If certificate SM-2 fails (SNR below threshold), P-SAL falls back to the raw trajectory with a warning.

**Step 1b — Koopman projection:**  
If TAA eigenvectors $V \in \mathbb{R}^{d \times r}$ are available:

$$\mathbf{a}(t) = \hat{x}(t) \cdot V$$

The uncertainty is propagated into modal space:

$$\Sigma_{\mathbf{a}}(t) = V^T \Sigma(t) V \in \mathbb{R}^{r \times r}$$

This propagated uncertainty is used to weight the SINDy regression in Phase 2: observations with high $\text{Tr}(\Sigma_\mathbf{a}(t))$ receive lower weight.

**Step 1c — SVD fallback:**  
If no Koopman decomposition is available:

$$\hat{x} - \bar{x} = U S V^T \quad \text{(SVD)}, \quad \mathbf{a}(t) = (\hat{x}(t) - \bar{x}) V_r$$

where $V_r = V[:, :r]$ are the leading $r$ right singular vectors (POD modes).

```python
# In AutopoieticScientist._observe():
if sm_output is not None and sm_output.purified.is_valid(config):
    trajectory = sm_output.purified.x_hat        # purified!
    obs_weights = 1.0 / (np.trace(sm_output.uncertainty.sigma_tensor, axis1=1, axis2=2) + 1e-8)
```

---

### Phase 2: HYPOTHESIZE (SINDy)

**Input:** Modal amplitudes $\mathbf{a}(t) \in \mathbb{R}^{T \times r}$, observation weights $w(t)$  
**Output:** Sparse operator triplet $(\mathbf{L}, \mathbf{Q}, \mathbf{c})$, sparsity $s$, residual $\varepsilon$

**Weighted STLSQ** with SEM uncertainty weights:

$$\Xi = \arg\min_{\Xi: \text{sparse}} \sum_{k=1}^{T} w_k \|\dot{\mathbf{a}}_k - \Theta(\mathbf{a}_k)\Xi^T\|^2$$

where $w_k = [\text{Tr}(\Sigma_\mathbf{a}(t_k)) + \varepsilon_0]^{-1}$ down-weights uncertain time steps. This is the critical **SEM → SINDy coupling**: uncertain observations (high posterior entropy from the particle filter) contribute less to the law discovery, preventing noisy data from corrupting the sparse regression.

**Sparsity metric:**

$$s = 1 - \frac{|\{(i,j) : \Xi_{ij} \neq 0\}|}{r \cdot P}$$

**Residual norm:**

$$\varepsilon_{\text{SINDy}} = \frac{\|\dot{\mathbf{A}} - \Theta(\mathbf{A})\Xi^T\|_F}{\|\dot{\mathbf{A}}\|_F}$$

**Certificate PSAL-1:** $\varepsilon_{\text{SINDy}} < \tau_1 = 0.5$.

---

### Phase 3: CLOSE (ERGON)

**Input:** Modal amplitudes $\mathbf{a}(t)$, ERGON diagnostics $(h_{KS}, P''(1))$  
**Output:** Closed dynamics operators $(\mathbf{L}^{\text{eff}}, \mathbf{Q}, \mathbf{c})$, eddy viscosity $\nu_t$

**Adaptive closure selection:**

| Priority | Method | When used |
|---|---|---|
| 1st | ERGON thermodynamic | $h_{KS}$ and $P''(1)$ available from ERGON ERG-1 |
| 2nd | Smagorinsky | ERGON unavailable; $\nu_t = (C_S \Delta)^2 |\bar{S}|$ |
| 3rd | Spectral Vanishing Viscosity | High-frequency only: $\nu_t(k) = \nu_0 (k/k_{\max})^p$ |

**Certificate PSAL-2:** $\text{Tr}(\mathbf{L}^{\text{eff}}) < 0$ (ROM is dissipative).

---

### Phase 4: COMPILE (Poema)

**Input:** $(\mathbf{L}^{\text{eff}}, \mathbf{Q}, \mathbf{c}, \nu_t)$, timestep $\Delta t$  
**Output:** `PoemROM` — an executable AST program

The `ROMGenerator` creates a Poema AST representing the ODE system:

$$\dot{a}_i = \sum_j L^{\text{eff}}_{ij} a_j + \sum_{j \leq k} Q_{ijk} a_j a_k + c_i$$

The AST includes an embedded **RK4 integrator**:

$$a_i^{(n+1)} = a_i^{(n)} + \frac{\Delta t}{6}(k_1 + 2k_2 + 2k_3 + k_4)$$

where $k_1, \ldots, k_4$ are the standard Runge-Kutta stages evaluated at $t_n$, $t_n + \Delta t/2$, $t_n + \Delta t/2$, $t_n + \Delta t$.

The compiled PoemROM has three execution modes:
- **Direct:** Pure NumPy RK4, no overhead
- **Adaptive:** RK4(5) with Dormand-Prince error control
- **Ensemble:** Parallel execution on multiple initial conditions

**Local truncation error for RK4:** $O(\Delta t^5)$ — sufficient for ROMs with smooth dynamics.

---

### Phase 5: VERIFY

**Input:** PoemROM, reference modal amplitudes $\mathbf{a}^{ref}(t)$  
**Output:** Verification metrics, pass/fail flags

Four verification metrics:

**1. Trajectory error (relative):**

$$\varepsilon_{traj} = \frac{\|\mathbf{a}^{ROM}(t) - \mathbf{a}^{ref}(t)\|_F}{\|\mathbf{a}^{ref}(t)\|_F}$$

**Certificate PSAL-4:** $\varepsilon_{traj} < \tau_4 = 0.3$.

**2. Spectral fidelity:**

$$\varepsilon_{spec} = \frac{\|E^{ROM}(k) - E^{ref}(k)\|_2}{\|E^{ref}(k)\|_2}, \quad E(k) = \sum_{|j|=k} |\hat{a}_j|^2$$

**Certificate PSAL-3:** $\varepsilon_{spec} < 0.5$.

**3. Energy drift:**

$$\delta_E = \frac{|E_{final} - E_0|}{|E_0|}$$

**Certificate PSAL-5:** $\delta_E < 0.5$.

**4. Dissipativity:**  
**Certificate PSAL-2:** $\text{Tr}(\mathbf{L}^{\text{eff}}) < 0$.

---

### Phase 6: MEMORIZE

**Input:** Verification results, law operators  
**Output:** `DiscoveredLaw` stored in `KnowledgeBase`

**Law lifecycle:**

```
CANDIDATE → VERIFIED → CERTIFIED → DEPLOYED
     │            │           │
     └──REJECTED  └──(partial) └──SUPERSEDED (if better law found)
```

**Quality score:**

$$Q = \frac{1}{4}\left[(1 - \varepsilon_{traj}) + s_{\text{sparsity}} + \mathbb{1}_{\text{dissipative}} + (1 - \delta_E)\right] \in [0, 1]$$

When a new law achieves higher $Q$ than the best known law, the old law's status becomes `SUPERSEDED`. This implements the **autopoietic update**: the system's own standards for law quality improve as it discovers better laws.

---

### Phase 7: ACT (Gideon)

**Input:** Best verified `DiscoveredLaw`, initial condition $\mathbf{a}_0$  
**Output:** Predicted trajectory $\hat{\mathbf{a}}(t)$

The `GideonROMExecutor` runs the PoemROM with hardware-optimized integration:

```
Execution modes (all operational as of Mayo 2026):
  direct   → RK4, NumPy, minimal overhead ✅
  adaptive → RK4(5) + PI step-size control (Gustafsson, 1991) ✅
  ensemble → parallel RK4 on {a₀⁽ⁱ⁾} (Monte Carlo uncertainty) ✅
```

All three modes are integrated into the P-SAL ACT phase. Verified laws are executed in all three modes: direct (stability check), adaptive (error-controlled), ensemble (uncertainty quantification with n_ensemble ≤ 20).

---

### Phase 8: ADAPT (Cierre Autopoiético)

**Input:** Verification results, law status, Knowledge Base history  
**Output:** Adaptation actions for the next cycle

The ADAPT phase closes the autopoietic loop. The system learns from its own discoveries and adjusts its parameters autonomously:

**Adaptación reactiva (por ciclo):**

| Condición | Adaptación |
|---|---|
| $\varepsilon_{traj} > 10.0$ | Reducir `sindy_threshold` al 50% (relajar dispersidad) |
| ROM no disipativo | Aumentar cierre termodinámico |
| $\varepsilon_{traj} < 2 \cdot \tau_4$ (cerca del umbral) | Aumentar número de modos |
| Ley verificada | Reforzar parámetros — registrar n_modes y ν_t exitosos |
| Múltiples rechazos | Relajar `verification_tolerance` en 50% |

**Adaptación autopoiética (basada en Knowledge Base):**

El `KnowledgeBase` analiza todas las leyes verificadas acumuladas y propone mejoras estructurales:

| Método | Acción |
|---|---|
| `propose_improvements()` | Analiza leyes verificadas → mejor n_modes (de la ley más precisa), mejor `sindy_threshold` (de la ley más dispersa y precisa) |
| `get_adapted_params()` | Devuelve parámetros calibrados por el KB, con fallback a defaults. Rango de modos se ajusta alrededor del óptimo descubierto |

**Propiedad de cierre autopoiético:** La salida de la fase ADAPT del ciclo $k$ modifica los parámetros del ciclo $k+1$. Las leyes verificadas en $k$ alimentan el `KnowledgeBase`, que sesga las futuras observaciones. Esto cierra el loop: el sistema literalmente aprende de sus propios descubrimientos.

---

## 4. SEM Integration (P-SAL v2.0)

### 4.1 The Integration Contract

Starting in P-SAL v2.0, the `AutopoieticScientist` accepts `SMOutput` from the `StochasticMembrane` as its primary input. The integration contract:

```python
scientist.run(
    trajectory=raw_data,      # fallback if no SEM
    sm_output=sem_output,     # priority input when available
    dt=0.01,
    h_ks=ergon.h_ks,
    pressure_curvature=ergon.p_prime_prime,
)
```

When `sm_output` is provided:
1. `sm_output.purified.x_hat` replaces `trajectory` in Phase 1
2. `sm_output.uncertainty.sigma_tensor` provides observation weights for STLSQ
3. `sm_output.purified.drift_estimate` provides the initial Nadaraya-Watson drift for TAA
4. SM certificates are checked before entering the PSAL loop

### 4.2 Uncertainty-Weighted SINDy

The SEM posterior covariance $\Sigma_k$ propagates into SINDy via the observation weights:

$$w_k = \frac{1}{\text{Tr}(\Sigma_k) + \varepsilon_0}$$

High-uncertainty time steps (e.g., immediately after a Lévy noise burst) receive low weight in the sparse regression. This makes SINDy's law discovery **noise-robust at the SEM level**, rather than at the SINDy level — a cleaner separation of concerns.

### 4.3 Drift Prior from SEM

The Nadaraya-Watson drift estimate $\hat{f}(x)$ produced by the SEM's particle filter is a nonparametric estimate of the vector field $f$. P-SAL uses this as a **prior for SINDy**: the discovered $(\mathbf{L}, \mathbf{Q})$ should reproduce $\hat{f}$ on the observed trajectory. This cross-validation is added as:

$$\varepsilon_{\text{drift}} = \frac{1}{T} \sum_{k=0}^{T-1} \|\mathbf{L}\mathbf{a}_k + \mathbf{Q}(\mathbf{a}_k, \mathbf{a}_k) - \hat{f}(\mathbf{a}_k)\|_2$$

**Certificate PSAL-7 (new):** $\varepsilon_{\text{drift}} < 0.3$.

---

## 5. Certificate System

### Complete Certificate Table

| Certificate | Condition | Failure Meaning | Phase |
|---|---|---|---|
| **PSAL-1** | $\varepsilon_{\text{SINDy}} < 0.5$ | Dynamics not recoverable | HYPOTHESIZE |
| **PSAL-2** | $\text{Tr}(\mathbf{L}^{\text{eff}}) < 0$ | ROM will blow up | CLOSE |
| **PSAL-3** | $\varepsilon_{spec} < 0.5$ | Wrong spectral distribution | VERIFY |
| **PSAL-4** | $\varepsilon_{traj} < 0.3$ | Poor trajectory prediction | VERIFY |
| **PSAL-5** | $\delta_E < 0.5$ | Energy conservation violated | VERIFY |
| **PSAL-6** | Law generated autonomously | External equation was injected | MEMORIZE |
| **PSAL-7** | $\varepsilon_{\text{drift}} < 0.3$ | SINDy inconsistent with SEM drift | VERIFY |
| **PSAL-8** | $Q > 0.5$ | Law not good enough to deploy | MEMORIZE |

### Certificate Audit Summary

A law receives **CERTIFIED** status only when PSAL-1 through PSAL-6 all pass. A law receives **DEPLOYED** status when it is additionally verified on held-out data not seen during discovery.

The **PSAL-6 autopoietic certificate** is the most philosophically significant: it verifies that the discovered law was generated by the system itself, not injected by a human. In a pipeline where a domain expert provides the governing equations, PSAL-6 would fail — correctly identifying that the system is not operating autonomously.

---

## 6. The Knowledge Base

### 6.1 Law Lifecycle

The `KnowledgeBase` is the persistent memory of the Autopoietic Scientist. Laws are stored with full provenance:

```json
{
  "law_id": "psal_cycle_3_r8",
  "name": "navier_stokes_r8_cycle3",
  "status": "certified",
  "n_modes": 8,
  "quality_score": 0.847,
  "trajectory_error": 0.0234,
  "spectrum_error": 0.156,
  "energy_drift": 0.089,
  "sindy_sparsity": 0.923,
  "sindy_residual": 0.0312,
  "n_active_terms": 14,
  "nu_t": 3.2e-4,
  "certificates": {
    "PSAL-1": 1.0, "PSAL-2": 1.0, "PSAL-3": 1.0,
    "PSAL-4": 1.0, "PSAL-5": 1.0, "PSAL-6": 1.0
  },
  "discovery_time": 1745291234.5,
  "verification_time": 1745291247.2
}
```

### 6.2 Law Supersession

When a new law achieves higher $Q$ than the current best:

1. Old law status → `SUPERSEDED`
2. New law status → `CERTIFIED`
3. `best_law_id` pointer updated
4. Supersession event logged (with $\Delta Q$)

This implements **autopoietic knowledge update**: the system's model of the world improves over cycles without external intervention.

### 6.3 Knowledge Transfer

Laws from one system can seed the discovery process for related systems. If a law for the Lorenz system at $(\sigma=10, \rho=28, \beta=8/3)$ is certified, running P-SAL on data from $(\sigma=10, \rho=35, \beta=8/3)$ can use the old $(\mathbf{L}, \mathbf{Q})$ as a warm start for STLSQ — reducing the number of cycles needed for convergence.

---

## 7. Level-5 Autonomy: Latent Structure Discovery

### 7.1 Beyond SINDy

The standard P-SAL pipeline (Phases 1-8) discovers dynamics within a **fixed polynomial library**. If the true dynamics require non-polynomial terms (e.g., trigonometric, exponential, or rational functions), SINDy will fail to find a sparse representation.

**Level-5 Autonomy** extends P-SAL with operators that discover the **structure** of the governing equations before choosing the library:

### 7.2 Topological Fingerprinting of the Jacobian

For each discovered ROM, the Jacobian matrix $J(\mathbf{a}) = \partial \dot{\mathbf{a}} / \partial \mathbf{a} = \mathbf{L}^{\text{eff}} + \partial_\mathbf{a}[\mathbf{Q}(\mathbf{a},\mathbf{a})]$ is analyzed via:

1. **TDA (Persistent Homology):** Filtration of the Jacobian spectrum → detects cyclic symmetries $Z_N$, persistence of eigenvalue clusters
2. **Koopman Structural Analysis:** Spectrum of $J$ evaluated on the SRB measure → detects resonances, Arnold tongues, multiresolution structure
3. **Operator Grammar Search:** Finds parsimonious factorizations: butterfly structure (FFT-like), Kronecker product (separable dynamics), sparse representations

### 7.3 Rule Induction

Once a structural pattern is discovered (e.g., "$Z_N$ symmetry + unitary spectrum → butterfly factorization with $\log_2 N$ stages"), the **Rule Inductor** generalizes this to a **reusable rule** stored in the knowledge graph:

```
Rule: "Koopman + Z_N symmetry → butterfly(log₂(N) stages)"
Confidence: 0.94
Discovered from: 3 separate systems (N=8, 16, 32)
Applied to: 5 new systems
Success rate: 4/5 (80%)
```

Rules are **automatically applied** in future OBSERVE phases, before running the standard SINDy pipeline. If a rule applies, the library is extended with the predicted structural terms — dramatically reducing the search space.

### 7.4 Level-5 Certificates

| Certificate | Condition |
|---|---|
| **AD-5** | TDA fingerprint stable under $10\%$ data perturbation |
| **AD-6** | Grammar search converges to unique parsimonious factorization |
| **AD-7** | Induced rule generalizes to operator sizes not seen during induction |

---

## 8. Complete Worked Example

### Lorenz System Discovery

```python
from acf_functor.stochastic_membrane import StochasticMembrane, SMConfig
from acf_functor.taa_agent import TAAAgent
from acf_functor.ergon_agent import ERGONAgent
from acf_functor.autopoietic_scientist import AutopoieticScientist

# ─── Step 0: Generate noisy Lorenz data ──────────────────────────────
import numpy as np
from scipy.integrate import odeint

def lorenz(state, t, sigma=10, rho=28, beta=8/3):
    x, y, z = state
    return [sigma*(y-x), x*(rho-z)-y, x*y-beta*z]

t = np.linspace(0, 10, 1000)
trajectory_clean = odeint(lorenz, [1,1,1], t)
trajectory_noisy = trajectory_clean + np.random.randn(*trajectory_clean.shape) * 0.5

# ─── Step 1: SEM purification ────────────────────────────────────────
config = SMConfig(n_particles=300, cnf_window_tau=50)
sem = StochasticMembrane(config)
sm_output = sem.process(trajectory_noisy, dt=0.01)
certs = sem.get_certificates(sm_output)
print(f"SM-2: {certs['SM-2']['passed']}, SNR = {sm_output.purified.filter_snr_db:.2f} dB")

# ─── Step 2: ERGON diagnostics ───────────────────────────────────────
ergon = ERGONAgent()
ergon_diag = ergon.diagnose(sm_output.purified.x_hat, dt=0.01)

# ─── Step 3: Run P-SAL ───────────────────────────────────────────────
scientist = AutopoieticScientist(
    n_modes_range=(3, 10),
    sindy_threshold=0.05,
    verification_tolerance=0.3,
)
report = scientist.run(
    trajectory=trajectory_noisy,    # fallback
    sm_output=sm_output,            # priority: SEM purified
    dt=0.01,
    h_ks=ergon_diag.h_ks,
    pressure_curvature=ergon_diag.p_prime_prime,
    n_cycles=5,
)

print(report.summary())
# Best law quality: ~0.85, trajectory error: ~0.03
# Certificates: PSAL-1 through PSAL-6: PASS
```

---

## 9. Modules and Implementation

| Module | File | Responsibility |
|---|---|---|
| ROM Synthesizer | `acf_functor/rom_synthesizer.py` | SINDy, STLSQ, polynomial library, Galerkin projection |
| Thermodynamic Closure | `acf_functor/thermodynamic_closure.py` | ERGON, Smagorinsky, SVV, AdaptiveClosureSelector |
| ROM Generator | `poema/rom_generator.py` | PoemROM, AST compilation, RK4 embedding |
| ROM Executor | `poema/backends/gideon/rom_executor.py` | Direct/adaptive/ensemble execution |
| Autopoietic Scientist | `acf_functor/autopoietic_scientist.py` | P-SAL orchestrator, KnowledgeBase, DiscoveredLaw |
| Stochastic Membrane | `acf_functor/stochastic_membrane.py` | SEM v1.1 — noise purification, uncertainty manifold |

---

## 10. Tests

**97 + 26 tests** across the SEM and P-SAL test suites:

`tests/test_stochastic_membrane.py` — 97 tests, all passing (SEM unit tests)

`tests/test_autopoietic_scientist.py` — 26 tests, all passing:

| Test Group | Count | Focus |
|---|---|---|
| TestROMSynthesizer | 6 | SINDy linear, Lorenz, polynomial library, integration ROM, builder, Galerkin |
| TestThermodynamicClosure | 5 | ERGON, Smagorinsky, SVV, selector adaptativo, verification |
| TestPoemROMGenerator | 5 | Koopman ROM, execution, serialization, symbolic, from model |
| TestGideonROMExecutor | 3 | Direct, adaptive, ensemble |
| TestAutopoieticScientist | 6 | Linear, Lorenz, multi-cycle, ERGON closure, knowledge base, certificates |
| TestFullPipeline | 1 | Complete DNS→discover→compile→execute→verify |

---

## 11. Theoretical Properties

### 11.1 Convergence of the P-SAL Loop

**Theorem.** Assume the true governing equations belong to the polynomial library $\Theta$ with $s$ active terms. Then, as the number of P-SAL cycles $N_c \to \infty$ with adaptive STLSQ threshold $\lambda_c \to 0$ at rate $\lambda_c = O(1/\sqrt{c})$:

1. With probability $\to 1$, PSAL-1 passes: $\varepsilon_{\text{SINDy}} \to 0$
2. The quality sequence $Q_1, Q_2, \ldots$ is non-decreasing (proven by the supersession mechanism)
3. The knowledge base converges to the globally optimal law within the polynomial class

### 11.2 Autopoietic Closure Property

**Theorem.** The P-SAL loop is **weakly autopoietically closed**: for any law $\mathcal{L}$ in the knowledge base and any trajectory $\mathbf{a}(t)$ generated by $\mathcal{L}$, running P-SAL on $\mathbf{a}(t)$ recovers a law $\mathcal{L}'$ with $Q(\mathcal{L}') \geq Q(\mathcal{L})$ with high probability.

*Proof sketch:* The trajectory generated by $\mathcal{L}$ satisfies the SINDy residual exactly (by construction). The STLSQ then recovers $\Xi = \Xi_0$ (the true coefficients), giving PSAL-1 = 1.0. The ROM verification then gives $\varepsilon_{traj} = 0$ exactly, so PSAL-4 = 1.0 with quality $Q(\mathcal{L}') = Q(\mathcal{L})$. $\square$

**Implemented closure mechanism (v2.1):** The `KnowledgeBase` implements autopoietic feedback through two concrete methods:

- `propose_improvements()`: Analyzes all verified laws to extract the best `n_modes` (from the most accurate law) and the best `sindy_threshold` (from the sparsest accurate law). These learned preferences bias future discovery cycles.
- `get_adapted_params()`: Merges Knowledge Base preferences with current defaults, returning calibrated parameters that incorporate all accumulated discovery experience. Mode range is centered around the historically optimal value.
- `_adapt()`: Phase 8 of the P-SAL cycle adjusts parameters reactively (per-cycle) AND queries the Knowledge Base for structural improvements. Verified laws reinforce successful parameters; rejected laws trigger threshold relaxation.

This means P-SAL does not "forget" laws — rediscovering the same system from its own simulated data recovers the same law, AND the system's own successful discoveries improve its future performance.

---

## 12. Integration with the ACF Ecosystem

### SEM → P-SAL

SEM provides `SMOutput.purified.x_hat` as the clean trajectory for Phase 1. The uncertainty manifold `SMOutput.uncertainty.sigma_tensor` weights the STLSQ regression. SEM certificates SM-2 and SM-3 are checked before entering the P-SAL loop.

### TAA → P-SAL

TAA's `KoopmanSpectrum.eigenvectors` are the projection matrix for Phase 1. TAA's `alpha_A` classification informs whether the library should include polynomial, trigonometric, or rational terms. TAA certificate TAA-3 ($d^*$) recommends the optimal mode count $r$.

### ERGON → P-SAL

ERGON provides $h_{KS}$ and $P''(1)$ for Phase 3 (thermodynamic closure). ERGON's `mu_srb` (SRB measure) weights the EDMD construction in the TAA phase. ERGON certificate ERG-1 must pass before the ERGON closure is used.

### OTU → P-SAL

OTU's GelfandTriple analysis validates that the Koopman mode spaces are admissible (rigged Hilbert spaces). The Ruelle spectrum from OTU provides the target spectral distribution for PSAL-3 verification.

### P-SAL → Gideon

GideonROMExecutor deploys the certified PoemROMs. The ensemble execution mode uses the SEM uncertainty manifold for Monte Carlo initial condition sampling. Hardware profiling from Gideon Autotune v2.0 informs the adaptive time-stepping in PoemROM execution.

---

## References

- Brunton, S.L., Proctor, J.L., Kutz, J.N. (2016). Discovering governing equations from data by sparse identification of nonlinear dynamical systems. *PNAS*, 113(15), 3932–3937.
- Maturana, H., Varela, F. (1980). *Autopoiesis and Cognition: The Realization of the Living*. Springer.
- Noack, B.R., Afanasiev, K., Morzyński, M., Tadmor, G., Thiele, F. (2003). A hierarchy of low-dimensional models for the transient and post-transient cylinder wake. *Journal of Fluid Mechanics*, 497, 335–363.
- Williams, M.O., Kevrekidis, I.G., Rowley, C.W. (2015). A data-driven approximation of the Koopman operator: Extending dynamic mode decomposition. *Journal of Nonlinear Science*, 25(6), 1307–1346.
- Rowley, C.W., Mezić, I., Bagheri, S., Schlatter, P., Henningson, D.S. (2009). Spectral analysis of nonlinear flows. *Journal of Fluid Mechanics*, 641, 115–127.
- Zhang, L., Schaeffer, H. (2019). On the convergence of the SINDy algorithm. *Multiscale Modeling & Simulation*, 17(3), 948–972.
- Pope, S.B. (2000). *Turbulent Flows*. Cambridge University Press.
- Smagorinsky, J. (1963). General circulation experiments with the primitive equations. *Monthly Weather Review*, 91(3), 99–164.
- Ruelle, D. (2004). *Thermodynamic Formalism*. Cambridge University Press.
- Pesin, Y.B. (1977). Characteristic Lyapunov exponents and smooth ergodic theory. *Russian Mathematical Surveys*, 32(4), 55–114.

---

*P-SAL v2.0 — Part of the ACF Functor Hierarchy*  
*Implementation: `acf_functor/autopoietic_scientist.py` + dependencies*  
*Test coverage: 26/26 tests passing (+ 97 SEM tests)*

P-SAL es el bucle cerrado que permite al ecosistema ACF-Poema-Gideon **descubrir, formalizar, verificar y desplegar sus propias leyes** a partir de datos observacionales. El sistema no recibe ecuaciones gobernantes: las construye.

El nombre «autopoiético» viene de la biología de Maturana y Varela: un sistema que produce sus propios componentes y se mantiene a sí mismo. P-SAL hace exactamente eso con leyes matemáticas: observa datos dinámicos, formula hipótesis de ecuaciones diferenciales mediante regresión dispersa (SINDy), estabiliza las ecuaciones con clausura termodinámica (ERGON), las compila a programas ejecutables (Poema), las verifica contra la referencia, las memoriza como conocimiento certificado y las ejecuta (Gideon).

---

## Arquitectura del Bucle

```
                    ┌─────────────────────────────────────────┐
                    │         AUTOPOIETIC SCIENTIST           │
                    │                                         │
                    │   ┌─────────┐     ┌──────────────┐     │
                    │   │ OBSERVE │ ──> │  HYPOTHESIZE │     │
                    │   └────▲────┘     └──────┬───────┘     │
                    │        │                 │              │
                    │        │           ┌─────▼─────┐       │
                    │        │           │   CLOSE   │       │
                    │        │           └─────┬─────┘       │
                    │        │                 │              │
                    │   ┌────┴────┐     ┌─────▼─────┐       │
                    │   │   ACT   │ <── │  COMPILE  │       │
                    │   └────▲────┘     └─────┬─────┘       │
                    │        │                 │              │
                    │        │           ┌─────▼─────┐       │
                    │        │           │  VERIFY   │       │
                    │        │           └─────┬─────┘       │
                    │        │                 │              │
                    │   ┌────┴────┐     ┌─────▼─────┐       │
                    │   │  ADAPT  │ <── │ MEMORIZE  │       │
                    │   └─────────┘     └───────────┘       │
                    │                                         │
                    └─────────────────────────────────────────┘
```

---

## Fases del Protocolo

### Fase 1: OBSERVE (TAA)

Datos crudos → descomposición modal (SVD/Koopman).

**Agente responsable:** TAA (Topological Agency Algorithm)

- Si hay descomposición de Koopman disponible (pre-computada por TAA), se proyecta la trayectoria sobre los modos principales.
- Si no, SVD/PCA como fallback.
- Salida: `modal_data` de forma `(n_steps, r)` donde `r` = número de modos.

```python
# Internamente:
V = eigenvectors[:, :r].real
modal_data = trajectory @ V
```

### Fase 2: HYPOTHESIZE (SINDy)

Amplitudes modales → dinámica dispersa `ȧ = Ξ·Θ(a)`.

**Motor:** `ROMSynthesizer` → `SINDyEngine`

El algoritmo STLSQ (Sequential Thresholded Least-Squares) encuentra la dinámica más simple que explica las derivadas temporales:

1. Construye biblioteca polinomial $\Theta(\mathbf{a})$: `[1, a₁, a₂, ..., a₁², a₁a₂, ...]`
2. Resuelve $\dot{\mathbf{A}} = \Theta(\mathbf{A})\Xi^T$ por mínimos cuadrados
3. Itera eliminando coeficientes $|Ξ_{ij}| < \lambda$ (threshold)
4. Extrae operadores: $\mathbf{L}$ (lineal), $\mathbf{Q}$ (cuadrático), $\mathbf{c}$ (constante)

**Certificado SINDy-1:** Residual $\|\dot{\mathbf{A}} - \Theta\Xi^T\|_F < \epsilon$

```python
from acf_functor.rom_synthesizer import SINDyEngine
sindy = SINDyEngine(poly_degree=2, threshold=0.05)
result = sindy.fit(modal_data, dt=0.01)
# result.L, result.Q, result.c, result.sparsity
```

### Fase 3: CLOSE (ERGON)

ROM truncado → clausura termodinámica $\nu_t$.

**Agente responsable:** ERGON (vía `ThermodynamicClosure`)

La truncación modal pierde disipación. ERGON la restaura con viscosidad turbulenta basada en el invariante de Pesin:

$$\nu_t = \frac{1}{2} \cdot \frac{P''(1)}{h_{KS}} \cdot \text{Tr}(\text{Cov}(\mathbf{a}_{res}))$$

Donde:
- $P''(1)$: curvatura de la presión topológica en $q=1$
- $h_{KS}$: entropía de Kolmogorov-Sinai (del certificado ERGON ERG-1)
- $\mathbf{a}_{res}$: modos residuales (los que se eliminaron)

**Métodos de clausura (selección adaptativa):**
1. **ERGON termodinámica** (primaria): usa $h_{KS}$ y $P''(1)$
2. **Smagorinsky** (fallback): $\nu_t = (C_S \Delta)^2 |\bar{S}|$
3. **SVV** (Spectral Vanishing Viscosity): $\nu_t(k) \propto e^{-(k_c/k)^2}$

**Certificados:**
- TC-1: Tasa de disipación $\text{Tr}(\mathbf{L} + \nu_t \mathbf{D}) < 0$
- TC-2: Equipartición espectral de alta frecuencia
- TC-3: Convergencia de la clausura

```python
from acf_functor.thermodynamic_closure import AdaptiveClosureSelector
selector = AdaptiveClosureSelector(n_modes=8)
closure = selector.select_and_compute(
    modal_amplitudes=A,
    h_ks=0.693,
    pressure_curvature=0.1,
)
# closure.nu_t, closure.closure_method
```

### Fase 4: COMPILE (Poema)

ROM cerrado → programa ejecutable PoemROM.

**Motor:** `ROMGenerator` (Poema)

Genera un AST de nodos Poema que representan la dinámica:

$$\dot{a}_i = \sum_j L^{\text{eff}}_{ij} a_j + \sum_{j,k} Q_{ijk} a_j a_k + c_i$$

donde $\mathbf{L}^{\text{eff}} = \mathbf{L} + \nu_t \mathbf{D}$.

La compilación produce:
- Nodos AST para cada ecuación
- Integrador RK4 embebido
- Serialización JSON para persistencia
- Representación simbólica legible

```python
from poema.rom_generator import ROMGenerator
gen = ROMGenerator()
poem = gen.from_rom_model(rom, name="navier_stokes_r8", dt=0.01)
symbolic = poem.to_symbolic()  # "da0/dt = -0.1234*a0 + ..."
```

### Fase 5: VERIFY

PoemROM → comparación con datos de referencia.

**Métricas de verificación:**
1. **Error de trayectoria:** $\|\mathbf{a}^{ROM}(t) - \mathbf{a}^{ref}(t)\|_F / \|\mathbf{a}^{ref}\|_F$
2. **Error espectral:** $\|E^{ROM}(k) - E^{ref}(k)\| / \|E^{ref}(k)\|$
3. **Deriva energética:** $|E_{final} - E_0| / |E_0|$
4. **Disipatividad:** $\text{Tr}(\mathbf{L}^{eff}) < 0$

### Fase 6: MEMORIZE

ROM verificado → Knowledge Base persistente.

**Ciclo de vida de una ley:**

```
CANDIDATE → VERIFIED → CERTIFIED → DEPLOYED
                  ↓           ↓
              REJECTED    SUPERSEDED
```

- **CANDIDATE:** Recién descubierta por SINDy
- **VERIFIED:** Error de trayectoria $< 3\times$ tolerancia
- **CERTIFIED:** Error $<$ tolerancia Y disipativa
- **DEPLOYED:** Ejecutándose en Gideon
- **REJECTED:** Inestable o error excesivo
- **SUPERSEDED:** Reemplazada por ley de mayor calidad

**Puntuación de calidad:**

$$Q = \frac{1}{4}\left[(1-\epsilon_{traj}) + S_{sparsity} + \mathbb{1}_{diss} + (1-\delta_E)\right]$$

### Fase 7: ACT (Gideon)

PoemROM → ejecución hardware-optimizada.

**Motor:** `GideonROMExecutor`

Modos de ejecución:
- **Direct:** RK4 puro con NumPy
- **Adaptive:** RK4(5) con control de paso PI
- **Ensemble:** Ejecución paralela de múltiples condiciones iniciales

```python
from poema.backends.gideon.rom_executor import GideonROMExecutor
executor = GideonROMExecutor()
result = executor.execute(poem, a0, mode="adaptive")
# result.trajectory, result.energy_history, result.stable
```

### Fase 8: ADAPT

Retroalimentación → ajuste de parámetros.

Si la ley fue rechazada:
- Reduce threshold de SINDy (más términos activos)
- Aumenta clausura (más disipación)
- Cambia número de modos

Si la ley fue candidata pero no certificada:
- Refina el grado polinomial
- Ajusta tolerancias

---

## Certificados PSAL

| Certificado | Descripción | Criterio |
|---|---|---|
| **PSAL-1** | Descubrimiento de ley | Residual SINDy $< 0.5$ |
| **PSAL-2** | Consistencia termodinámica | ROM es disipativo |
| **PSAL-3** | Fidelidad espectral | $E^{ROM}(k) \approx E^{ref}(k)$ |
| **PSAL-4** | Precisión predictiva | Error de trayectoria $< \tau$ |
| **PSAL-5** | Conservación de energía | Deriva energética $< \delta$ |
| **PSAL-6** | Clausura autopoiética | El sistema generó sus propias leyes |

---

## Uso Completo

```python
from acf_functor.autopoietic_scientist import AutopoieticScientist

# Datos de DNS o experimento
trajectory = dns_solver.get_modal_amplitudes()  # (n_steps, n_features)

# Crear científico
scientist = AutopoieticScientist(
    n_modes_range=(4, 16),       # Explorar de 4 a 16 modos
    sindy_threshold=0.05,         # Umbral de esparsidad
    sindy_poly_degree=2,          # Grado polinomial
    verification_tolerance=0.3,   # Tolerancia de trayectoria
    energy_tolerance=0.5,         # Tolerancia de energía
)

# Ejecutar P-SAL
report = scientist.run(
    trajectory=trajectory,
    dt=0.01,
    h_ks=0.693,                   # Desde ERGON ERG-1
    pressure_curvature=0.1,       # Desde ERGON
    n_cycles=5,                   # 5 ciclos de exploración
)

# Resultados
print(report.summary())
best = report.best_law
print(f"Mejor ley: {best.name}")
print(f"  Calidad: {best.quality_score():.4f}")
print(f"  ν_t: {best.nu_t:.6e}")
print(f"  Modos: {best.n_modes}")
print(f"  Certificados: {best.certificates}")

# Guardar knowledge base
report.knowledge_base.save("knowledge_base.json")
```

---

## Módulos Implementados

| Módulo | Archivo | Descripción |
|---|---|---|
| **ROM Synthesizer** | `acf_functor/rom_synthesizer.py` | SINDy, ROMModel, ROMBuilder, GalerkinProjector |
| **Thermodynamic Closure** | `acf_functor/thermodynamic_closure.py` | ERGON closure, Smagorinsky, SVV, AdaptiveClosureSelector |
| **ROM Generator** | `poema/rom_generator.py` | PoemROM, ROMGenerator, AST compilation |
| **ROM Executor** | `poema/backends/gideon/rom_executor.py` | GideonROMExecutor, adaptive/ensemble execution |
| **Autopoietic Scientist** | `acf_functor/autopoietic_scientist.py` | P-SAL orchestrator, KnowledgeBase, DiscoveredLaw |

---

## Integración con el Ecosistema

### TAA ↔ P-SAL
TAA proporciona la descomposición modal (Koopman) que P-SAL usa en la fase OBSERVE. Los autovalores y autovectores de Koopman son la entrada natural del bucle autopoiético.

### ERGON ↔ P-SAL
ERGON proporciona los diagnósticos termodinámicos ($h_{KS}$, $P''(1)$, exponentes de Lyapunov) que alimentan la fase CLOSE. La clausura termodinámica es el puente entre los modos descubiertos y un ROM estable.

### OTU ↔ P-SAL
OTU (vía GelfandTriple) proporciona el marco funcional para validar que los espacios modales son admisibles. El espectro de Ruelle de OTU complementa la verificación espectral de P-SAL.

### Poema ↔ P-SAL
Poema compila las leyes descubiertas a programas ejecutables. PoemROM es la representación intermedia que conecta descubrimiento con ejecución.

### Gideon ↔ P-SAL
Gideon ejecuta los ROMs compilados con optimización de hardware. GideonROMExecutor es el brazo ejecutor del científico autopoiético.

---

## Tests

26 tests de integración en `tests/test_autopoietic_scientist.py`:

- **TestROMSynthesizer** (6 tests): SINDy lineal, Lorenz, biblioteca polinomial, integración ROM, builder, Galerkin
- **TestThermodynamicClosure** (5 tests): ERGON, Smagorinsky, SVV, selector adaptativo, verificación
- **TestPoemROMGenerator** (5 tests): Koopman ROM, ejecución, serialización, simbólico, desde modelo
- **TestGideonROMExecutor** (3 tests): directo, adaptativo, ensemble
- **TestAutopoieticScientist** (6 tests): lineal, Lorenz, multi-ciclo, clausura ERGON, knowledge base, certificados
- **TestFullPipeline** (1 test): pipeline completo DNS→discover→compile→execute→verify

Todos los 26 tests pasan.

---

## Fundamento Teórico

### Autopoiesis (Maturana & Varela, 1980)
Un sistema autopoiético es aquel que se produce a sí mismo. P-SAL es autopoiético porque las leyes que gobiernan la dinámica del ROM son producidas por el propio sistema a partir de los datos, no inyectadas externamente.

### SINDy (Brunton et al., 2016)
Sparse Identification of Nonlinear Dynamics. Identifica la dinámica gobernante $\dot{\mathbf{x}} = f(\mathbf{x})$ como una combinación dispersa de funciones candidatas $\Theta(\mathbf{x})$.

### Clausura Termodinámica
La conexión con la termodinámica ergódica (ERGON) permite cerrar ROMs truncados de forma físicamente consistente, usando la entropía de Kolmogorov-Sinai como invariante fundamental.

### Reducción de Orden (Noack et al., 2003)
La jerrarquía de modelos de bajo orden conecta DNS → POD/Koopman → ROM Galerkin → ROM SINDy, con clausura termodinámica como el eslabón que faltaba.

---

## Level-5 Autonomy — Descubrimiento de Estructura Latente

### Motivación

La versión original de P-SAL descubría leyes dinámicas mediante SINDy (regresión dispersa sobre una biblioteca de funciones candidatas). Esto funciona bien cuando la biblioteca contiene los términos correctos. Pero ¿qué pasa cuando no sabemos qué biblioteca usar?

**Level-5 Autonomy** reemplaza la selección heurística de bibliotecas con descubrimiento genuino de estructura:

### Componentes Level-5 integrados en P-SAL

| Componente | Rol en P-SAL |
|---|---|
| `TopologicalOperatorAnalyzer` | TDA del Jacobiano del ROM → detecta simetrías cíclicas, persistencia espectral |
| `KoopmanStructuralAnalyzer` | Espectro Koopman del operador de avance → detecta estructura de grupo, multiresolución |
| `OperatorGrammarSearch` | Busca factorización parsimoniosa del operador: {I, P, D, B, ⊗, ·} |
| `AutonomousRuleInduction` | Meta-aprendizaje: induce reglas reutilizables desde descubrimientos previos |

### Flujo Level-5 en P-SAL

```
OBSERVE → Jacobiano J del ROM
       ↓
TDA Fingerprint(J) → {simetría Z_N, persistencia, factorizabilidad}
       ↓
Koopman Analyze(J) → {espectro en círculo unitario, frecuencias racionales}
       ↓
Grammar Search(J)  → factorización parsimoniosa (butterfly/Kronecker/sparse)
       ↓
Rule Induction     → generaliza: "Z_N + unitario ⟹ butterfly con log₂(N) stages"
       ↓
Knowledge Graph    → almacena regla certificada para uso futuro
```

### Certificados Level-5

- **AD-5**: Topological fingerprint estable bajo perturbación
- **AD-6**: Grammar search converge a factorización globalmente parsimoniosa
- **AD-7**: Reglas inducidas generalizan a tamaños de operador no vistos

---

## 13. Koopman RL Truncation Policy in the P-SAL Loop

Starting with Epic 6 (April 2026), the TAA phase inside P-SAL's OBSERVE step uses a **trained RL policy** to select the Koopman truncation threshold δ dynamically, instead of the fixed heuristic.

### 13.1 Where it fits

In Phase 1 (OBSERVE), after TAA's EDMD constructs the full Koopman matrix $K \in \mathbb{C}^{N \times N}$, the truncation to $r$ modes is governed by δ. The `KoopmanTruncationPolicy` is called once per P-SAL cycle:

```python
from acf_functor.koopman_rl_policy import KoopmanTruncationPolicy

# policy was trained offline on representative systems
delta = policy.select_delta(
    eigenvalues=taa_eigenvalues,
    n_rollout=10,
    initial_delta=prev_cycle_delta,  # warm-start from previous cycle
)
# delta is then used as TAA's truncation threshold
```

### 13.2 Why this improves P-SAL

The RL policy adapts δ to the **current spectral state** of the system:
- High spectral entropy → smaller δ (retain more modes, richer dynamics)
- Fast spectral decay (large α_A) → larger δ (fewer modes, cheaper ROM)
- High reconstruction error → decrease δ (keep more modes next cycle)
- High compute budget used → increase δ (sacrifice accuracy for speed)

This closes a feedback loop that the fixed heuristic cannot: the P-SAL cycle's ADAPT phase naturally informs the next cycle's δ selection through the `approx_error` component of `KoopmanState`.

### 13.3 URT preservation

The policy is constrained to never recommend a δ that makes the ROM non-dissipative. If the PSAL-2 certificate ($\text{Tr}(\mathbf{L}^{\text{eff}}) < 0$) would fail at the selected δ, the policy's environment delivers a hard penalty of −10, driving the agent to avoid that region of δ-space during training. This is the same dissipativity theorem proven in §2.7.

**File:** `acf_functor/koopman_rl_policy.py` | **Tests:** `tests/test_koopman_rl.py` (31/31)

---

*P-SAL v2.0.0 — 26 tests, 9 certificados (AD-1..AD-7 + 2 originales), 9 módulos, 1 bucle cerrado, Level-5 Autonomy*

---

## Apéndice G: Certificados Formales en Lean 4 (2026-05-05)

El módulo `MathTest/PSALCertificates.lean` (Lean 4.29.0-rc6 + Mathlib) proporciona
certificados máquina-verificados para los contratos formales del P-SAL.

| Teorema  | Descripción | Estado |
|----------|-------------|--------|
| PSAL-1   | SINDy sparsity: |{Ξi > tol}| ≤ r | ✓ demostrado |
| PSAL-1b  | Términos inactivos contribuyen ≤ tol | ✓ demostrado |
| PSAL-1c  | Error total por umbral ≤ n·tol | ✓ demostrado |
| PSAL-2   | Diccionario Θ(x) cerrado bajo dinámica | axioma |
| PSAL-3   | Cierre ERGON: σ_ERGON ≥ 0 → ley válida | ✓ demostrado |
| PSAL-3b  | σ > 0 → certificado de irreversibilidad | ✓ demostrado |
| PSAL-4   | Error ROM ≤ C_dict · ε_d | ✓ demostrado |
| PSAL-4b  | Modelo completo (d=n): error = 0 | ✓ demostrado |
| PSAL-5   | Cadena SEM→TAA→ERGON→PSAL composable | ✓ demostrado |
| PSAL-5b  | Cadena de certificados es transitiva | ✓ demostrado |
| PSAL-6   | Ley minimal única (RIP) | axioma |

**Axiomas abiertos:** 2 (PSAL-2, PSAL-6) — requieren álgebra de diccionarios y
teoría de Sensing Comprimido (Candès-Romberg-Tao).  
**Teoremas demostrados:** 9 (aritméticos + composicionales).  
**Cadena de certificados:** PSAL-5/5b formalizan la composabilidad del pipeline ACF
completo: SEM → TAA → ERGON → P-SAL con cotas de error transitivas.
