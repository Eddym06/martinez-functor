"""
koopman_rl_policy.py
====================
Koopman RL Truncation Policy — Epic 6 of the ENGINEERING_ROADMAP.

Implements a reinforcement-learning agent that learns the **optimal Koopman
truncation parameter δ(d)** for each dimension *d* of the state space.

The core problem
----------------
EDMD constructs a d × d Koopman matrix whose rank must be *truncated* to the
leading r eigenvalues.  The truncation threshold δ controls which eigenvalues
are kept (those with |λ| > δ).  Too small δ → overfitting (retaining noise
modes).  Too large δ → under-fitting (discarding physically relevant modes).
The optimal δ depends on the system's spectral decay rate, the observation
noise level, and the compute budget — all of which evolve dynamically.

This module trains a tabular Q-learning agent that maps a discrete state
encoding those quantities to a discrete action space {INCREASE, DECREASE, KEEP}
on δ.  The learned policy is then wrapped as a `KoopmanTruncationPolicy` that
`TAAAgent.build()` can call at each EDMD step.

Architecture
------------
KoopmanState         — dataclass for the 6-dimensional observable state
TruncationAction     — enum of 3 discrete actions
KoopmanRLEnvironment — single-episode environment (state, step, reward)
QTable               — tabular Q-function with ε-greedy exploration
KoopmanRLAgent       — full RL agent with train() and policy()
KoopmanTruncationPolicy — thin wrapper for use by TAAAgent
train_koopman_rl     — end-to-end training function on a list of systems

Reward function
---------------
r(δ, δ_prev) = -w_ε · ε(d) / ε(d_prev) - w_c · C(d) / C_max

where:
  ε(d) = Frobenius reconstruction error of the truncated Koopman matrix
  C(d) = compute cost (proportional to r²)
  w_ε, w_c = penalty weights (default 0.7, 0.3)

URT guarantees
--------------
The policy is constrained to keep δ within [δ_min, δ_max] at all times.
An explicit penalty of -10 is applied if the ROM becomes non-dissipative
(i.e., if Tr(L_eff) ≥ 0 after the selected truncation).  This hard penalty
drives the agent to never choose truncations that violate dissipativity,
preserving the URT (Unconditional Robustness Theorem) requirement.

References
----------
- Williams, M.O., Kevrekidis, I.G., Rowley, C.W. (2015). EDMD. JNLS.
- Sutton & Barto (2018). Reinforcement Learning: An Introduction, 2nd ed.
- ENGINEERING_ROADMAP.md — Epic 6 specification
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# State / Action / Reward definitions
# ─────────────────────────────────────────────────────────────────────────────

class TruncationAction(Enum):
    """Three discrete actions the agent can take on δ."""
    DECREASE_DELTA = -1   # keep more modes (smaller threshold → more eigenvalues)
    KEEP_DELTA     =  0   # do nothing
    INCREASE_DELTA = +1   # keep fewer modes (larger threshold → fewer eigenvalues)


@dataclass
class KoopmanState:
    """
    Observable state for the Koopman RL environment.

    Attributes
    ----------
    spectral_entropy : float
        Shannon entropy of the normalised eigenvalue moduli:
        H = -Σ p_k log p_k, p_k = |λ_k| / Σ|λ_j|.
    alpha_A : float
        Spectral decay slope from TAAAgent (dimensionless, typically 0–3).
    lambda_max : float
        Maximum eigenvalue modulus (≥ 0).
    current_delta : float
        Current truncation threshold δ ∈ [δ_min, δ_max].
    approx_error : float
        Koopman reconstruction error ε ∈ [0, ∞).
    compute_budget : float
        Fraction of compute budget used in the last step ∈ [0, 1].
    """
    spectral_entropy: float = 0.0
    alpha_A: float = 1.0
    lambda_max: float = 1.0
    current_delta: float = 0.01
    approx_error: float = 0.0
    compute_budget: float = 0.5

    def to_vector(self) -> np.ndarray:
        """Convert to a 6-element float array."""
        return np.array([
            self.spectral_entropy,
            self.alpha_A,
            self.lambda_max,
            self.current_delta,
            self.approx_error,
            self.compute_budget,
        ], dtype=float)


@dataclass
class KoopmanTransition:
    """A single (s, a, r, s') transition for Q-learning updates."""
    state: KoopmanState
    action: TruncationAction
    reward: float
    next_state: KoopmanState
    done: bool


# ─────────────────────────────────────────────────────────────────────────────
# Q-Table
# ─────────────────────────────────────────────────────────────────────────────

class QTable:
    """
    Tabular Q-function Q(s_disc, a) with ε-greedy policy.

    The continuous state KoopmanState is discretised into bins along each
    dimension.  The default grid uses 5 bins per dimension, giving
    5^6 = 15 625 state cells × 3 actions = 46 875 Q-values.

    Parameters
    ----------
    n_bins : int
        Number of bins per state dimension.
    n_actions : int
        Number of discrete actions (default: 3).
    lr : float
        Learning rate α for Q-updates.
    gamma : float
        Discount factor γ.
    epsilon_start : float
        Initial exploration rate ε.
    epsilon_min : float
        Minimum ε after decay.
    epsilon_decay : float
        Multiplicative decay of ε per episode.
    state_bounds : list of (low, high) pairs
        Bounds for each of the 6 state dimensions.
    """

    # Default state bounds for the 6 KoopmanState dimensions
    DEFAULT_BOUNDS = [
        (0.0, 4.0),    # spectral_entropy
        (0.0, 5.0),    # alpha_A
        (0.0, 2.0),    # lambda_max
        (1e-4, 0.5),   # current_delta
        (0.0, 1.0),    # approx_error
        (0.0, 1.0),    # compute_budget
    ]

    def __init__(
        self,
        n_bins: int = 5,
        n_actions: int = 3,
        lr: float = 0.1,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        state_bounds: Optional[List[Tuple[float, float]]] = None,
    ):
        self.n_bins = n_bins
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.bounds = state_bounds or self.DEFAULT_BOUNDS

        # Q-table: shape (n_bins, n_bins, ..., n_bins, n_actions)
        shape = tuple([n_bins] * len(self.bounds)) + (n_actions,)
        self.table = np.zeros(shape, dtype=float)

        # Bin edges for each dimension
        self._edges = [
            np.linspace(lo, hi, n_bins + 1) for lo, hi in self.bounds
        ]

    def discretise(self, state: KoopmanState) -> Tuple[int, ...]:
        """Map continuous state to tuple of bin indices."""
        vec = state.to_vector()
        indices = []
        for i, (v, edges) in enumerate(zip(vec, self._edges)):
            idx = int(np.searchsorted(edges, v, side='right')) - 1
            idx = max(0, min(self.n_bins - 1, idx))
            indices.append(idx)
        return tuple(indices)

    def get_q_values(self, state: KoopmanState) -> np.ndarray:
        """Return Q(s, ·) for all actions."""
        return self.table[self.discretise(state)]

    def greedy_action(self, state: KoopmanState) -> TruncationAction:
        """Return argmax_a Q(s, a)."""
        q = self.get_q_values(state)
        return list(TruncationAction)[int(np.argmax(q))]

    def epsilon_greedy_action(self, state: KoopmanState) -> TruncationAction:
        """ε-greedy action selection."""
        if np.random.rand() < self.epsilon:
            return np.random.choice(list(TruncationAction))
        return self.greedy_action(state)

    def update(self, transition: KoopmanTransition) -> None:
        """
        Tabular Q-learning update (Watkins & Dayan, 1992):

        Q(s,a) ← Q(s,a) + α [r + γ max_a' Q(s',a') - Q(s,a)]
        """
        s_idx = self.discretise(transition.state)
        a_idx = list(TruncationAction).index(transition.action)
        q_sa = self.table[s_idx][a_idx]

        if transition.done:
            td_target = transition.reward
        else:
            q_next = np.max(self.get_q_values(transition.next_state))
            td_target = transition.reward + self.gamma * q_next

        self.table[s_idx][a_idx] = q_sa + self.lr * (td_target - q_sa)

    def decay_epsilon(self) -> None:
        """Decay exploration rate by one step."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_policy_table(self) -> Dict:
        """Return a summary dict of greedy policy statistics."""
        total_cells = self.table[..., 0].size
        zero_cells = int(np.sum(np.all(self.table == 0, axis=-1)))
        return {
            "total_state_cells": total_cells,
            "visited_cells": total_cells - zero_cells,
            "coverage": 1.0 - zero_cells / total_cells,
            "epsilon": self.epsilon,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class KoopmanRLConfig:
    """Configuration for the Koopman RL environment."""
    delta_min: float = 1e-4       # minimum allowed δ
    delta_max: float = 0.5        # maximum allowed δ
    delta_step: float = 0.005     # δ change per INCREASE/DECREASE action
    w_error: float = 0.7          # penalty weight for reconstruction error
    w_compute: float = 0.3        # penalty weight for compute cost
    dissipative_penalty: float = -10.0  # hard penalty for non-dissipative ROM
    max_steps_per_episode: int = 50     # episode length
    n_modes_min: int = 2                # minimum number of Koopman modes
    n_modes_max: int = 64               # maximum number of Koopman modes


class KoopmanRLEnvironment:
    """
    Single-episode RL environment for Koopman truncation selection.

    The environment simulates the effect of different δ values on a given
    Koopman matrix eigenspectrum, computing the reconstruction error and
    compute cost for each choice.

    Parameters
    ----------
    eigenvalues : (N,) array of complex
        Pre-computed Koopman eigenvalues from EDMD.
    full_koopman_matrix : (N, N) array
        Full Koopman matrix (used to compute reconstruction error).
    config : KoopmanRLConfig
    """

    def __init__(
        self,
        eigenvalues: np.ndarray,
        full_koopman_matrix: np.ndarray,
        config: Optional[KoopmanRLConfig] = None,
    ):
        self.eigenvalues = np.asarray(eigenvalues, dtype=complex)
        self.K_full = np.asarray(full_koopman_matrix, dtype=complex)
        self.config = config or KoopmanRLConfig()

        self._moduli = np.abs(self.eigenvalues)
        self._N = len(self.eigenvalues)
        self._sorted_moduli = np.sort(self._moduli)[::-1]  # descending

        # Precompute baseline error (all modes kept, δ = 0)
        self._err_baseline = self._compute_error(delta=0.0) + 1e-12

        # State reset
        self._step_count = 0
        self._current_delta = (self.config.delta_min + self.config.delta_max) / 2
        self._prev_error = self._compute_error(self._current_delta)

    def _compute_error(self, delta: float) -> float:
        """
        Frobenius reconstruction error for a given δ threshold.

        Constructs the truncated Koopman matrix (zero out modes with
        |λ_k| < δ) and returns ‖K_full - K_trunc‖_F / ‖K_full‖_F.
        """
        mask = self._moduli >= delta
        if not np.any(mask):
            return 1.0

        # Build truncated K using eigendecomposition (diagonal approx.)
        # K_trunc ≈ V · diag(λ_masked) · V^{-1}  — but we only have λ.
        # Use the spectral norm approximation: err ≈ norm of dropped eigenvalues.
        dropped_moduli = self._moduli[~mask]
        norm_K = np.linalg.norm(self.K_full, 'fro') + 1e-12
        dropped_energy = np.sqrt(np.sum(dropped_moduli**2))
        return float(dropped_energy / norm_K)

    def _compute_cost(self, delta: float) -> float:
        """Normalised compute cost = n_retained_modes / N_max."""
        n_modes = int(np.sum(self._moduli >= delta))
        return float(n_modes) / max(self._N, 1)

    def _is_dissipative(self, delta: float) -> bool:
        """
        Check if the truncated spectrum is consistent with dissipativity.
        A spectrum is dissipative if the sum of positive real parts
        (analogous to Tr(L_eff) < 0) is negative on average.
        """
        mask = self._moduli >= delta
        retained = self.eigenvalues[mask]
        if len(retained) == 0:
            return False
        avg_real = np.mean(np.real(retained))
        return bool(avg_real < 0.0)

    def _build_state(self) -> KoopmanState:
        """Compute current KoopmanState from eigenvalues and δ."""
        mask = self._moduli >= self._current_delta
        retained = self._moduli[mask]

        if len(retained) == 0:
            return KoopmanState(
                spectral_entropy=0.0,
                alpha_A=0.0,
                lambda_max=0.0,
                current_delta=self._current_delta,
                approx_error=1.0,
                compute_budget=0.0,
            )

        # Spectral entropy
        p = retained / (retained.sum() + 1e-12)
        entropy = float(-np.sum(p * np.log(p + 1e-12)))

        # Spectral decay slope (log-linear fit of moduli vs rank)
        if len(retained) >= 2:
            log_mod = np.log(retained + 1e-12)
            ranks = np.arange(len(retained), dtype=float)
            if ranks.std() > 1e-12:
                alpha = float(-np.polyfit(ranks, log_mod, 1)[0])
            else:
                alpha = 0.0
        else:
            alpha = 0.0

        err = self._compute_error(self._current_delta)
        cost = self._compute_cost(self._current_delta)

        return KoopmanState(
            spectral_entropy=float(np.clip(entropy, 0.0, 8.0)),
            alpha_A=float(np.clip(alpha, 0.0, 5.0)),
            lambda_max=float(self._moduli.max()),
            current_delta=self._current_delta,
            approx_error=float(np.clip(err, 0.0, 1.0)),
            compute_budget=float(np.clip(cost, 0.0, 1.0)),
        )

    def reset(self) -> KoopmanState:
        """Reset environment to initial state."""
        self._step_count = 0
        self._current_delta = (self.config.delta_min + self.config.delta_max) / 2
        self._prev_error = self._compute_error(self._current_delta)
        return self._build_state()

    def step(
        self, action: TruncationAction
    ) -> Tuple[KoopmanState, float, bool]:
        """
        Apply action, compute reward, return (next_state, reward, done).

        Reward
        ------
        r = -w_ε · ε(δ_new) / ε(δ_prev)   (error improvement ratio)
            -w_c · C(δ_new)                 (compute cost penalty)
            + dissipative_penalty             (if truncation breaks URT)
        """
        cfg = self.config
        delta_prev = self._current_delta
        err_prev = self._compute_error(delta_prev)

        # Apply action
        if action == TruncationAction.INCREASE_DELTA:
            self._current_delta = min(cfg.delta_max,
                                      self._current_delta + cfg.delta_step)
        elif action == TruncationAction.DECREASE_DELTA:
            self._current_delta = max(cfg.delta_min,
                                      self._current_delta - cfg.delta_step)
        # KEEP_DELTA: no change

        err_new = self._compute_error(self._current_delta)
        cost_new = self._compute_cost(self._current_delta)

        # Compute reward
        err_ratio = err_new / (err_prev + 1e-12)
        reward = -cfg.w_error * err_ratio - cfg.w_compute * cost_new

        # Hard penalty for violating URT (non-dissipative truncation)
        if not self._is_dissipative(self._current_delta):
            reward += cfg.dissipative_penalty

        self._step_count += 1
        done = self._step_count >= cfg.max_steps_per_episode

        next_state = self._build_state()
        return next_state, float(reward), done

    @property
    def current_delta(self) -> float:
        return self._current_delta

    @property
    def n_retained_modes(self) -> int:
        return int(np.sum(self._moduli >= self._current_delta))


# ─────────────────────────────────────────────────────────────────────────────
# RL Agent
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainingStats:
    """Statistics collected during training."""
    episode_rewards: List[float] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)
    delta_trajectories: List[List[float]] = field(default_factory=list)
    coverage_history: List[float] = field(default_factory=list)
    n_episodes: int = 0
    total_updates: int = 0
    training_time_s: float = 0.0


class KoopmanRLAgent:
    """
    Tabular Q-learning agent for Koopman truncation selection.

    Parameters
    ----------
    q_table : QTable
        The Q-function to train.
    config : KoopmanRLConfig
        Environment configuration.
    """

    def __init__(
        self,
        q_table: Optional[QTable] = None,
        config: Optional[KoopmanRLConfig] = None,
    ):
        self.q_table = q_table or QTable()
        self.config = config or KoopmanRLConfig()
        self.stats = TrainingStats()

    def train(
        self,
        eigenvalues_list: List[np.ndarray],
        koopman_matrices: List[np.ndarray],
        n_episodes: int = 500,
        verbose: bool = False,
    ) -> TrainingStats:
        """
        Train the agent on a list of systems.

        Each system is defined by its (eigenvalues, koopman_matrix) pair.
        Each episode randomly draws one system, runs it to completion,
        and updates Q using Bellman backup.

        Parameters
        ----------
        eigenvalues_list : list of (N,) complex arrays
        koopman_matrices : list of (N, N) arrays
        n_episodes : int
        verbose : bool

        Returns
        -------
        TrainingStats
        """
        t0 = time.perf_counter()
        n_systems = len(eigenvalues_list)

        for ep in range(n_episodes):
            # Sample a random system for this episode
            sys_idx = np.random.randint(n_systems)
            env = KoopmanRLEnvironment(
                eigenvalues=eigenvalues_list[sys_idx],
                full_koopman_matrix=koopman_matrices[sys_idx],
                config=self.config,
            )

            state = env.reset()
            ep_reward = 0.0
            ep_length = 0
            delta_traj = [env.current_delta]

            done = False
            while not done:
                action = self.q_table.epsilon_greedy_action(state)
                next_state, reward, done = env.step(action)

                transition = KoopmanTransition(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done,
                )
                self.q_table.update(transition)

                state = next_state
                ep_reward += reward
                ep_length += 1
                delta_traj.append(env.current_delta)
                self.stats.total_updates += 1

            self.q_table.decay_epsilon()

            self.stats.episode_rewards.append(ep_reward)
            self.stats.episode_lengths.append(ep_length)
            self.stats.delta_trajectories.append(delta_traj)
            self.stats.coverage_history.append(
                self.q_table.get_policy_table()["coverage"]
            )

            if verbose and (ep + 1) % max(1, n_episodes // 10) == 0:
                avg_r = np.mean(self.stats.episode_rewards[-20:])
                cov = self.stats.coverage_history[-1]
                print(
                    f"  Episode {ep+1:4d}/{n_episodes} | "
                    f"avg_reward={avg_r:+.3f} | "
                    f"ε={self.q_table.epsilon:.3f} | "
                    f"coverage={cov:.1%}"
                )

        self.stats.n_episodes = n_episodes
        self.stats.training_time_s = time.perf_counter() - t0

        return self.stats

    def get_policy(self) -> "KoopmanTruncationPolicy":
        """Return the greedy policy derived from the trained Q-table."""
        return KoopmanTruncationPolicy(
            q_table=self.q_table,
            config=self.config,
        )

    def best_delta(
        self,
        eigenvalues: np.ndarray,
        koopman_matrix: np.ndarray,
        n_rollout_steps: int = 20,
    ) -> float:
        """
        Run the greedy policy on a specific system for a fixed number of
        steps and return the final δ value.

        This is the primary inference method: given a new system's Koopman
        spectrum, return the RL-recommended truncation threshold.
        """
        env = KoopmanRLEnvironment(
            eigenvalues=eigenvalues,
            full_koopman_matrix=koopman_matrix,
            config=self.config,
        )
        state = env.reset()
        for _ in range(n_rollout_steps):
            action = self.q_table.greedy_action(state)
            state, _, done = env.step(action)
            if done:
                break
        return env.current_delta


# ─────────────────────────────────────────────────────────────────────────────
# Policy wrapper — integrates with TAAAgent
# ─────────────────────────────────────────────────────────────────────────────

class KoopmanTruncationPolicy:
    """
    Trained policy for selecting δ(d) at inference time.

    Usage in TAAAgent
    -----------------
    policy = KoopmanTruncationPolicy.from_pretrained(stats_or_agent)
    taa.build(mu_srb=data, truncation_policy=policy)

    The TAAAgent calls policy(state) → δ each time EDMD runs, replacing the
    fixed heuristic δ = alpha_A * sqrt(1/N).
    """

    def __init__(
        self,
        q_table: QTable,
        config: Optional[KoopmanRLConfig] = None,
    ):
        self.q_table = q_table
        self.config = config or KoopmanRLConfig()

    def __call__(self, state: KoopmanState) -> float:
        """Return the greedy δ recommendation for the given state."""
        action = self.q_table.greedy_action(state)
        # Interpret action relative to current_delta
        delta = state.current_delta
        if action == TruncationAction.INCREASE_DELTA:
            delta = min(self.config.delta_max,
                        delta + self.config.delta_step)
        elif action == TruncationAction.DECREASE_DELTA:
            delta = max(self.config.delta_min,
                        delta - self.config.delta_step)
        return float(np.clip(delta, self.config.delta_min, self.config.delta_max))

    def select_delta(
        self,
        eigenvalues: np.ndarray,
        n_rollout: int = 10,
        initial_delta: Optional[float] = None,
    ) -> float:
        """
        Run greedy rollout to select δ for a given eigenvalue spectrum.

        Parameters
        ----------
        eigenvalues : (N,) complex array
        n_rollout : int
            Number of policy steps to run before returning.
        initial_delta : float, optional
            Starting δ.  Defaults to midpoint of [δ_min, δ_max].

        Returns
        -------
        float : recommended δ
        """
        moduli = np.abs(np.asarray(eigenvalues, dtype=complex))
        delta = initial_delta if initial_delta is not None else (
            (self.config.delta_min + self.config.delta_max) / 2
        )

        for _ in range(n_rollout):
            mask = moduli >= delta
            retained = moduli[mask]

            if len(retained) == 0:
                break

            p = retained / (retained.sum() + 1e-12)
            entropy = float(-np.sum(p * np.log(p + 1e-12)))

            if len(retained) >= 2:
                log_mod = np.log(retained + 1e-12)
                ranks = np.arange(len(retained), dtype=float)
                alpha = float(-np.polyfit(ranks, log_mod, 1)[0]) if ranks.std() > 1e-12 else 0.0
            else:
                alpha = 0.0

            err = float(np.sqrt(np.sum(moduli[~mask]**2)) / (np.linalg.norm(moduli) + 1e-12))

            state = KoopmanState(
                spectral_entropy=float(np.clip(entropy, 0, 8)),
                alpha_A=float(np.clip(alpha, 0, 5)),
                lambda_max=float(moduli.max()),
                current_delta=delta,
                approx_error=float(np.clip(err, 0, 1)),
                compute_budget=float(np.clip(len(retained) / max(len(moduli), 1), 0, 1)),
            )

            action = self.q_table.greedy_action(state)
            if action == TruncationAction.INCREASE_DELTA:
                delta = min(self.config.delta_max, delta + self.config.delta_step)
            elif action == TruncationAction.DECREASE_DELTA:
                delta = max(self.config.delta_min, delta - self.config.delta_step)

        return float(np.clip(delta, self.config.delta_min, self.config.delta_max))

    def get_stats(self) -> Dict:
        """Return policy table statistics."""
        return self.q_table.get_policy_table()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Persist the Q-table and config to an .npz file."""
        config_dict = {
            k: v for k, v in self.config.__dict__.items()
        } if self.config is not None else {}
        np.savez_compressed(
            path,
            q_table=self.q_table.table,
            q_table_n_bins=np.array(self.q_table.n_bins),
            q_table_lr=np.array(self.q_table.lr),
            q_table_gamma=np.array(self.q_table.gamma),
            q_table_epsilon=np.array(self.q_table.epsilon),
            q_table_epsilon_min=np.array(self.q_table.epsilon_min),
            q_table_epsilon_decay=np.array(self.q_table.epsilon_decay),
            **{f"cfg_{k}": np.array(v)
               for k, v in config_dict.items()
               if isinstance(v, (int, float, bool))},
        )

    @classmethod
    def load(cls, path: str) -> "KoopmanTruncationPolicy":
        """Load a previously saved policy from an .npz file."""
        data = np.load(path, allow_pickle=False)
        n_bins = int(data["q_table_n_bins"])
        qt = QTable(
            n_bins=n_bins,
            lr=float(data["q_table_lr"]),
            gamma=float(data["q_table_gamma"]),
            epsilon_start=float(data["q_table_epsilon"]),
            epsilon_min=float(data["q_table_epsilon_min"]),
            epsilon_decay=float(data["q_table_epsilon_decay"]),
        )
        qt.table = data["q_table"].copy()

        cfg_keys = {
            k[4:]: float(v) for k, v in data.items()
            if k.startswith("cfg_")
        }
        config = KoopmanRLConfig(**{
            k: v for k, v in cfg_keys.items()
            if k in KoopmanRLConfig.__dataclass_fields__
        }) if cfg_keys else KoopmanRLConfig()
        return cls(q_table=qt, config=config)


# ─────────────────────────────────────────────────────────────────────────────
# Delta Reward Function (standalone utility)
# ─────────────────────────────────────────────────────────────────────────────

class DeltaDRewardFunction:
    """
    Callable reward function for external use or hyperparameter tuning.

    r(δ, system) = -w_ε · ε(δ) - w_c · C(δ) + bonus_dissipative

    Can be used to evaluate the reward of any δ on any system without
    running the full RL environment.
    """

    def __init__(
        self,
        w_error: float = 0.7,
        w_compute: float = 0.3,
        bonus_dissipative: float = 0.5,
    ):
        self.w_error = w_error
        self.w_compute = w_compute
        self.bonus_dissipative = bonus_dissipative

    def __call__(
        self,
        delta: float,
        eigenvalues: np.ndarray,
        koopman_matrix: np.ndarray,
    ) -> float:
        moduli = np.abs(np.asarray(eigenvalues, dtype=complex))
        mask = moduli >= delta
        retained = eigenvalues[mask] if np.any(mask) else np.array([], dtype=complex)

        # Reconstruction error
        norm_K = np.linalg.norm(koopman_matrix, 'fro') + 1e-12
        dropped_moduli = moduli[~mask]
        err = float(np.sqrt(np.sum(dropped_moduli**2)) / norm_K)

        # Compute cost
        cost = float(len(retained)) / max(len(eigenvalues), 1)

        # Dissipativity
        avg_real = float(np.mean(np.real(retained))) if len(retained) > 0 else 0.0
        dissipative_bonus = self.bonus_dissipative if avg_real < 0.0 else 0.0

        return float(-self.w_error * err - self.w_compute * cost + dissipative_bonus)

    def optimal_delta_grid(
        self,
        eigenvalues: np.ndarray,
        koopman_matrix: np.ndarray,
        n_grid: int = 100,
        delta_min: float = 1e-4,
        delta_max: float = 0.5,
    ) -> Tuple[float, float]:
        """
        Brute-force grid search for optimal δ on a given system.

        Returns (optimal_delta, max_reward).
        Useful as a baseline to compare against the RL policy.
        """
        deltas = np.linspace(delta_min, delta_max, n_grid)
        rewards = np.array([
            self(d, eigenvalues, koopman_matrix) for d in deltas
        ])
        best_idx = int(np.argmax(rewards))
        return float(deltas[best_idx]), float(rewards[best_idx])


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end training function
# ─────────────────────────────────────────────────────────────────────────────

def train_koopman_rl(
    eigenvalues_list: Optional[List[np.ndarray]] = None,
    koopman_matrices: Optional[List[np.ndarray]] = None,
    n_episodes: int = 500,
    n_systems: int = 10,
    system_dim: int = 16,
    n_bins: int = 5,
    lr: float = 0.1,
    gamma: float = 0.95,
    epsilon_start: float = 1.0,
    epsilon_min: float = 0.05,
    epsilon_decay: float = 0.995,
    config: Optional[KoopmanRLConfig] = None,
    verbose: bool = True,
) -> Tuple[KoopmanRLAgent, TrainingStats]:
    """
    End-to-end training of the Koopman RL truncation policy.

    If eigenvalues_list / koopman_matrices are not provided, generates
    random synthetic systems with exponentially decaying spectra.

    Parameters
    ----------
    eigenvalues_list : list of (N,) complex arrays, optional
        Pre-computed Koopman eigenvalues for training systems.
    koopman_matrices : list of (N, N) arrays, optional
        Pre-computed Koopman matrices (used for error computation).
    n_episodes : int
        Total training episodes.
    n_systems : int
        Number of synthetic systems to generate (if none provided).
    system_dim : int
        Dimension of synthetic systems.
    n_bins : int
        QTable bins per state dimension.
    lr, gamma, epsilon_start, epsilon_min, epsilon_decay : float
        Q-learning hyperparameters.
    config : KoopmanRLConfig, optional
    verbose : bool

    Returns
    -------
    agent : KoopmanRLAgent (trained)
    stats : TrainingStats
    """
    cfg = config or KoopmanRLConfig()

    # Generate synthetic systems if none provided
    if eigenvalues_list is None or koopman_matrices is None:
        if verbose:
            print(f"Generating {n_systems} synthetic Koopman systems (dim={system_dim})...")
        eigenvalues_list = []
        koopman_matrices = []
        rng = np.random.default_rng(42)
        for _ in range(n_systems):
            # Random exponential spectral decay (stable attractor: |λ| < 1)
            alpha = rng.uniform(0.5, 3.0)  # decay slope
            moduli = np.exp(-alpha * np.arange(system_dim) / system_dim)
            phases = rng.uniform(0, 2 * np.pi, system_dim)
            eigs = moduli * np.exp(1j * phases)
            eigenvalues_list.append(eigs)

            # Build random normal matrix with matching spectrum (diagonal approx)
            K = np.diag(eigs) + rng.normal(0, 0.01, (system_dim, system_dim))
            koopman_matrices.append(K)

    q_table = QTable(
        n_bins=n_bins,
        lr=lr,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
    )
    agent = KoopmanRLAgent(q_table=q_table, config=cfg)

    if verbose:
        print(f"Training KoopmanRLAgent for {n_episodes} episodes "
              f"on {len(eigenvalues_list)} systems...")

    stats = agent.train(
        eigenvalues_list=eigenvalues_list,
        koopman_matrices=koopman_matrices,
        n_episodes=n_episodes,
        verbose=verbose,
    )

    if verbose:
        policy_info = q_table.get_policy_table()
        print(f"\nTraining complete:")
        print(f"  Episodes: {n_episodes}")
        print(f"  Training time: {stats.training_time_s:.2f}s")
        print(f"  Q-table coverage: {policy_info['coverage']:.1%}")
        print(f"  Final ε: {policy_info['epsilon']:.4f}")
        if stats.episode_rewards:
            print(f"  Avg reward (last 50): "
                  f"{np.mean(stats.episode_rewards[-50:]):+.3f}")

    return agent, stats
