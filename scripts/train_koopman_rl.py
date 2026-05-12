#!/usr/bin/env python3
"""
train_koopman_rl.py — Offline training of the Koopman RL truncation policy
============================================================================

Trains a Q-learning KoopmanTruncationPolicy on a battery of canonical
dynamical systems (logistic map, tent map, dissipative contractions) and
persists the trained agent to disk.

Usage
-----
    cd "/home/Martínez's Invariant"
    source .venv/bin/activate
    python scripts/train_koopman_rl.py [--episodes 500] [--output artifacts/]

After training the script:
  1. Saves the policy to ``artifacts/koopman_rl_policy.npz``
  2. Saves a JSON training log to ``artifacts/koopman_rl_training_log.json``
  3. Validates the policy on held-out test systems

The saved policy can then be loaded and used in TAAAgent:

    from acf_functor.koopman_rl_policy import KoopmanTruncationPolicy
    policy = KoopmanTruncationPolicy.load("artifacts/koopman_rl_policy.npz")
    agent.build(truncation_policy=policy)
    cert = agent.certify()
    print(cert.TAA_12_rl_delta)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Callable, List, Tuple

import numpy as np

# Make sure the workspace root is in the path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from acf_functor.koopman_rl_policy import (
    KoopmanRLConfig,
    KoopmanTruncationPolicy,
    train_koopman_rl,
)
from acf_functor.taa_agent import TAAAgent


# ---------------------------------------------------------------------------
# Canonical test systems
# ---------------------------------------------------------------------------

def logistic_map(r: float = 3.7) -> Callable:
    """Logistic map x_{n+1} = r·x·(1-x), chaotic for r ≈ 3.7."""
    def T(x: np.ndarray) -> np.ndarray:
        return np.array([r * x[0] * (1.0 - x[0])])
    return T


def tent_map(mu: float = 2.0) -> Callable:
    """Tent map — strongly chaotic, h_KS = log(mu) ≈ 0.693."""
    def T(x: np.ndarray) -> np.ndarray:
        v = float(x[0])
        return np.array([mu * v if v < 0.5 else mu * (1.0 - v)])
    return T


def dissipative_map(alpha: float = 0.7) -> Callable:
    """Contractive map x → α·x + 0.1·sin(x), should retain few modes."""
    def T(x: np.ndarray) -> np.ndarray:
        v = float(x[0])
        v_next = alpha * v + 0.1 * np.sin(v)
        return np.array([np.clip(v_next, -0.99, 0.99)])
    return T


TRAIN_SYSTEMS: List[Tuple[str, Callable, Tuple[float, float]]] = [
    ("logistic-3.7",    logistic_map(3.7),   (0.01, 0.99)),
    ("logistic-4.0",    logistic_map(4.0),   (0.01, 0.99)),
    ("tent-2.0",        tent_map(2.0),        (0.01, 0.99)),
    ("dissipative-0.7", dissipative_map(0.7), (-0.99, 0.99)),
    ("dissipative-0.5", dissipative_map(0.5), (-0.99, 0.99)),
]

TEST_SYSTEMS: List[Tuple[str, Callable, Tuple[float, float]]] = [
    ("logistic-3.9",    logistic_map(3.9),   (0.01, 0.99)),
    ("dissipative-0.8", dissipative_map(0.8), (-0.99, 0.99)),
]


# ---------------------------------------------------------------------------
# Collect Koopman spectra from real TAAAgent runs
# ---------------------------------------------------------------------------

def collect_spectra(
    systems: List[Tuple[str, Callable, Tuple[float, float]]],
    n_obs: int = 32,
    n_traj: int = 1000,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Build a TAAAgent for each system, run EDMD, and return spectra."""
    eigenvalues_list = []
    koopman_matrices = []
    for name, T_fn, domain in systems:
        print(f"  EDMD for {name} …", flush=True)
        taa = TAAAgent(T=T_fn, domain=domain, n_obs=n_obs, n_traj=n_traj)
        taa.build()
        if taa._eigenvalues is None or taa._K is None:
            print(f"    WARNING: EDMD failed for {name}, skipping")
            continue
        eigenvalues_list.append(taa._eigenvalues.copy())
        koopman_matrices.append(taa._K.copy())
    return eigenvalues_list, koopman_matrices


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_training(n_episodes: int = 500, output_dir: str = "artifacts") -> None:
    os.makedirs(output_dir, exist_ok=True)

    print(f"[Koopman RL] Collecting EDMD spectra from {len(TRAIN_SYSTEMS)} systems …")
    t0 = time.perf_counter()

    eigenvalues_list, koopman_matrices = collect_spectra(TRAIN_SYSTEMS, n_obs=32)

    if not eigenvalues_list:
        print("[Koopman RL] WARNING: No valid spectra collected; using synthetic systems.")

    print(f"[Koopman RL] Training on {len(eigenvalues_list)} spectra, "
          f"{n_episodes} episodes …", flush=True)

    agent, stats = train_koopman_rl(
        eigenvalues_list=eigenvalues_list if eigenvalues_list else None,
        koopman_matrices=koopman_matrices if koopman_matrices else None,
        n_episodes=n_episodes,
        n_systems=10,     # synthetic fallback count
        system_dim=32,
        verbose=True,
    )

    elapsed = time.perf_counter() - t0
    print(f"[Koopman RL] Training complete in {elapsed:.1f}s")

    # Save policy
    policy = agent.get_policy()
    policy_path = os.path.join(output_dir, "koopman_rl_policy.npz")
    policy.save(policy_path)
    print(f"[Koopman RL] Policy saved → {policy_path}")

    # Save training log
    log_path = os.path.join(output_dir, "koopman_rl_training_log.json")
    with open(log_path, "w") as f:
        json.dump(
            {
                "n_systems": len(TRAIN_SYSTEMS),
                "n_spectra_used": len(eigenvalues_list),
                "n_episodes": n_episodes,
                "elapsed_s": elapsed,
                "training_time_s": float(stats.training_time_s),
                "q_table_coverage": float(agent.q_table.get_policy_table()["coverage"]),
                "q_table_nonzero": int(np.count_nonzero(agent.q_table.table)),
                "q_table_max": float(np.max(agent.q_table.table)),
                "q_table_min": float(np.min(agent.q_table.table)),
            },
            f,
            indent=2,
        )
    print(f"[Koopman RL] Training log saved → {log_path}")

    # Validate on held-out systems
    print("\n[Koopman RL] Validation on held-out systems:")
    loaded_policy = KoopmanTruncationPolicy.load(policy_path)
    test_eigs, test_Ks = collect_spectra(TEST_SYSTEMS, n_obs=32)
    for i, (name, _, _) in enumerate(TEST_SYSTEMS):
        if i >= len(test_eigs):
            print(f"  {name:<22} EDMD failed, skipped")
            continue
        delta_rl = loaded_policy.select_delta(test_eigs[i], n_rollout=10)
        print(f"  {name:<22} δ_RL={delta_rl:.4f}")

    print("\n[Koopman RL] Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Koopman RL truncation policy")
    parser.add_argument("--episodes", type=int, default=500,
                        help="Number of RL episodes (default: 500)")
    parser.add_argument("--output", type=str, default="artifacts",
                        help="Output directory (default: artifacts/)")
    args = parser.parse_args()
    run_training(n_episodes=args.episodes, output_dir=args.output)

