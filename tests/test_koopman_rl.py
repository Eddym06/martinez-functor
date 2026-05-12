"""
tests/test_koopman_rl.py
========================
Unit and integration tests for the Koopman RL Truncation Policy (Epic 6).

Test groups
-----------
TestKoopmanState         — dataclass construction and serialisation
TestQTable               — discretisation, Q-update, epsilon-greedy
TestKoopmanRLEnvironment — reset, step, reward structure
TestKoopmanRLAgent       — training loop on synthetic systems
TestKoopmanTruncationPolicy — inference API and select_delta
TestDeltaDRewardFunction — reward function and grid search
TestTrainKoopmanRL       — end-to-end integration
"""

import numpy as np
import pytest

from acf_functor.koopman_rl_policy import (
    DeltaDRewardFunction,
    KoopmanRLAgent,
    KoopmanRLConfig,
    KoopmanRLEnvironment,
    KoopmanState,
    KoopmanTruncationPolicy,
    QTable,
    TrainingStats,
    TruncationAction,
    train_koopman_rl,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def stable_spectrum():
    """6-mode stable system: exponential spectral decay, eigenvalues inside unit circle."""
    rng = np.random.default_rng(0)
    moduli = np.exp(-np.linspace(0.2, 2.0, 6))
    phases = rng.uniform(0, 2 * np.pi, 6)
    eigs = moduli * np.exp(1j * phases)
    K = np.diag(eigs)
    return eigs, K


@pytest.fixture
def large_spectrum():
    """32-mode system for performance tests."""
    rng = np.random.default_rng(7)
    moduli = np.exp(-np.arange(32) / 8.0)
    phases = rng.uniform(0, 2 * np.pi, 32)
    eigs = moduli * np.exp(1j * phases)
    K = np.diag(eigs) + rng.normal(0, 0.01, (32, 32))
    return eigs, K


@pytest.fixture
def default_env(stable_spectrum):
    eigs, K = stable_spectrum
    return KoopmanRLEnvironment(eigenvalues=eigs, full_koopman_matrix=K)


@pytest.fixture
def trained_agent(stable_spectrum, large_spectrum):
    """Lightly trained agent (100 episodes) for inference tests."""
    eigs_s, K_s = stable_spectrum
    eigs_l, K_l = large_spectrum
    agent = KoopmanRLAgent()
    agent.train(
        eigenvalues_list=[eigs_s, eigs_l],
        koopman_matrices=[K_s, K_l],
        n_episodes=100,
        verbose=False,
    )
    return agent


# ─────────────────────────────────────────────────────────────────────────────
# TestKoopmanState
# ─────────────────────────────────────────────────────────────────────────────

class TestKoopmanState:
    def test_default_construction(self):
        s = KoopmanState()
        assert s.spectral_entropy == 0.0
        assert s.alpha_A == 1.0
        assert s.current_delta == 0.01
        assert s.approx_error == 0.0

    def test_to_vector_shape(self):
        s = KoopmanState(
            spectral_entropy=1.5,
            alpha_A=2.0,
            lambda_max=0.9,
            current_delta=0.05,
            approx_error=0.1,
            compute_budget=0.3,
        )
        v = s.to_vector()
        assert v.shape == (6,)
        assert v[0] == pytest.approx(1.5)
        assert v[3] == pytest.approx(0.05)

    def test_to_vector_dtype(self):
        s = KoopmanState()
        assert s.to_vector().dtype == float


# ─────────────────────────────────────────────────────────────────────────────
# TestQTable
# ─────────────────────────────────────────────────────────────────────────────

class TestQTable:
    def test_table_shape(self):
        qt = QTable(n_bins=4)
        assert qt.table.shape == (4, 4, 4, 4, 4, 4, 3)

    def test_discretise_in_bounds(self):
        qt = QTable(n_bins=5)
        s = KoopmanState(spectral_entropy=2.0, alpha_A=2.5,
                         lambda_max=1.0, current_delta=0.25,
                         approx_error=0.5, compute_budget=0.5)
        idx = qt.discretise(s)
        assert len(idx) == 6
        assert all(0 <= i < 5 for i in idx)

    def test_discretise_clipped_extremes(self):
        qt = QTable(n_bins=5)
        s_low = KoopmanState(spectral_entropy=-100.0, alpha_A=0.0,
                              lambda_max=0.0, current_delta=1e-6,
                              approx_error=-5.0, compute_budget=-1.0)
        s_high = KoopmanState(spectral_entropy=1000.0, alpha_A=100.0,
                               lambda_max=10.0, current_delta=10.0,
                               approx_error=100.0, compute_budget=2.0)
        for idx in [qt.discretise(s_low), qt.discretise(s_high)]:
            assert all(0 <= i < 5 for i in idx)

    def test_q_update_changes_value(self):
        qt = QTable(n_bins=5)
        s = KoopmanState()
        before = qt.get_q_values(s).copy()

        from acf_functor.koopman_rl_policy import KoopmanTransition
        t = KoopmanTransition(
            state=s,
            action=TruncationAction.KEEP_DELTA,
            reward=1.0,
            next_state=s,
            done=True,
        )
        qt.update(t)
        after = qt.get_q_values(s)
        assert not np.allclose(before, after)

    def test_greedy_action_returns_truncation_action(self):
        qt = QTable(n_bins=5)
        s = KoopmanState()
        action = qt.greedy_action(s)
        assert isinstance(action, TruncationAction)

    def test_epsilon_greedy_returns_valid_action(self):
        qt = QTable(n_bins=5, epsilon_start=1.0)  # pure exploration
        s = KoopmanState()
        for _ in range(20):
            action = qt.epsilon_greedy_action(s)
            assert isinstance(action, TruncationAction)

    def test_epsilon_decay(self):
        qt = QTable(epsilon_start=1.0, epsilon_min=0.05, epsilon_decay=0.9)
        qt.decay_epsilon()
        assert qt.epsilon == pytest.approx(0.9)
        for _ in range(100):
            qt.decay_epsilon()
        assert qt.epsilon >= qt.epsilon_min


# ─────────────────────────────────────────────────────────────────────────────
# TestKoopmanRLEnvironment
# ─────────────────────────────────────────────────────────────────────────────

class TestKoopmanRLEnvironment:
    def test_reset_returns_koopman_state(self, default_env):
        state = default_env.reset()
        assert isinstance(state, KoopmanState)
        assert 0.0 <= state.approx_error <= 1.0
        assert state.compute_budget >= 0.0

    def test_step_returns_triple(self, default_env):
        default_env.reset()
        result = default_env.step(TruncationAction.KEEP_DELTA)
        assert len(result) == 3
        next_state, reward, done = result
        assert isinstance(next_state, KoopmanState)
        assert isinstance(reward, float)
        assert isinstance(done, bool)

    def test_increase_delta_raises_delta(self, default_env):
        default_env.reset()
        delta_before = default_env.current_delta
        default_env.step(TruncationAction.INCREASE_DELTA)
        assert default_env.current_delta >= delta_before

    def test_decrease_delta_lowers_delta(self, default_env):
        # Start near max delta
        default_env._current_delta = default_env.config.delta_max - 0.05
        default_env.step(TruncationAction.DECREASE_DELTA)
        assert default_env.current_delta <= default_env.config.delta_max - 0.05

    def test_delta_stays_in_bounds(self, default_env):
        cfg = default_env.config
        default_env.reset()
        for _ in range(100):
            action = np.random.choice(list(TruncationAction))
            default_env.step(action)
            assert cfg.delta_min <= default_env.current_delta <= cfg.delta_max

    def test_episode_terminates(self, default_env):
        cfg = default_env.config
        state = default_env.reset()
        done = False
        n_steps = 0
        while not done:
            _, _, done = default_env.step(TruncationAction.KEEP_DELTA)
            n_steps += 1
            assert n_steps <= cfg.max_steps_per_episode + 1

    def test_reward_penalises_non_dissipative(self, stable_spectrum):
        """Non-stable spectrum should trigger hard penalty."""
        eigs_s, K_s = stable_spectrum
        # Create a non-dissipative spectrum (positive real parts)
        eigs_unstable = np.array([0.5 + 0.1j, 0.3 - 0.2j, 0.1 + 0.05j])
        K_unstable = np.diag(eigs_unstable)
        env = KoopmanRLEnvironment(
            eigenvalues=eigs_unstable,
            full_koopman_matrix=K_unstable,
            config=KoopmanRLConfig(delta_min=0.0, delta_max=0.0, delta_step=0.0),
        )
        env.reset()
        _, reward, _ = env.step(TruncationAction.KEEP_DELTA)
        # With positive real parts and dissipative_penalty=-10, reward is penalised
        assert reward < 0


# ─────────────────────────────────────────────────────────────────────────────
# TestKoopmanRLAgent
# ─────────────────────────────────────────────────────────────────────────────

class TestKoopmanRLAgent:
    def test_training_returns_stats(self, stable_spectrum):
        eigs, K = stable_spectrum
        agent = KoopmanRLAgent()
        stats = agent.train(
            eigenvalues_list=[eigs],
            koopman_matrices=[K],
            n_episodes=20,
            verbose=False,
        )
        assert isinstance(stats, TrainingStats)
        assert len(stats.episode_rewards) == 20
        assert stats.total_updates > 0

    def test_episode_rewards_finite(self, stable_spectrum):
        eigs, K = stable_spectrum
        agent = KoopmanRLAgent()
        stats = agent.train(
            eigenvalues_list=[eigs],
            koopman_matrices=[K],
            n_episodes=30,
            verbose=False,
        )
        assert all(np.isfinite(r) for r in stats.episode_rewards)

    def test_coverage_increases(self, stable_spectrum, large_spectrum):
        eigs_s, K_s = stable_spectrum
        eigs_l, K_l = large_spectrum
        agent = KoopmanRLAgent()
        stats = agent.train(
            eigenvalues_list=[eigs_s, eigs_l],
            koopman_matrices=[K_s, K_l],
            n_episodes=100,
            verbose=False,
        )
        # Coverage should grow
        assert stats.coverage_history[-1] >= stats.coverage_history[0]

    def test_best_delta_in_bounds(self, trained_agent, stable_spectrum):
        eigs, K = stable_spectrum
        delta = trained_agent.best_delta(eigs, K, n_rollout_steps=10)
        cfg = trained_agent.config
        assert cfg.delta_min <= delta <= cfg.delta_max

    def test_get_policy_returns_policy(self, trained_agent):
        policy = trained_agent.get_policy()
        assert isinstance(policy, KoopmanTruncationPolicy)


# ─────────────────────────────────────────────────────────────────────────────
# TestKoopmanTruncationPolicy
# ─────────────────────────────────────────────────────────────────────────────

class TestKoopmanTruncationPolicy:
    def test_call_returns_float(self, trained_agent, stable_spectrum):
        eigs, _ = stable_spectrum
        policy = trained_agent.get_policy()
        moduli = np.abs(eigs)
        state = KoopmanState(
            spectral_entropy=1.0, alpha_A=1.0, lambda_max=float(moduli.max()),
            current_delta=0.05, approx_error=0.1, compute_budget=0.5,
        )
        delta = policy(state)
        assert isinstance(delta, float)

    def test_call_delta_in_bounds(self, trained_agent):
        policy = trained_agent.get_policy()
        cfg = policy.config
        for _ in range(20):
            state = KoopmanState(
                spectral_entropy=np.random.uniform(0, 4),
                alpha_A=np.random.uniform(0, 5),
                lambda_max=np.random.uniform(0, 2),
                current_delta=np.random.uniform(cfg.delta_min, cfg.delta_max),
                approx_error=np.random.uniform(0, 1),
                compute_budget=np.random.uniform(0, 1),
            )
            delta = policy(state)
            assert cfg.delta_min <= delta <= cfg.delta_max

    def test_select_delta_in_bounds(self, trained_agent, stable_spectrum):
        eigs, _ = stable_spectrum
        policy = trained_agent.get_policy()
        delta = policy.select_delta(eigs, n_rollout=10)
        cfg = policy.config
        assert cfg.delta_min <= delta <= cfg.delta_max

    def test_get_stats_keys(self, trained_agent):
        policy = trained_agent.get_policy()
        stats = policy.get_stats()
        assert "total_state_cells" in stats
        assert "coverage" in stats
        assert "epsilon" in stats

    def test_save_and_load_roundtrip(self, trained_agent, tmp_path):
        """Saved policy should produce identical δ to the original."""
        policy = trained_agent.get_policy()
        path = str(tmp_path / "policy.npz")
        policy.save(path)
        loaded = KoopmanTruncationPolicy.load(path)
        assert loaded is not None
        # Both policies should produce the same δ on the same input
        rng = np.random.default_rng(7)
        eigs = np.exp(-rng.uniform(0.1, 2.0, 16)) * np.exp(1j * rng.uniform(0, 2 * np.pi, 16))
        delta_orig = policy.select_delta(eigs, n_rollout=5)
        delta_loaded = loaded.select_delta(eigs, n_rollout=5)
        assert abs(delta_orig - delta_loaded) < 1e-9, (
            f"δ mismatch after save/load: orig={delta_orig}, loaded={delta_loaded}"
        )

    def test_load_delta_in_bounds(self, trained_agent, tmp_path):
        policy = trained_agent.get_policy()
        path = str(tmp_path / "policy2.npz")
        policy.save(path)
        loaded = KoopmanTruncationPolicy.load(path)
        cfg = loaded.config
        eigs = np.exp(-np.linspace(0.1, 2.0, 8)) * np.exp(1j * np.linspace(0, np.pi, 8))
        delta = loaded.select_delta(eigs, n_rollout=10)
        assert cfg.delta_min <= delta <= cfg.delta_max


# ─────────────────────────────────────────────────────────────────────────────
# TestDeltaDRewardFunction
# ─────────────────────────────────────────────────────────────────────────────

class TestDeltaDRewardFunction:
    def test_reward_negative_for_high_error(self, stable_spectrum):
        eigs, K = stable_spectrum
        rf = DeltaDRewardFunction()
        # Large δ → few modes retained → high error → negative reward
        r = rf(delta=0.99, eigenvalues=eigs, koopman_matrix=K)
        assert r < 0

    def test_reward_finite(self, stable_spectrum):
        eigs, K = stable_spectrum
        rf = DeltaDRewardFunction()
        for delta in [0.001, 0.01, 0.1, 0.5]:
            r = rf(delta=delta, eigenvalues=eigs, koopman_matrix=K)
            assert np.isfinite(r)

    def test_optimal_delta_grid_returns_valid(self, stable_spectrum):
        eigs, K = stable_spectrum
        rf = DeltaDRewardFunction()
        opt_delta, opt_reward = rf.optimal_delta_grid(eigs, K, n_grid=20)
        assert np.isfinite(opt_delta)
        assert np.isfinite(opt_reward)
        assert 1e-4 <= opt_delta <= 0.5


# ─────────────────────────────────────────────────────────────────────────────
# TestTrainKoopmanRL
# ─────────────────────────────────────────────────────────────────────────────

class TestTrainKoopmanRL:
    def test_train_returns_agent_and_stats(self):
        agent, stats = train_koopman_rl(
            n_episodes=20, n_systems=3, system_dim=8, verbose=False,
        )
        assert isinstance(agent, KoopmanRLAgent)
        assert isinstance(stats, TrainingStats)
        assert stats.n_episodes == 20

    def test_policy_from_trained_agent_works(self):
        agent, _ = train_koopman_rl(
            n_episodes=50, n_systems=5, system_dim=8, verbose=False,
        )
        policy = agent.get_policy()
        # Quick inference on a random spectrum
        rng = np.random.default_rng(99)
        eigs = np.exp(-np.linspace(0.1, 2.0, 8)) * np.exp(1j * rng.uniform(0, 2 * np.pi, 8))
        delta = policy.select_delta(eigs, n_rollout=5)
        assert np.isfinite(delta)
        cfg = policy.config
        assert cfg.delta_min <= delta <= cfg.delta_max
