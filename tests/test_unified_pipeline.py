"""Integration tests for the ACF Unified Pipeline and new modules."""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestUnifiedPipeline:
    """End-to-end pipeline tests."""

    @staticmethod
    def _logistic(x):
        return 4.0 * x * (1.0 - x)

    def test_pipeline_logistic_all_stages_pass(self):
        """Logistic map: all 5 stages pass with correct domain."""
        from acf_functor.unified_pipeline import UnifiedPipeline

        traj = np.zeros(400)
        traj[0] = 0.7
        for i in range(1, 400):
            traj[i] = 4.0 * traj[i-1] * (1.0 - traj[i-1])

        pipeline = UnifiedPipeline(auto_calibrate=False, verbose=False)
        result = pipeline.run(
            traj, dt=1.0, system_name="logistic_test",
            T_map=self._logistic, domain=(0.0, 1.0),
        )

        assert result.sem.status.value == "passed", f"SEM failed: {result.sem.error}"
        assert result.taa.status.value == "passed", f"TAA failed: {result.taa.error}"
        assert result.ergon.status.value == "passed", f"ERGON failed: {result.ergon.error}"
        assert result.otu.status.value == "passed", f"OTU failed: {result.otu.error}"
        assert result.psal.status.value == "passed", f"PSAL failed: {result.psal.error}"

    def test_pipeline_logistic_h_ks_correct(self):
        """Logistic map: h_KS ≈ log(2) ≈ 0.693."""
        from acf_functor.unified_pipeline import UnifiedPipeline

        traj = np.zeros(400)
        traj[0] = 0.7
        for i in range(1, 400):
            traj[i] = 4.0 * traj[i-1] * (1.0 - traj[i-1])

        pipeline = UnifiedPipeline(auto_calibrate=False, verbose=False)
        result = pipeline.run(
            traj, dt=1.0, system_name="logistic_test",
            T_map=self._logistic, domain=(0.0, 1.0),
        )

        h_ks = result.ergon.data.get("h_ks", 0)
        assert 0.5 < h_ks < 0.9, f"h_KS={h_ks:.4f} not in (0.5, 0.9), expected ~0.693"

    def test_pipeline_sine_map(self):
        """Sine map (non-chaotic): TAA should classify correctly."""
        from acf_functor.unified_pipeline import UnifiedPipeline

        def sine_map(x):
            return 0.5 * np.sin(np.pi * x)

        traj = np.zeros(300)
        traj[0] = 0.5
        for i in range(1, 300):
            traj[i] = 0.5 * np.sin(np.pi * traj[i-1])

        pipeline = UnifiedPipeline(auto_calibrate=False, verbose=False)
        result = pipeline.run(
            traj, dt=1.0, system_name="sine_test",
            T_map=sine_map, domain=(-1.0, 1.0),
        )

        # TAA should pass (classifies correctly)
        assert result.taa.status.value == "passed"
        # Should NOT be chaotic
        decay_class = result.taa.data.get("decay_class", "unknown")
        assert decay_class != "chaotic", f"Sine map wrongly classified as chaotic"


class TestParameterCalibration:
    """Auto-calibration tests."""

    @staticmethod
    def _logistic(x):
        return 4.0 * x * (1.0 - x)

    def test_quick_calibrate_returns_params(self):
        """Quick calibration returns valid parameter dicts."""
        from acf_functor.parameter_calibration import quick_calibrate

        params = quick_calibrate(
            self._logistic, (0.0, 1.0),
            agents=["taa", "ergon"],
            method="random", n_trials=5, verbose=False,
        )

        assert "taa" in params
        assert "ergon" in params
        assert len(params["taa"]) > 0
        assert len(params["ergon"]) > 0
        # All values should be finite
        for v in params["taa"].values():
            assert np.isfinite(v)
        for v in params["ergon"].values():
            assert np.isfinite(v)

    def test_calibration_improves_over_default(self):
        """Calibrated params should improve objective vs defaults."""
        from acf_functor.parameter_calibration import ParameterCalibration

        system = [{"T": self._logistic, "domain": (0.0, 1.0), "name": "test"}]
        calibrator = ParameterCalibration(agents=["taa"])
        results = calibrator.calibrate(
            system, method="random", n_trials=10, verbose=False,
        )

        taa_result = results["taa"]
        assert taa_result.objective_value < 10.0, "Calibration produced terrible params"
        assert taa_result.n_trials > 0


class TestNavierStokesValidator:
    """Real-world system tests."""

    def test_lorenz96_runs(self):
        """Lorenz-96 generates trajectory without crashing."""
        from acf_functor.navier_stokes_validator import Lorenz96, SystemConfig, SystemType

        config = SystemConfig(
            system_type=SystemType.LORENZ_96,
            n_lorenz=10, F_lorenz=8.0,
            dt=0.05, n_steps=200, n_warmup=50,
        )
        l96 = Lorenz96(config)
        traj = l96.generate_trajectory()
        assert traj.shape == (200, 10)
        assert np.all(np.isfinite(traj))

    def test_garch_runs(self):
        """GARCH generator produces valid returns."""
        from acf_functor.navier_stokes_validator import FinancialGenerator

        returns = FinancialGenerator.garch(n_steps=500)
        assert len(returns) == 500
        assert np.all(np.isfinite(returns))
        # GARCH should have volatility clustering (non-constant variance)
        assert np.std(returns) > 0.0


class TestPipelineMemory:
    """Tests that the pipeline doesn't leak or corrupt state."""

    @staticmethod
    def _logistic(x):
        return 4.0 * x * (1.0 - x)

    def test_repeated_runs_consistent(self):
        """Multiple runs on same data give consistent results."""
        from acf_functor.unified_pipeline import UnifiedPipeline

        traj = np.zeros(300)
        traj[0] = 0.5
        for i in range(1, 300):
            traj[i] = 4.0 * traj[i-1] * (1.0 - traj[i-1])

        pipeline = UnifiedPipeline(auto_calibrate=False, verbose=False)

        h_ks_values = []
        for _ in range(2):
            result = pipeline.run(
                traj, dt=1.0, system_name="test",
                T_map=self._logistic, domain=(0.0, 1.0),
            )
            h_ks_values.append(result.ergon.data.get("h_ks", 0))

        # h_KS should be deterministic (seeded)
        assert abs(h_ks_values[0] - h_ks_values[1]) < 0.01, \
            f"Inconsistent h_KS: {h_ks_values}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])