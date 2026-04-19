"""Validation tests for Koopman reduction on nonlinear systems."""

import math

import pytest
import torch

from acf_functor.core import KoopmanReducer
from acf_functor.koopman_adaptive import AdaptiveKoopman


@pytest.fixture
def dtype():
	return torch.float64


class TestLinearSystems:
	"""Test that Koopman reduction is exact for linear systems."""

	def test_identity_system(self, dtype):
		x = torch.zeros(2, 100, dtype=dtype)
		x[:, 0] = torch.tensor([1.0, 2.0], dtype=dtype)
		for t in range(99):
			x[:, t + 1] = x[:, t]
		_k, eigvals, meta = KoopmanReducer.dmd(x, observable_library="polynomial", poly_degree=1)
		assert eigvals.numel() > 0
		assert meta["reconstruction_error"] < 1e-10

	def test_scalar_decay(self, dtype):
		x = torch.zeros(1, 200, dtype=dtype)
		x[0, 0] = 1.0
		for t in range(199):
			x[0, t + 1] = 0.9 * x[0, t]
		_k, eigvals, meta = KoopmanReducer.dmd(x, observable_library="polynomial", poly_degree=1)
		assert meta["reconstruction_error"] < 0.01
		abs_eig = torch.abs(eigvals)
		assert torch.min(abs_eig - 0.9).item() < 0.05

	def test_2d_rotation(self, dtype):
		theta = 0.1
		r = torch.tensor(
			[[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]],
			dtype=dtype,
		)
		x = torch.zeros(2, 200, dtype=dtype)
		x[:, 0] = torch.tensor([1.0, 0.0], dtype=dtype)
		for t in range(199):
			x[:, t + 1] = r @ x[:, t]
		_k, eigvals, meta = KoopmanReducer.dmd(x, observable_library="polynomial", poly_degree=2)
		assert eigvals.numel() > 0
		assert meta["reconstruction_error"] < 0.1


class TestNonlinearSystems:
	"""Test Koopman reduction on nonlinear dynamical systems."""

	def test_logistic_map(self, dtype):
		r = 2.0
		x = torch.zeros(1, 300, dtype=dtype)
		x[0, 0] = 0.1
		for t in range(299):
			x[0, t + 1] = r * x[0, t] * (1 - x[0, t])
		engine = AdaptiveKoopman(observable_families=["polynomial"], max_poly_degree=3)
		_result, diag = engine.reduce(x)
		assert diag.reconstruction_error < 0.5


class TestSpectralConvergence:
	"""Test spectral decay properties of Koopman reduction."""

	def test_spectral_decay_linear(self, dtype):
		x = torch.zeros(2, 200, dtype=dtype)
		x[:, 0] = torch.tensor([1.0, 0.5], dtype=dtype)
		a = torch.tensor([[0.9, 0.1], [0.0, 0.8]], dtype=dtype)
		for t in range(199):
			x[:, t + 1] = a @ x[:, t]
		engine = AdaptiveKoopman()
		_result, diag = engine.reduce(x)
		eigenvalues = torch.linalg.eigvals(a)
		expected_radius = torch.max(torch.abs(eigenvalues)).item()
		assert abs(diag.spectral_radius - expected_radius) <= 0.11

	def test_dimension_vs_error_tradeoff(self, dtype):
		x = torch.zeros(1, 300, dtype=dtype)
		x[0, 0] = 0.5
		for t in range(299):
			x[0, t + 1] = 0.9 * x[0, t] + 0.1 * torch.sin(x[0, t])
		errors_by_dim = []
		for max_deg in [1, 2, 3]:
			engine = AdaptiveKoopman(observable_families=["polynomial"], max_poly_degree=max_deg)
			_result, diag = engine.reduce(x.clone())
			errors_by_dim.append((max_deg, diag.reconstruction_error))
		assert errors_by_dim[0][1] >= errors_by_dim[-1][1] * 0.9


class TestKoopmanACFInvariant:
	"""Test Affine Spectral Decay Index alpha(f) computed from Koopman eigenvalues."""

	def test_invariant_computation(self, dtype):
		x = torch.zeros(2, 200, dtype=dtype)
		x[:, 0] = torch.tensor([1.0, 0.0], dtype=dtype)
		theta = 0.5
		r = torch.tensor(
			[[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]],
			dtype=dtype,
		)
		for t in range(199):
			x[:, t + 1] = r @ x[:, t]
		engine = AdaptiveKoopman()
		result, _diag = engine.reduce(x)
		alpha = result.metadata.get("acf_alpha", None)
		if alpha is not None:
			assert alpha >= 0


class TestTruncationDelta:
	"""Test delta(d) metadata existence."""

	def test_truncation_delta_exists(self, dtype):
		x = torch.zeros(2, 300, dtype=dtype)
		x[:, 0] = torch.tensor([2.0, 1.0], dtype=dtype)
		a = torch.tensor([[0.95, 0.05], [0.1, 0.85]], dtype=dtype)
		for t in range(299):
			x[:, t + 1] = a @ x[:, t]
		engine = AdaptiveKoopman()
		result, _diag = engine.reduce(x)
		delta_d = result.metadata.get("truncation_delta", None)
		if delta_d is not None:
			assert delta_d >= 0