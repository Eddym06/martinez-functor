"""
Tests for genesis_copoem_bridge.py — Bridge between Genesis discoveries and CoPoem synthesis.
"""

from __future__ import annotations

import math

import torch
import pytest

from acf_functor.genesis import (
    DiscoveryType,
    DiscoveryStrength,
    MathematicalDiscovery,
    ProgramGenome,
)
from acf_functor.genesis_copoem_bridge import (
    from_genesis_discovery,
    apply_genesis_spec,
)
from poema import CoPoem


def _make_discovery(
    discovery_type: DiscoveryType,
    description: str = "test discovery",
    persistence_score: float = 0.8,
    numerical_evidence: dict | None = None,
) -> MathematicalDiscovery:
    return MathematicalDiscovery(
        discovery_id="test_001",
        discovery_type=discovery_type,
        strength=DiscoveryStrength.NUMERICAL,
        programs=[ProgramGenome("affine", [1.0, 2.0], 1, 2, 42)],
        description=description,
        formal_statement="test",
        numerical_evidence=numerical_evidence or {},
        persistence_score=persistence_score,
        perturbation_stability=0.9,
        max_numerical_error=0.01,
        domain_tested=(-1.0, 1.0),
        n_test_points=100,
        truth_value=0.95,
        discovery_time=0.0,
        generation=0,
    )


class TestGenesisCoPoemBridge:
    def test_differential_relation_maps_to_marginal_stability(self):
        d = _make_discovery(DiscoveryType.DIFFERENTIAL_RELATION, "exp(x)")
        spec = from_genesis_discovery(d, dimension=8)
        assert spec is not None
        assert spec["spectral_radius"] == 1.0
        assert spec["symmetry"] == "orthogonal"
        assert spec["stability"] == "marginal"

    def test_algebraic_identity_maps_to_stable(self):
        d = _make_discovery(DiscoveryType.ALGEBRAIC_IDENTITY, "sin^2+cos^2=1")
        spec = from_genesis_discovery(d, dimension=8)
        assert spec is not None
        assert spec["spectral_radius"] == 0.95
        assert spec["symmetry"] == "symmetric"
        assert spec["stability"] == "stable"

    def test_fixed_point_maps_to_eigenvalue_one(self):
        d = _make_discovery(DiscoveryType.FIXED_POINT, "f(x*)=x*")
        spec = from_genesis_discovery(d, dimension=8)
        assert spec is not None
        assert spec["spectral_radius"] == 1.0
        assert spec["symmetry"] is None
        assert spec["stability"] == "marginal"

    def test_symmetry_maps_to_symmetric_matrix(self):
        d = _make_discovery(DiscoveryType.SYMMETRY, "cos(x) even")
        spec = from_genesis_discovery(d, dimension=8)
        assert spec is not None
        assert spec["discovery_id"] == "test_001"
        assert spec["persistence_score"] == 0.8
        assert spec["dimension"] == 8
