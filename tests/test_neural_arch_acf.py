"""
tests/test_neural_arch_acf.py
==============================
Tests for the Neural Architecture ACF domain.

Covers:
- LayerFingerprint computation
- NeuralArchACF.fingerprint() for various model types
- ArchFingerprint construction and alpha_profile
- Similarity computation (metric axioms)
- NASReplacementSearch search and pareto_front
- ArchitectureDatabase add/search/persist
- Bottleneck analysis
"""

import hashlib
import math
import tempfile
import os
import pytest
import numpy as np
import torch
import torch.nn as nn


sys_insert = __import__("sys").path.insert(0, __import__("os").path.dirname(
    __import__("os").path.dirname(os.path.abspath(__file__))
))

from acf_functor.neural_arch_acf import (
    LayerAnalyzer,
    LayerFingerprint,
    LayerKind,
    ArchFingerprint,
    ArchSimilarityResult,
    NeuralArchACF,
    ArchitectureDatabase,
    NASReplacementSearch,
    build_known_architectures_db,
)


# ─────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def small_mlp():
    return nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 10))


@pytest.fixture(scope="module")
def deep_mlp():
    return nn.Sequential(
        nn.Linear(64, 128), nn.GELU(),
        nn.Linear(128, 128), nn.GELU(),
        nn.Linear(128, 64), nn.GELU(),
        nn.Linear(64, 10),
    )


@pytest.fixture(scope="module")
def analyzer():
    return NeuralArchACF(compute_koopman_dynamics=False)


@pytest.fixture(scope="module")
def small_fp(analyzer, small_mlp):
    return analyzer.fingerprint(small_mlp, name="SmallMLP")


@pytest.fixture(scope="module")
def deep_fp(analyzer, deep_mlp):
    return analyzer.fingerprint(deep_mlp, name="DeepMLP")


# ─────────────────────────────────────────────────────────────────────────
# LayerAnalyzer tests
# ─────────────────────────────────────────────────────────────────────────

class TestLayerAnalyzer:
    def test_classify_linear(self):
        a = LayerAnalyzer()
        assert a._classify_layer(nn.Linear(32, 64)) == LayerKind.LINEAR

    def test_classify_conv2d(self):
        a = LayerAnalyzer()
        assert a._classify_layer(nn.Conv2d(3, 16, 3)) == LayerKind.CONV2D

    def test_classify_activation(self):
        a = LayerAnalyzer()
        assert a._classify_layer(nn.ReLU()) == LayerKind.ACTIVATION

    def test_weight_matrix_linear(self):
        a = LayerAnalyzer()
        lin = nn.Linear(32, 64)
        W = a._weight_matrix(lin)
        assert W is not None
        assert W.shape == (64, 32)

    def test_weight_matrix_conv2d(self):
        a = LayerAnalyzer()
        conv = nn.Conv2d(4, 8, kernel_size=3)
        W = a._weight_matrix(conv)
        assert W is not None
        assert W.shape[0] == 8  # out channels

    def test_weight_invariants_shape(self):
        a = LayerAnalyzer()
        W = np.random.randn(16, 32).astype(float)
        spec, eff, st, ent = a._compute_weight_invariants(W)
        assert spec > 0
        assert eff > 0
        assert st > 0
        assert ent >= 0

    def test_alpha_range(self):
        a = LayerAnalyzer()
        lin = nn.Linear(32, 64)
        nn.init.normal_(lin.weight, std=0.1)
        alpha = a._compute_layer_alpha(lin, LayerKind.LINEAR)
        assert 0.0 <= alpha <= 3.0

    def test_nc_class_values(self):
        a = LayerAnalyzer()
        assert a._nc_class(0.1) == "NC0"
        assert a._nc_class(0.3) == "NC1"
        assert a._nc_class(0.7) == "NC2"
        assert a._nc_class(1.5) == "NC3"

    def test_analyze_returns_layer_fingerprint(self):
        a = LayerAnalyzer()
        lin = nn.Linear(32, 64)
        lf = a.analyze("test_layer", lin)
        assert isinstance(lf, LayerFingerprint)
        assert lf.name == "test_layer"
        assert lf.kind == LayerKind.LINEAR
        assert 0 <= lf.alpha <= 3.0
        assert lf.param_count == 32 * 64 + 64  # weights + bias

    def test_to_vector_shape(self):
        a = LayerAnalyzer()
        lin = nn.Linear(16, 32)
        lf = a.analyze("lf", lin)
        v = lf.to_vector()
        assert v.shape == (9,)
        assert np.all(np.isfinite(v))


# ─────────────────────────────────────────────────────────────────────────
# ArchFingerprint tests
# ─────────────────────────────────────────────────────────────────────────

class TestArchFingerprint:
    def test_fingerprint_returns_arch_fingerprint(self, small_fp):
        assert isinstance(small_fp, ArchFingerprint)

    def test_arch_name(self, small_fp):
        assert small_fp.arch_name == "SmallMLP"

    def test_layers_not_empty(self, small_fp):
        assert len(small_fp.layer_fingerprints) > 0

    def test_global_alpha_nonneg(self, small_fp):
        assert small_fp.global_alpha >= 0

    def test_global_nc_class_valid(self, small_fp):
        assert small_fp.global_nc_class in {"NC0", "NC1", "NC2", "NC3"}

    def test_total_params_positive(self, small_fp):
        assert small_fp.total_params > 0

    def test_rademacher_bound_nonneg(self, small_fp):
        assert small_fp.rademacher_bound >= 0

    def test_optimal_depth_positive(self, small_fp):
        assert small_fp.optimal_depth >= 1

    def test_fingerprint_hash_nonempty(self, small_fp):
        assert len(small_fp.fingerprint_hash) > 0

    def test_alpha_profile_shape(self, small_fp):
        ap = small_fp.alpha_profile()
        assert ap.ndim == 1
        assert len(ap) == len(small_fp.layer_fingerprints)

    def test_nc_profile(self, small_fp):
        nc = small_fp.nc_profile()
        assert len(nc) == len(small_fp.layer_fingerprints)
        for c in nc:
            assert c in {"NC0", "NC1", "NC2", "NC3"}

    def test_summary_string(self, small_fp):
        s = small_fp.summary()
        assert "SmallMLP" in s
        assert "α_global" in s

    def test_deep_has_more_layers(self, small_fp, deep_fp):
        assert len(deep_fp.layer_fingerprints) > len(small_fp.layer_fingerprints)


# ─────────────────────────────────────────────────────────────────────────
# Similarity tests
# ─────────────────────────────────────────────────────────────────────────

class TestArchSimilarity:
    def test_self_similarity_high(self, analyzer, small_fp):
        sim = analyzer.similarity(small_fp, small_fp)
        assert sim.combined_score >= 0.8

    def test_similarity_symmetric(self, analyzer, small_fp, deep_fp):
        sim_ab = analyzer.similarity(small_fp, deep_fp)
        sim_ba = analyzer.similarity(deep_fp, small_fp)
        assert abs(sim_ab.combined_score - sim_ba.combined_score) < 0.01

    def test_similarity_score_range(self, analyzer, small_fp, deep_fp):
        sim = analyzer.similarity(small_fp, deep_fp)
        assert 0.0 <= sim.combined_score <= 1.0

    def test_cosine_range(self, analyzer, small_fp, deep_fp):
        sim = analyzer.similarity(small_fp, deep_fp)
        assert -1.0 <= sim.cosine_similarity <= 1.0

    def test_l2_nonneg(self, analyzer, small_fp, deep_fp):
        sim = analyzer.similarity(small_fp, deep_fp)
        assert sim.l2_distance >= 0.0

    def test_summary_method(self, analyzer, small_fp, deep_fp):
        sim = analyzer.similarity(small_fp, deep_fp)
        s = sim.summary()
        assert "SmallMLP" in s or "DeepMLP" in s


# ─────────────────────────────────────────────────────────────────────────
# ArchitectureDatabase tests
# ─────────────────────────────────────────────────────────────────────────

class TestArchitectureDatabase:
    def test_add_and_get(self, small_fp):
        db = ArchitectureDatabase()
        db.add(small_fp)
        assert db.get("SmallMLP") is small_fp

    def test_len(self, small_fp, deep_fp):
        db = ArchitectureDatabase()
        db.add(small_fp)
        db.add(deep_fp)
        assert len(db) == 2

    def test_all_returns_list(self, small_fp, deep_fp):
        db = ArchitectureDatabase()
        db.add(small_fp)
        db.add(deep_fp)
        all_fps = db.all()
        assert isinstance(all_fps, list)
        assert len(all_fps) == 2

    def test_save_and_load(self, small_fp, tmp_path):
        db = ArchitectureDatabase()
        db.add(small_fp)
        path = str(tmp_path / "test_db.json")
        db.save(path)
        assert os.path.exists(path)

        db2 = ArchitectureDatabase()
        db2.load(path)
        fp2 = db2.get("SmallMLP")
        assert fp2 is not None
        assert fp2.arch_name == "SmallMLP"

    def test_known_architectures_db(self):
        db = build_known_architectures_db()
        assert len(db) >= 4
        all_fps = db.all()
        for fp in all_fps:
            assert fp.global_alpha >= 0

    def test_search_returns_result(self, analyzer, small_fp, deep_fp):
        db = ArchitectureDatabase()
        db.add(small_fp)
        db.add(deep_fp)
        result = analyzer.search(small_fp, db, top_k=2)
        assert len(result.candidates) == 2
        assert result.best_match is not None


# ─────────────────────────────────────────────────────────────────────────
# NASReplacementSearch tests
# ─────────────────────────────────────────────────────────────────────────

class TestNASReplacementSearch:
    def test_search_returns_candidates(self):
        nas = NASReplacementSearch(target_metric="balanced")
        spec = {
            "in_dim": 16, "out_dim": 4, "max_layers": 3,
            "hidden_dims": [16, 32], "activations": ["relu"],
        }
        candidates = nas.search(spec, n_candidates=5)
        assert len(candidates) == 5

    def test_candidates_sorted_by_score(self):
        nas = NASReplacementSearch()
        spec = {"in_dim": 8, "out_dim": 2, "max_layers": 2, "hidden_dims": [8, 16]}
        candidates = nas.search(spec, n_candidates=6)
        scores = [s for _, s in candidates]
        assert scores == sorted(scores, reverse=True)

    def test_pareto_front_nonempty(self):
        nas = NASReplacementSearch()
        spec = {"in_dim": 8, "out_dim": 2, "max_layers": 2, "hidden_dims": [8]}
        candidates = nas.search(spec, n_candidates=4)
        pareto = nas.pareto_front(candidates)
        assert len(pareto) >= 1

    def test_all_fingerprints_valid(self):
        nas = NASReplacementSearch()
        spec = {"in_dim": 8, "out_dim": 2, "hidden_dims": [8], "max_layers": 2}
        candidates = nas.search(spec, n_candidates=3)
        for fp, score in candidates:
            assert isinstance(fp, ArchFingerprint)
            assert math.isfinite(score)

    def test_accuracy_target(self):
        nas = NASReplacementSearch(target_metric="accuracy")
        spec = {"in_dim": 16, "out_dim": 4, "hidden_dims": [32, 64]}
        candidates = nas.search(spec, n_candidates=4)
        assert len(candidates) == 4


# ─────────────────────────────────────────────────────────────────────────
# Bottleneck analysis tests
# ─────────────────────────────────────────────────────────────────────────

class TestBottleneckAnalysis:
    def test_bottleneck_analysis_keys(self, analyzer, deep_fp):
        report = analyzer.bottleneck_analysis(deep_fp)
        assert "arch" in report
        assert "bottlenecks" in report
        assert "compression_opportunities" in report
        assert "total_compressible_params" in report

    def test_total_compressible_nonneg(self, analyzer, deep_fp):
        report = analyzer.bottleneck_analysis(deep_fp)
        assert report["total_compressible_params"] >= 0

    def test_recommend_architecture_returns_fp(self, analyzer):
        fp = analyzer.recommend_architecture({
            "task": "classification", "input_dim": 32, "output_dim": 4
        })
        assert isinstance(fp, ArchFingerprint)
        assert "recommended_classification" in fp.arch_name
