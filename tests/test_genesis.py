"""Tests for Evolution 20: Mathematical Genesis."""

import torch
import pytest

from acf_functor.genesis import (
    DiscoveryStrength,
    DiscoveryType,
    FingerprintEngine,
    GenesisOrchestrator,
    GenesisReport,
    MathematicalDiscovery,
    ProgramGenerator,
    ProgramGenome,
    RelationDetector,
)


@pytest.fixture
def genesis() -> GenesisOrchestrator:
    return GenesisOrchestrator(
        domain=(-3.0, 3.0),
        identity_threshold=1e-6,
        dtype=torch.float64,
    )


class TestProgramGeneration:
    def test_random_generation(self) -> None:
        generator = ProgramGenerator()
        programs = generator.generate_random(20, seed=42)
        assert len(programs) == 20
        for genome, func in programs:
            assert isinstance(genome, ProgramGenome)
            assert callable(func)

    def test_structured_generation(self) -> None:
        generator = ProgramGenerator()
        programs = generator.generate_structured(["polynomial", "trigonometric"], n_variants=5, seed=42)
        assert len(programs) == 10

    def test_mutation(self) -> None:
        generator = ProgramGenerator()
        genome, _ = generator.generate_random(1, seed=42)[0]
        mutations = generator.mutate(genome, n_mutations=5)
        assert len(mutations) == 5
        for mutated_genome, mutated_func in mutations:
            assert callable(mutated_func)
            assert len(mutated_genome.coefficients) == len(genome.coefficients)

    def test_programs_are_evaluable(self) -> None:
        generator = ProgramGenerator()
        x = torch.linspace(-3, 3, 100, dtype=torch.float64)
        for genome, func in generator.generate_random(10, seed=42):
            y = func(x)
            assert y.shape == x.shape
            assert not torch.isnan(y).all()

    def test_genome_hash_unique(self) -> None:
        generator = ProgramGenerator()
        hashes = {genome.genome_hash for genome, _ in generator.generate_random(20, seed=42)}
        assert len(hashes) >= 10


class TestFingerprinting:
    def test_fingerprint_sin(self) -> None:
        engine = FingerprintEngine()
        fp = engine.fingerprint(torch.sin)
        assert fp is not None
        assert len(fp.evaluation_digest) > 0
        assert fp.symmetry_signature["odd"] > 0.9

    def test_fingerprint_cos(self) -> None:
        engine = FingerprintEngine()
        fp = engine.fingerprint(torch.cos)
        assert fp is not None
        assert fp.symmetry_signature["even"] > 0.9

    def test_fingerprint_rejects_nan(self) -> None:
        engine = FingerprintEngine()

        def bad_func(x: torch.Tensor) -> torch.Tensor:
            return torch.tensor(float("nan")).expand_as(x)

        assert engine.fingerprint(bad_func) is None

    def test_similar_functions_similar_fingerprints(self) -> None:
        engine = FingerprintEngine()
        fp1 = engine.fingerprint(torch.sin)
        fp2 = engine.fingerprint(lambda x: torch.sin(x) + 1e-10 * torch.randn_like(x))
        assert fp1 is not None and fp2 is not None
        assert fp1.similarity(fp2) > 0.9

    def test_different_functions_different_fingerprints(self) -> None:
        engine = FingerprintEngine()
        fp1 = engine.fingerprint(torch.sin)
        fp2 = engine.fingerprint(torch.exp)
        assert fp1 is not None and fp2 is not None
        assert fp1.similarity(fp2) < 0.5


class TestRelationDetection:
    def test_detect_identity(self) -> None:
        detector = RelationDetector(identity_threshold=1e-6)
        g1 = ProgramGenome("sin", [1.0], 1, 1, 1)
        g2 = ProgramGenome("sin_copy", [1.0], 1, 1, 2)
        disc = detector.detect_identity(torch.sin, torch.sin, g1, g2)
        assert disc is not None
        assert disc.discovery_type == DiscoveryType.ALGEBRAIC_IDENTITY
        assert disc.max_numerical_error < 1e-10

    def test_detect_non_identity(self) -> None:
        detector = RelationDetector(identity_threshold=1e-6)
        g1 = ProgramGenome("sin", [1.0], 1, 1, 1)
        g2 = ProgramGenome("cos", [1.0], 1, 1, 2)
        assert detector.detect_identity(torch.sin, torch.cos, g1, g2) is None

    def test_detect_even_symmetry(self) -> None:
        detector = RelationDetector(identity_threshold=1e-6)
        genome = ProgramGenome("cos", [1.0], 1, 1, 1)
        disc = detector.detect_symmetry(torch.cos, genome)
        assert disc is not None
        assert disc.discovery_type == DiscoveryType.SYMMETRY
        assert "even" in disc.description

    def test_detect_odd_symmetry(self) -> None:
        detector = RelationDetector(identity_threshold=1e-6)
        genome = ProgramGenome("sin", [1.0], 1, 1, 1)
        disc = detector.detect_symmetry(torch.sin, genome)
        assert disc is not None
        assert "odd" in disc.description

    def test_detect_exp_differential(self) -> None:
        detector = RelationDetector(identity_threshold=1e-4)
        genome = ProgramGenome("exp", [1.0], 1, 1, 1)
        disc = detector.detect_differential_relation(torch.exp, genome)
        if disc is not None:
            assert disc.discovery_type == DiscoveryType.DIFFERENTIAL_RELATION


class TestGenesisOrchestrator:
    def test_basic_run(self, genesis: GenesisOrchestrator) -> None:
        report = genesis.run(n_generations=2, programs_per_generation=30, seed=42, verbose=False)
        assert isinstance(report, GenesisReport)
        assert report.programs_generated > 0
        assert report.total_time_seconds > 0

    def test_discovers_something(self, genesis: GenesisOrchestrator) -> None:
        report = genesis.run(
            n_generations=3,
            programs_per_generation=50,
            seed=42,
            templates=["polynomial", "trigonometric"],
            verbose=False,
        )
        assert report.programs_survived_filter > 0

    def test_report_summary(self, genesis: GenesisOrchestrator) -> None:
        report = genesis.run(n_generations=2, programs_per_generation=20, seed=42, verbose=False)
        summary = report.summary()
        assert "Programs Generated" in summary
        assert "Discoveries" in summary

    def test_deterministic_with_seed(self) -> None:
        g1 = GenesisOrchestrator(dtype=torch.float64)
        g2 = GenesisOrchestrator(dtype=torch.float64)
        r1 = g1.run(n_generations=1, programs_per_generation=20, seed=42, verbose=False)
        r2 = g2.run(n_generations=1, programs_per_generation=20, seed=42, verbose=False)
        assert r1.programs_generated == r2.programs_generated
        assert r1.programs_survived_filter == r2.programs_survived_filter

    def test_known_identity_rediscovery(self) -> None:
        orchestrator = GenesisOrchestrator(domain=(-3.0, 3.0), identity_threshold=1e-6)
        detector = orchestrator.detector
        g1 = ProgramGenome("sin2_plus_cos2", [1.0], 2, 3, 1)
        g2 = ProgramGenome("one", [1.0], 1, 1, 2)
        disc = detector.detect_identity(
            lambda x: torch.sin(x) ** 2 + torch.cos(x) ** 2,
            lambda x: torch.ones_like(x),
            g1,
            g2,
        )
        assert disc is not None
        assert disc.discovery_type == DiscoveryType.ALGEBRAIC_IDENTITY
        assert disc.max_numerical_error < 1e-10

    def test_statistics_collected(self, genesis: GenesisOrchestrator) -> None:
        report = genesis.run(n_generations=2, programs_per_generation=20, seed=42, verbose=False)
        assert "discovery_types" in report.statistics
        assert "mean_persistence" in report.statistics

    def test_discovery_has_truth_value(self, genesis: GenesisOrchestrator) -> None:
        report = genesis.run(
            n_generations=3,
            programs_per_generation=50,
            seed=42,
            templates=["polynomial", "trigonometric"],
            verbose=False,
        )
        for discovery in report.discoveries:
            assert 0 <= discovery.truth_value <= 1

    def test_discovery_summary_format(self) -> None:
        discovery = MathematicalDiscovery(
            discovery_id="test",
            discovery_type=DiscoveryType.ALGEBRAIC_IDENTITY,
            strength=DiscoveryStrength.NUMERICAL,
            programs=[],
            description="f = g",
            formal_statement="forall x: f(x) = g(x)",
            numerical_evidence={"error": 1e-10},
            persistence_score=0.99,
            perturbation_stability=0.95,
            max_numerical_error=1e-10,
            domain_tested=(-3, 3),
            n_test_points=2000,
            truth_value=0.99,
            discovery_time=1.5,
            generation=3,
        )
        summary = discovery.summary()
        assert "ALGEBRAIC_IDENTITY" in summary
        assert "f = g" in summary
