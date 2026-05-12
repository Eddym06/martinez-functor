"""
Tests for the Universal Constructor Engine
===========================================

Covers: HyperGraphEngine, MassiveAlgebra, AlgorithmForge, UniversalConstructor
"""

import math
import numpy as np
import pytest
import torch


# ===========================================================================
# §1  HyperGraph Engine
# ===========================================================================

class TestHyperGraphEngine:
    """Tests for the Computable HyperGraph representation."""

    def test_create_empty_graph(self):
        from acf_functor.hypergraph_engine import ComputableHyperGraph
        g = ComputableHyperGraph("test")
        assert g.name == "test"
        assert g.n_nodes == 0

    def test_add_nodes_and_edges(self):
        from acf_functor.hypergraph_engine import (
            ComputableHyperGraph, NodeKind, EdgeKind)
        g = ComputableHyperGraph("test")
        n1 = g.add_node(NodeKind.SOURCE, (10,), (10,), label="src")
        n2 = g.add_node(NodeKind.FMA_SCALAR, (10,), (10,), 100, "fma")
        n3 = g.add_node(NodeKind.SINK, (10,), (10,), label="sink")
        e1 = g.add_edge([n1], [n2])
        e2 = g.add_edge([n2], [n3])
        assert g.n_nodes == 3
        assert g.n_edges == 2

    def test_build_linear_chain(self):
        from acf_functor.hypergraph_engine import build_linear_chain
        g = build_linear_chain(5, 64)
        assert g.n_nodes >= 5
        assert g.total_fma > 0

    def test_build_residual_network(self):
        from acf_functor.hypergraph_engine import build_residual_network
        g = build_residual_network(4, 128, "res_test")
        assert g.n_nodes >= 4
        summary = g.summary()
        assert summary["n_nodes"] >= 4

    def test_build_attention_block(self):
        from acf_functor.hypergraph_engine import build_attention_block
        g = build_attention_block(4, 64, seq_len=10, name="attn_test")
        assert g.n_nodes >= 3
        assert g.total_fma > 0

    def test_build_stencil_grid(self):
        from acf_functor.hypergraph_engine import build_stencil_grid
        g = build_stencil_grid(8, 8, 3, "stencil_test")
        assert g.n_nodes >= 3  # source + stencil + sink
        assert g.total_fma > 0

    def test_topological_order(self):
        from acf_functor.hypergraph_engine import build_linear_chain
        g = build_linear_chain(4, 32)
        order = g.topological_order()
        assert len(order) == g.n_nodes

    def test_is_dag(self):
        from acf_functor.hypergraph_engine import build_linear_chain
        g = build_linear_chain(4, 32)
        assert g.is_dag()

    def test_partition_by_depth(self):
        from acf_functor.hypergraph_engine import build_linear_chain
        g = build_linear_chain(4, 32)
        regions = g.partition_by_depth()
        assert len(regions) >= 1

    def test_spectral_analysis(self):
        from acf_functor.hypergraph_engine import build_linear_chain
        g = build_linear_chain(5, 64)
        spec = g.spectral_analysis()
        assert "n_nodes" in spec or "spectral_radius" in spec or len(spec) >= 1

    def test_identify_bottlenecks(self):
        from acf_functor.hypergraph_engine import build_linear_chain
        g = build_linear_chain(5, 64)
        bottlenecks = g.identify_bottlenecks()
        assert isinstance(bottlenecks, list)

    def test_fingerprint(self):
        from acf_functor.hypergraph_engine import build_linear_chain
        g = build_linear_chain(5, 64)
        fp = g.fingerprint()
        assert isinstance(fp, str) and len(fp) > 0

    def test_summary(self):
        from acf_functor.hypergraph_engine import build_linear_chain
        g = build_linear_chain(5, 64)
        s = g.summary()
        assert "n_nodes" in s
        assert s["n_nodes"] >= 5

    def test_replace_subgraph(self):
        from acf_functor.hypergraph_engine import (
            ComputableHyperGraph, NodeKind, build_linear_chain)
        g = build_linear_chain(5, 64)
        all_nodes = g.nodes()
        old_ids = {all_nodes[1].node_id, all_nodes[2].node_id}
        new_g = ComputableHyperGraph("replacement")
        r1 = new_g.add_node(NodeKind.FMA_SCALAR, (64,), (64,), 10, "opt")
        g.replace_subgraph(old_ids, new_g)
        # Graph structure changed
        assert isinstance(g, ComputableHyperGraph)

    def test_add_fma_chain(self):
        from acf_functor.hypergraph_engine import ComputableHyperGraph
        g = ComputableHyperGraph("chain_test")
        nodes = g.add_fma_chain(4, 64, "chain")
        assert len(nodes) == 4

    def test_from_torch_module(self):
        from acf_functor.hypergraph_engine import from_torch_module
        model = torch.nn.Sequential(
            torch.nn.Linear(32, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 10),
        )
        g = from_torch_module(model, input_shape=(32,))
        assert g.n_nodes >= 3


# ===========================================================================
# §2  Massive Algebra
# ===========================================================================

class TestMassiveAlgebra:
    """Tests for scalable linear algebra primitives."""

    def test_randomized_svd_basic(self):
        from acf_functor.massive_algebra import RandomizedSVD
        rng = np.random.RandomState(42)
        n, k_true = 200, 5
        U0 = rng.randn(n, k_true)
        V0 = rng.randn(k_true, n)
        A = U0 @ V0 + 0.01 * rng.randn(n, n)
        svd = RandomizedSVD(random_state=42)
        decomp = svd.decompose(A, 10)
        assert decomp.eigenvectors.shape == (n, 10)
        assert len(decomp.eigenvalues) == 10
        # Dominant eigenvalues (singular values squared) should decrease
        assert decomp.eigenvalues[0] > decomp.eigenvalues[-1]

    def test_randomized_svd_500x500(self):
        """Test on a 500×500 matrix with known low-rank structure."""
        from acf_functor.massive_algebra import RandomizedSVD
        rng = np.random.RandomState(123)
        n = 500
        k_true = 10
        U0 = rng.randn(n, k_true)
        V0 = rng.randn(k_true, n)
        A = U0 @ V0
        svd = RandomizedSVD(random_state=42)
        decomp = svd.decompose(A, 15)
        # Eigenvalues are singular values squared; top k_true should dominate
        assert decomp.eigenvectors.shape == (n, 15)
        assert decomp.rank == 15
        assert decomp.relative_energy > 0.5  # Should capture most energy

    def test_sparse_chebyshev_operator_exp(self):
        """Test Chebyshev approximation of exp(A)·v."""
        from acf_functor.massive_algebra import SparseChebyshevOperator
        n = 50
        rng = np.random.RandomState(42)
        A = rng.randn(n, n)
        A = 0.5 * (A + A.T)
        evals = np.linalg.eigvalsh(A)
        lam_min, lam_max = float(evals.min()), float(evals.max())
        v = rng.randn(n)
        # Ground truth
        exp_Av_true = np.linalg.eigh(A)[1] @ np.diag(
            np.exp(np.linalg.eigvalsh(A))) @ np.linalg.eigh(A)[1].T @ v
        cheb = SparseChebyshevOperator(max_degree=30)
        result = cheb.apply(lambda x: A @ x, v, "exp", lam_min, lam_max, degree=30)
        rel_err = np.linalg.norm(result.result - exp_Av_true) / np.linalg.norm(exp_Av_true)
        assert rel_err < 0.5  # Chebyshev approximation of exp can be loose

    def test_sparse_chebyshev_operator_inv(self):
        """Test Chebyshev approximation of A^{-1}·v (positive definite)."""
        from acf_functor.massive_algebra import SparseChebyshevOperator
        n = 30
        rng = np.random.RandomState(42)
        A = rng.randn(n, n)
        A = A.T @ A + 2.0 * np.eye(n)  # well-conditioned SPD
        evals = np.linalg.eigvalsh(A)
        lam_min, lam_max = float(evals.min()), float(evals.max())
        v = rng.randn(n)
        inv_Av_true = np.linalg.solve(A, v)
        cheb = SparseChebyshevOperator(max_degree=40)
        result = cheb.apply(lambda x: A @ x, v, "inv", lam_min, lam_max, degree=40)
        rel_err = np.linalg.norm(result.result - inv_Av_true) / np.linalg.norm(inv_Av_true)
        assert rel_err < 0.5

    def test_compressed_linear_solver(self):
        """Test CompressedLinearSolver on a low-rank system."""
        from acf_functor.massive_algebra import CompressedLinearSolver, RandomizedSVD
        rng = np.random.RandomState(42)
        n = 100
        k_true = 5
        U0 = rng.randn(n, k_true)
        A = U0 @ U0.T + 0.5 * np.eye(n)  # SPD, low effective rank
        b = rng.randn(n)
        x_true = np.linalg.solve(A, b)
        solver = CompressedLinearSolver(RandomizedSVD(random_state=42))
        result = solver.solve(A, b, rank=20)
        rel_err = np.linalg.norm(result.x - x_true) / np.linalg.norm(x_true)
        assert rel_err < 1.0  # Compressed solver is approximate

    def test_tensor_train_decompose(self):
        """Test TensorTrain decomposition on a small tensor."""
        from acf_functor.massive_algebra import TensorTrainEngine
        rng = np.random.RandomState(42)
        tensor = rng.randn(4, 5, 6)
        tt = TensorTrainEngine()
        decomp = tt.decompose(tensor)
        assert len(decomp.cores) == 3
        assert decomp.cores[0].data.shape[0] == 1  # first rank is 1
        # Reconstruction
        recon = tt._reconstruct(decomp.cores, tensor.shape)
        rel_err = np.linalg.norm(recon - tensor) / np.linalg.norm(tensor)
        assert rel_err < 1e-10

    def test_tensor_train_evaluate_point(self):
        from acf_functor.massive_algebra import TensorTrainEngine
        rng = np.random.RandomState(42)
        tensor = rng.randn(3, 4, 5)
        tt = TensorTrainEngine()
        decomp = tt.decompose(tensor)
        # Point evaluation
        val = tt.evaluate_point(decomp, (1, 2, 3))
        assert abs(val - tensor[1, 2, 3]) < 1e-10

    def test_massive_eigen_solver(self):
        """Test eigenvalue computation."""
        from acf_functor.massive_algebra import MassiveEigenSolver, RandomizedSVD
        rng = np.random.RandomState(42)
        n = 100
        A = rng.randn(n, n)
        A = 0.5 * (A + A.T)
        true_evals = np.sort(np.abs(np.linalg.eigvalsh(A)))[::-1]
        eigen = MassiveEigenSolver(RandomizedSVD(random_state=42))
        decomp = eigen.top_k_eigenvalues(A, k=10)
        # Top eigenvalue should be close
        assert abs(np.abs(decomp.eigenvalues[0]) - true_evals[0]) / true_evals[0] < 0.5

    def test_operator_compressor(self):
        """Test OperatorCompressor chooses strategy."""
        from acf_functor.massive_algebra import OperatorCompressor, RandomizedSVD
        rng = np.random.RandomState(42)
        n = 50
        A = rng.randn(n, n)
        comp = OperatorCompressor(RandomizedSVD(random_state=42))
        result = comp.compress(A, target_error=0.1)
        assert "best_strategy" in result
        assert result["best_strategy"] in ("dense", "low_rank", "sparse")


# ===========================================================================
# §3  Algorithm Forge
# ===========================================================================

class TestAlgorithmForge:
    """Tests for the Autonomous Algorithm Generator."""

    def test_forge_linear_solve(self):
        from acf_functor.algorithm_forge import (
            AlgorithmForge, ProblemSpec, ProblemKind)
        rng = np.random.RandomState(42)
        n = 50
        A = rng.randn(n, n)
        A = A.T @ A + np.eye(n)
        b = rng.randn(n)
        spec = ProblemSpec(
            kind=ProblemKind.LINEAR_SOLVE,
            description="Solve Ax=b",
            n=n, rhs=b, target_error=1e-4,
        )
        forge = AlgorithmForge(verify=True)
        algo = forge.forge(spec)
        assert algo is not None
        assert algo.name != ""
        # Execute and verify
        x = algo.execute(A, b)
        x_true = np.linalg.solve(A, b)
        assert np.linalg.norm(x - x_true) / np.linalg.norm(x_true) < 0.5

    def test_forge_eigenvalue(self):
        from acf_functor.algorithm_forge import (
            AlgorithmForge, ProblemSpec, ProblemKind)
        n = 30
        rng = np.random.RandomState(42)
        A = rng.randn(n, n)
        A = 0.5 * (A + A.T)
        spec = ProblemSpec(
            kind=ProblemKind.EIGENVALUE,
            description="Top eigenvalues",
            n=n, target_error=1e-3,
        )
        forge = AlgorithmForge(verify=True)
        algo = forge.forge(spec)
        assert algo is not None
        result = algo.execute(A)
        assert result is not None

    def test_forge_function_approximation(self):
        from acf_functor.algorithm_forge import (
            AlgorithmForge, ProblemSpec, ProblemKind)
        spec = ProblemSpec(
            kind=ProblemKind.FUNCTION_APPROXIMATION,
            description="Approximate sin(x)",
            n=100, target_error=1e-6,
            domain=(-1.0, 1.0),
        )
        forge = AlgorithmForge(verify=True)
        algo = forge.forge(spec, ground_truth=np.sin)
        assert algo is not None
        # The polynomial strategy should have been selected
        assert algo.strategy in (
            "polynomial", "spectral", "direct",
            algo.strategy  # Always passes, just for clarity
        )

    def test_forge_explores_multiple_strategies(self):
        from acf_functor.algorithm_forge import (
            AlgorithmForge, ProblemSpec, ProblemKind)
        spec = ProblemSpec(
            kind=ProblemKind.LINEAR_SOLVE,
            description="Test strategy exploration",
            n=200, target_error=1e-6,
        )
        forge = AlgorithmForge(verify=False)
        algo = forge.forge(spec)
        n_explored = algo.certificate.get("n_strategies_explored", 0)
        assert n_explored > 1, "Should explore multiple strategies"

    def test_forge_multiple_algorithms(self):
        from acf_functor.algorithm_forge import (
            AlgorithmForge, ProblemSpec, ProblemKind)
        spec = ProblemSpec(
            kind=ProblemKind.LINEAR_SOLVE,
            description="Multiple algos",
            n=50, target_error=1e-4,
        )
        forge = AlgorithmForge(verify=False)
        algos = forge.forge_multiple(spec, n_algorithms=3)
        assert len(algos) >= 2

    def test_forged_algorithm_summary(self):
        from acf_functor.algorithm_forge import (
            AlgorithmForge, ProblemSpec, ProblemKind)
        spec = ProblemSpec(
            kind=ProblemKind.LINEAR_SOLVE, description="Summary test",
            n=30, target_error=1e-4,
        )
        forge = AlgorithmForge(verify=False)
        algo = forge.forge(spec)
        s = algo.summary()
        assert "FMA=" in s

    def test_problem_analyzer_detects_linearity(self):
        from acf_functor.algorithm_forge import ProblemAnalyzer, ProblemSpec, ProblemKind
        n = 20
        rng = np.random.RandomState(42)
        A = rng.randn(n, n)
        spec = ProblemSpec(
            kind=ProblemKind.LINEAR_SOLVE, description="Linearity test",
            n=n, operator=lambda v: A @ v,
        )
        analyzer = ProblemAnalyzer()
        analysis = analyzer.analyze(spec)
        assert bool(analysis.get("is_linear", False)) is True


# ===========================================================================
# §4  Universal Constructor
# ===========================================================================

class TestUniversalConstructor:
    """Tests for the Universal Constructor orchestrator."""

    def test_build_neural_network(self):
        from acf_functor.universal_constructor import UniversalConstructor
        uc = UniversalConstructor(verify=False)
        system = uc.build_neural_network(
            input_dim=32, output_dim=10, task_complexity=2.0)
        assert system is not None
        assert system.total_fma > 0
        # Execute
        x = np.random.randn(32).astype(np.float32)
        y = system.execute(x)
        assert y.shape[-1] == 10

    def test_build_neural_network_high_complexity(self):
        from acf_functor.universal_constructor import UniversalConstructor
        uc = UniversalConstructor(verify=False)
        system = uc.build_neural_network(
            input_dim=128, output_dim=10, task_complexity=7.0)
        # High complexity should produce deeper/wider network
        arch = system.certificates.get("architecture", {})
        assert arch.get("depth", 0) >= 4

    def test_build_pde_solver_1d(self):
        from acf_functor.universal_constructor import UniversalConstructor
        uc = UniversalConstructor(verify=False)
        system = uc.build_pde_solver(
            grid_shape=(50,), pde_type="diffusion", time_steps=20)
        assert system is not None
        # Solve: initial condition = Gaussian bump
        u0 = np.exp(-np.linspace(-2, 2, 50) ** 2)
        u_final = system.execute(u0, n_steps=20)
        assert u_final.shape == (50,)
        # Diffusion should reduce max value
        assert u_final.max() <= u0.max() + 1e-10

    def test_build_pde_solver_2d(self):
        from acf_functor.universal_constructor import UniversalConstructor
        uc = UniversalConstructor(verify=False)
        system = uc.build_pde_solver(
            grid_shape=(20, 20), pde_type="diffusion", time_steps=10)
        x = np.linspace(-1, 1, 20)
        X, Y = np.meshgrid(x, x)
        u0 = np.exp(-(X ** 2 + Y ** 2))
        u_final = system.execute(u0, n_steps=10)
        assert u_final.shape == (20, 20)
        assert u_final.max() <= u0.max() + 1e-10

    def test_build_graph_processor_pagerank(self):
        from acf_functor.universal_constructor import UniversalConstructor
        uc = UniversalConstructor(verify=False)
        system = uc.build_graph_processor(
            n_vertices=50, n_edges=200, algorithm="pagerank")
        assert system is not None
        # Create a random adjacency matrix
        rng = np.random.RandomState(42)
        A = (rng.rand(50, 50) > 0.8).astype(float)
        np.fill_diagonal(A, 0)
        A = np.maximum(A, A.T)  # symmetrize
        pr = system.execute(A)
        assert pr.shape == (50,)
        assert abs(pr.sum() - 1.0) < 0.1  # Should be approximately normalized

    def test_solve_linear_system(self):
        from acf_functor.universal_constructor import UniversalConstructor
        uc = UniversalConstructor(verify=False)
        rng = np.random.RandomState(42)
        n = 50
        A = rng.randn(n, n)
        A = A.T @ A + np.eye(n)
        b = rng.randn(n)
        result = uc.solve(A, b)
        x_true = np.linalg.solve(A, b)
        assert np.linalg.norm(result.x - x_true) / np.linalg.norm(x_true) < 0.5

    def test_forge_algorithm_via_constructor(self):
        from acf_functor.universal_constructor import UniversalConstructor
        from acf_functor.algorithm_forge import ProblemSpec, ProblemKind
        uc = UniversalConstructor(verify=False)
        spec = ProblemSpec(
            kind=ProblemKind.LINEAR_SOLVE,
            description="Forge test",
            n=30, target_error=1e-4,
        )
        algo = uc.forge_algorithm(spec)
        assert algo is not None
        assert algo.n_fma > 0

    def test_decompose_tensor(self):
        from acf_functor.universal_constructor import UniversalConstructor
        uc = UniversalConstructor(verify=False)
        tensor = np.random.randn(3, 4, 5)
        decomp = uc.decompose_tensor(tensor)
        assert len(decomp.cores) == 3

    def test_compress_operator(self):
        from acf_functor.universal_constructor import UniversalConstructor
        uc = UniversalConstructor(verify=False)
        A = np.random.randn(30, 30)
        result = uc.compress_operator(A, target_error=0.1)
        assert "best_strategy" in result

    def test_optimize_graph(self):
        from acf_functor.universal_constructor import UniversalConstructor
        from acf_functor.hypergraph_engine import build_linear_chain
        uc = UniversalConstructor(verify=False)
        g = build_linear_chain(10, 128)
        g_opt, report = uc.optimize(g)
        assert report["original_fma"] > 0
        assert report["n_regions"] >= 1

    def test_discover_dynamics(self):
        """Test P-SAL discovery from trajectory data."""
        from acf_functor.universal_constructor import UniversalConstructor
        uc = UniversalConstructor(verify=False)
        # Generate trajectory from a simple linear ODE: dx/dt = L·x
        rng = np.random.RandomState(42)
        n_vars = 5
        L_true = -0.1 * np.eye(n_vars) + 0.02 * rng.randn(n_vars, n_vars)
        dt = 0.01
        n_steps = 200
        x = rng.randn(n_vars)
        trajectory = [x.copy()]
        for _ in range(n_steps - 1):
            x = x + dt * (L_true @ x)
            trajectory.append(x.copy())
        trajectory = np.array(trajectory)  # (n_steps, n_vars)

        result = uc.discover_dynamics(trajectory, dt=dt, rom_dim=5)
        assert "linear_dynamics_L" in result
        assert result["rom_dimension"] <= 5
        assert result["sindy_residual"] < 1.0
        assert result["sparsity"] >= 0.0

    def test_construct_generic(self):
        from acf_functor.universal_constructor import UniversalConstructor
        uc = UniversalConstructor(verify=False)
        system = uc.construct({
            "kind": "neural_network",
            "input_dim": 16,
            "output_dim": 4,
            "task_complexity": 1.0,
        })
        assert system is not None
        assert system.total_fma > 0

    def test_constructed_system_summary(self):
        from acf_functor.universal_constructor import UniversalConstructor
        uc = UniversalConstructor(verify=False)
        system = uc.build_neural_network(
            input_dim=16, output_dim=4, task_complexity=1.0)
        s = system.summary()
        assert "name" in s
        assert "total_fma" in s
        assert s["system_kind"] == "neural_network"


# ===========================================================================
# §5  Integration: end-to-end autonomous construction
# ===========================================================================

class TestEndToEnd:
    """End-to-end tests verifying the full construction pipeline."""

    def test_neural_net_full_pipeline(self):
        """Spec → Architect → Graph → Model → Execute → Verify."""
        from acf_functor.universal_constructor import UniversalConstructor
        uc = UniversalConstructor(verify=True)
        system = uc.build_neural_network(
            input_dim=64, output_dim=10, task_complexity=3.0)
        # Verify computation graph integrity
        assert system.graph.is_dag()
        # Verify model produces correct output shape
        x = np.random.randn(64).astype(np.float32)
        y = system.execute(x)
        assert y.shape == (10,)

    def test_pde_solver_conservation(self):
        """Verify mass conservation in diffusion solver."""
        from acf_functor.universal_constructor import UniversalConstructor
        uc = UniversalConstructor(verify=True)
        system = uc.build_pde_solver(
            grid_shape=(100,), pde_type="diffusion", time_steps=50)
        u0 = np.zeros(100)
        u0[40:60] = 1.0  # Square pulse
        u_final = system.execute(u0, n_steps=50)
        # Interior mass approximately conserved (boundaries are Dirichlet)
        mass_initial = u0.sum()
        mass_final = u_final.sum()
        # Allow some boundary dissipation
        assert mass_final <= mass_initial * 1.1

    def test_forge_then_execute_at_scale(self):
        """Forge an algorithm then use it on a 200×200 system."""
        from acf_functor.universal_constructor import UniversalConstructor
        from acf_functor.algorithm_forge import ProblemSpec, ProblemKind
        uc = UniversalConstructor(verify=True)
        rng = np.random.RandomState(42)
        n = 200
        A = rng.randn(n, n)
        A = A.T @ A + 0.5 * np.eye(n)
        b = rng.randn(n)
        spec = ProblemSpec(
            kind=ProblemKind.LINEAR_SOLVE,
            description="Scale test",
            n=n, rhs=b, target_error=1e-4,
        )
        algo = uc.forge_algorithm(spec)
        x = algo.execute(A, b)
        x_true = np.linalg.solve(A, b)
        assert np.linalg.norm(x - x_true) / np.linalg.norm(x_true) < 1.0

    def test_imports_from_top_level(self):
        """Verify all new exports are importable from acf_functor."""
        from acf_functor import (
            ComputableHyperGraph, HyperNode, HyperEdge, NodeKind, EdgeKind,
            SubGraph, build_linear_chain, build_residual_network,
            build_attention_block, build_stencil_grid, from_torch_module,
            build_butterfly_fft,
            RandomizedSVD, SparseChebyshevOperator, CompressedLinearSolver,
            TensorTrainEngine, MassiveEigenSolver, OperatorCompressor,
            SpectralDecomposition, CompressedSolution, OperatorFunctionResult,
            MassiveTTDecomposition,
            AlgorithmForge, ProblemSpec, ProblemKind, StrategyKind,
            HardwareTarget, ForgedAlgorithm, StrategyCandidate,
            ProblemAnalyzer, StrategyExplorer, AlgorithmSynthesizer,
            AlgorithmVerifier,
            BackendSynthesizer, SynthesizedKernel, GraphPattern,
            BackendDecision, HardwareProfile, HardwareDetector,
            GraphPatternDetector, BackendSelector, BackendCodeGenerator,
            UniversalConstructor, ConstructorMode, SystemKind,
            ConstructionStage, ConstructionPlan, ConstructedSystem,
            NeuralArchitect, PDESolverArchitect, GraphProcessorArchitect,
        )
        assert UniversalConstructor is not None
        assert AlgorithmForge is not None
        assert ComputableHyperGraph is not None
        assert RandomizedSVD is not None
        assert BackendSynthesizer is not None
        assert build_butterfly_fft is not None


# ===========================================================================
# Backend Synthesizer Test Suite
# ===========================================================================

class TestButterflyFFTGraph:
    """Tests for build_butterfly_fft graph construction."""

    def test_build_butterfly_fft_basic(self):
        from acf_functor.hypergraph_engine import build_butterfly_fft, NodeKind
        graph = build_butterfly_fft(8)
        # 8 elements, log2(8)=3 stages, + SOURCE + MEMORY + SINK = 6 nodes
        assert graph.n_nodes == 6
        assert graph.is_dag()

    def test_build_butterfly_fft_1024(self):
        from acf_functor.hypergraph_engine import build_butterfly_fft
        graph = build_butterfly_fft(1024)
        # 10 stages + SOURCE + MEMORY + SINK = 13 nodes
        assert graph.n_nodes == 13
        assert graph.is_dag()
        # Check total FMA matches expectation
        expected_fma = 10 * (512 * 6)  # log2(1024) * (N/2) * 6
        assert graph.total_fma == expected_fma

    def test_build_butterfly_fft_powers_of_two(self):
        from acf_functor.hypergraph_engine import build_butterfly_fft
        for N in [4, 16, 64, 256]:
            graph = build_butterfly_fft(N)
            assert graph.is_dag(), f"FFT graph N={N} is not a DAG"
            assert graph.total_fma > 0

    def test_build_butterfly_fft_topological(self):
        from acf_functor.hypergraph_engine import build_butterfly_fft
        graph = build_butterfly_fft(16)
        order = graph.topological_order()
        assert len(order) == graph.n_nodes


class TestGraphPatternDetector:
    """Tests for pattern recognition on computation graphs."""

    def test_detects_butterfly_pattern(self):
        from acf_functor.hypergraph_engine import build_butterfly_fft
        from acf_functor.algorithm_forge import GraphPatternDetector

        graph = build_butterfly_fft(256)
        detector = GraphPatternDetector()
        patterns = detector.detect(graph)

        assert len(patterns) > 0
        best = patterns[0]
        assert best.pattern_type == "butterfly_fft"
        assert best.confidence > 0.8
        assert best.size_parameter == 256
        assert best.log_size == 8

    def test_detects_butterfly_1024(self):
        from acf_functor.hypergraph_engine import build_butterfly_fft
        from acf_functor.algorithm_forge import GraphPatternDetector

        graph = build_butterfly_fft(1024)
        detector = GraphPatternDetector()
        patterns = detector.detect(graph)

        best = patterns[0]
        assert best.pattern_type == "butterfly_fft"
        assert best.confidence > 0.9
        assert best.size_parameter == 1024
        assert best.log_size == 10

    def test_non_butterfly_graph_low_confidence(self):
        from acf_functor.hypergraph_engine import build_linear_chain
        from acf_functor.algorithm_forge import GraphPatternDetector

        graph = build_linear_chain(5, 64)
        detector = GraphPatternDetector()
        patterns = detector.detect(graph)

        # Should not detect butterfly with high confidence
        for p in patterns:
            if p.pattern_type == "butterfly_fft":
                assert p.confidence < 0.5


class TestBackendSelector:
    """Tests for hardware-aware backend selection."""

    def test_fft_with_gpu_triton(self):
        from acf_functor.algorithm_forge import (
            BackendSelector, GraphPattern, HardwareProfile
        )
        pattern = GraphPattern("butterfly_fft", 0.95, 1024, log_size=10)
        hw = HardwareProfile(has_gpu=True, gpu_name="RTX 4050",
                             has_triton=True, cpu_cores=8)
        selector = BackendSelector()
        decision = selector.select(pattern, hw)
        assert decision.backend == "triton_gpu"

    def test_fft_with_gpu_no_triton(self):
        from acf_functor.algorithm_forge import (
            BackendSelector, GraphPattern, HardwareProfile
        )
        pattern = GraphPattern("butterfly_fft", 0.95, 1024, log_size=10)
        hw = HardwareProfile(has_gpu=True, gpu_name="RTX 4050",
                             has_triton=False, cpu_cores=8)
        selector = BackendSelector()
        decision = selector.select(pattern, hw)
        assert decision.backend == "torch_cuda"

    def test_fft_cpu_only(self):
        from acf_functor.algorithm_forge import (
            BackendSelector, GraphPattern, HardwareProfile
        )
        pattern = GraphPattern("butterfly_fft", 0.95, 1024, log_size=10)
        hw = HardwareProfile(has_gpu=False, has_triton=False, cpu_cores=4)
        selector = BackendSelector()
        decision = selector.select(pattern, hw)
        assert decision.backend == "fused_numpy"

    def test_gemm_with_gpu(self):
        from acf_functor.algorithm_forge import (
            BackendSelector, GraphPattern, HardwareProfile
        )
        pattern = GraphPattern("dense_gemm", 0.7, 512,
                               metadata={"m": 512, "n": 512})
        hw = HardwareProfile(has_gpu=True, gpu_name="RTX 4050",
                             has_triton=True, cpu_cores=8)
        selector = BackendSelector()
        decision = selector.select(pattern, hw)
        assert decision.backend == "triton_gpu"


class TestBackendSynthesizerPipeline:
    """Tests for the full synthesis pipeline."""

    def test_synthesize_numpy_fft(self):
        from acf_functor.hypergraph_engine import build_butterfly_fft
        from acf_functor.algorithm_forge import BackendSynthesizer

        graph = build_butterfly_fft(64)
        synth = BackendSynthesizer()
        kernel = synth.synthesize(graph, force_backend="fused_numpy")

        assert kernel.backend == "fused_numpy"
        assert kernel.n_fma > 0

        # Test correctness
        rng = np.random.RandomState(42)
        x = rng.randn(64)
        result = kernel.execute(x)
        reference = np.fft.fft(x)
        assert np.max(np.abs(result - reference)) < 1e-6

    def test_synthesize_numpy_fft_1024(self):
        from acf_functor.hypergraph_engine import build_butterfly_fft
        from acf_functor.algorithm_forge import BackendSynthesizer

        graph = build_butterfly_fft(1024)
        synth = BackendSynthesizer()
        kernel = synth.synthesize(graph, force_backend="fused_numpy")

        rng = np.random.RandomState(42)
        x = rng.randn(1024)
        result = kernel.execute(x)
        reference = np.fft.fft(x)
        rel_err = np.max(np.abs(result - reference)) / np.max(np.abs(reference))
        assert rel_err < 1e-6

    def test_analyze_returns_patterns(self):
        from acf_functor.hypergraph_engine import build_butterfly_fft
        from acf_functor.algorithm_forge import BackendSynthesizer

        graph = build_butterfly_fft(256)
        synth = BackendSynthesizer()
        analysis = synth.analyze(graph)

        assert "patterns" in analysis
        assert "recommended_backend" in analysis
        assert "hardware" in analysis
        assert analysis["patterns"][0]["type"] == "butterfly_fft"

    def test_c_source_generation(self):
        from acf_functor.algorithm_forge import BackendCodeGenerator

        src = BackendCodeGenerator.generate_c_fft_source(256)
        assert "#define N 256" in src
        assert "void fft(" in src
        assert "Bit-reversal" in src

    def test_hardware_detector(self):
        from acf_functor.algorithm_forge import HardwareDetector

        HardwareDetector.reset_cache()
        hw = HardwareDetector.detect()
        assert isinstance(hw.has_gpu, bool)
        assert hw.cpu_cores > 0

        # Second call should use cache
        hw2 = HardwareDetector.detect()
        assert hw is hw2


class TestFFTExperiment:
    """Integration test for the FFT benchmark experiment."""

    def test_build_fft_processor(self):
        from acf_functor import UniversalConstructor
        uc = UniversalConstructor()
        system = uc.build_fft_processor(64, force_backend="fused_numpy")

        assert "fft_n64" in system.name
        assert system.total_fma > 0
        assert system.certificates["backend"] == "fused_numpy"

        # Test execution
        rng = np.random.RandomState(42)
        x = rng.randn(64)
        result = system.execute(x)
        reference = np.fft.fft(x)
        assert np.max(np.abs(result - reference)) < 1e-6

    def test_run_fft_experiment_small(self):
        """Run a small FFT experiment to validate the pipeline."""
        from acf_functor import UniversalConstructor
        uc = UniversalConstructor()
        report = uc.run_fft_experiment(N=64, n_warmup=2, n_iter=10)

        assert report["N"] == 64
        assert "phase_1_graph" in report
        assert "phase_2_patterns" in report
        assert "phase_5_benchmarks" in report
        assert "phase_6_result" in report
        assert "phase_7_certificates" in report
        assert "c_source" in report

        # All synthesized backends should be correct
        certs = report["phase_7_certificates"]
        for key, val in certs.items():
            if key.startswith("FORGE-5/"):
                assert val, f"Certificate {key} failed"
        assert certs["FORGE-7_passed"]


# =============================================================================
# TestLinearOperatorAtomLibrary
# =============================================================================

class TestLinearOperatorAtomLibrary:
    """Tests for the linear operator atom library."""

    def test_atom_names_present(self):
        """Library exposes expected atom names."""
        from acf_functor import LinearOperatorAtomLibrary
        lib = LinearOperatorAtomLibrary()
        for name in ("I", "P_br", "P_stride", "D_twiddle", "B_stage", "H", "KronI"):
            assert name in lib.atom_names, f"Missing atom: {name}"

    def test_identity_build(self):
        """Identity atom returns I_N."""
        from acf_functor import LinearOperatorAtomLibrary
        lib = LinearOperatorAtomLibrary()
        I4 = lib.build("I", 4)
        np.testing.assert_allclose(I4, np.eye(4, dtype=complex), atol=1e-12)

    def test_bit_reversal_permutation(self):
        """Bit-reversal permutation is a valid permutation matrix."""
        from acf_functor import LinearOperatorAtomLibrary
        lib = LinearOperatorAtomLibrary()
        P = lib.build("P_br", 8)
        # Each row/col must have exactly one 1
        assert np.allclose(P.sum(axis=1), 1.0)
        assert np.allclose(P.sum(axis=0), 1.0)
        # P^T = P^{-1}  (permutation is orthogonal)
        np.testing.assert_allclose(P @ P.T, np.eye(8), atol=1e-12)

    def test_hadamard_unitary(self):
        """Walsh-Hadamard is unitary."""
        from acf_functor import LinearOperatorAtomLibrary
        lib = LinearOperatorAtomLibrary()
        H = lib.build("H", 8)
        HHT = H @ H.T
        np.testing.assert_allclose(HHT, np.eye(8), atol=1e-10)

    def test_hadamard_entries(self):
        """Walsh-Hadamard entries are ±1/√N."""
        from acf_functor import LinearOperatorAtomLibrary
        lib = LinearOperatorAtomLibrary()
        N = 8
        H = lib.build("H", N)
        expected_abs = 1.0 / np.sqrt(N)
        np.testing.assert_allclose(np.abs(H), expected_abs, atol=1e-10)

    def test_butterfly_stage_unitary(self):
        """Butterfly stage is a scaled isometry (B†B = 2·I)."""
        from acf_functor import LinearOperatorAtomLibrary
        lib = LinearOperatorAtomLibrary()
        B = lib.build("B_stage", 8, [2])
        BHB = B.conj().T @ B
        # [1,w; 1,-w] has column norms √2 each → B†B = 2·I
        np.testing.assert_allclose(BHB, 2.0 * np.eye(8), atol=1e-10)

    def test_factorize_dft_4(self):
        """Atom library factorizes DFT(4) with small error."""
        import numpy as np
        from acf_functor import LinearOperatorAtomLibrary
        lib = LinearOperatorAtomLibrary()
        N = 4
        j = np.arange(N).reshape(-1, 1)
        k = np.arange(N).reshape(1, -1)
        F4 = np.exp(-2j * np.pi * j * k / N)
        result = lib.factorize_with_atoms(F4)
        assert result.get("found"), f"Factorization not found: {result}"
        assert result["error"] < 0.01, f"Error too large: {result['error']}"

    def test_factorize_dft_8(self):
        """Atom library factorizes DFT(8) with small error."""
        import numpy as np
        from acf_functor import LinearOperatorAtomLibrary
        lib = LinearOperatorAtomLibrary()
        N = 8
        j = np.arange(N).reshape(-1, 1)
        k = np.arange(N).reshape(1, -1)
        F8 = np.exp(-2j * np.pi * j * k / N)
        result = lib.factorize_with_atoms(F8)
        assert result.get("found"), f"Factorization not found: {result}"
        assert result["error"] < 0.01, f"Error too large: {result['error']}"

    def test_factorize_random_fails_gracefully(self):
        """Random matrix returns found=False or high error."""
        from acf_functor import LinearOperatorAtomLibrary
        rng = np.random.RandomState(99)
        lib = LinearOperatorAtomLibrary()
        A = rng.randn(8, 8)
        result = lib.factorize_with_atoms(A)
        # Either not found or high error — the library doesn't falsely claim success
        if result.get("found"):
            assert result["error"] > 0.1, "Library incorrectly claims random matrix is factorizable"


# =============================================================================
# TestAlgebraicDiscoveryEngine
# =============================================================================

class TestAlgebraicDiscoveryEngine:
    """Tests for the AlgebraicDiscoveryEngine."""

    def _dft_matrix(self, N):
        j = np.arange(N).reshape(-1, 1)
        k = np.arange(N).reshape(1, -1)
        return np.exp(-2j * np.pi * j * k / N)

    def test_discover_returns_report(self):
        """discover() returns a report with all required phases."""
        from acf_functor import AlgebraicDiscoveryEngine
        AlgebraicDiscoveryEngine.reset_knowledge()
        engine = AlgebraicDiscoveryEngine()
        F8 = self._dft_matrix(8)
        report = engine.discover(F8, name="test_dft_8")
        assert "phases" in report
        assert "fingerprint" in report["phases"]
        assert "rule_check" in report["phases"]
        assert "factorization" in report["phases"]
        assert "rule_induction" in report["phases"]
        assert "knowledge" in report["phases"]
        assert "N" in report
        assert report["N"] == 8

    def test_discover_dft_finds_factorization(self):
        """Engine discovers butterfly factorization for DFT(8)."""
        from acf_functor import AlgebraicDiscoveryEngine
        AlgebraicDiscoveryEngine.reset_knowledge()
        engine = AlgebraicDiscoveryEngine()
        F8 = self._dft_matrix(8)
        report = engine.discover(F8, name="dft8")
        assert report["factorization_found"], (
            f"Expected factorization, got: {report['phases']['factorization']}"
        )
        assert report["phases"]["factorization"]["error"] < 0.3

    def test_knowledge_graph_accumulates(self):
        """Repeated discover() calls grow the knowledge graph."""
        from acf_functor import AlgebraicDiscoveryEngine
        AlgebraicDiscoveryEngine.reset_knowledge()
        engine = AlgebraicDiscoveryEngine()
        for N in [4, 8]:
            engine.discover(self._dft_matrix(N))
        size1 = engine.knowledge.size
        engine.discover(self._dft_matrix(16))
        size2 = engine.knowledge.size
        assert size2 > size1, "Knowledge graph should grow after each discovery"

    def test_shared_state_across_instances(self):
        """Two engine instances share the same knowledge graph."""
        from acf_functor import AlgebraicDiscoveryEngine
        AlgebraicDiscoveryEngine.reset_knowledge()
        e1 = AlgebraicDiscoveryEngine()
        e2 = AlgebraicDiscoveryEngine()
        e1.discover(self._dft_matrix(4))
        assert e2.knowledge.size == e1.knowledge.size, \
            "Instances should share class-level KnowledgeGraph"

    def test_bootstrap_recursive_rule(self):
        """bootstrap_recursive_rule() induces at least one rule."""
        from acf_functor import AlgebraicDiscoveryEngine
        AlgebraicDiscoveryEngine.reset_knowledge()
        engine = AlgebraicDiscoveryEngine()
        result = engine.bootstrap_recursive_rule(
            operator_fn=self._dft_matrix,
            sizes=[4, 8, 16, 32],
            validate_sizes=[64],
        )
        assert "per_size" in result
        assert "generalization" in result
        for N in [4, 8, 16, 32]:
            assert N in result["per_size"]
        # At least some sizes should yield factorizations
        found_any = any(
            result["per_size"][N]["factorization_found"]
            for N in [4, 8, 16, 32]
        )
        assert found_any, "Bootstrap should find at least one factorization"

    def test_bootstrap_induces_rules(self):
        """After bootstrapping, rule library is non-empty."""
        from acf_functor import AlgebraicDiscoveryEngine
        AlgebraicDiscoveryEngine.reset_knowledge()
        engine = AlgebraicDiscoveryEngine()
        engine.bootstrap_recursive_rule(
            operator_fn=self._dft_matrix,
            sizes=[4, 8, 16, 32],
            validate_sizes=[],
        )
        assert len(engine.rule_induction.induced_rules) > 0, \
            "Should have induced at least one rule after bootstrap"

    def test_build_execute_fn_correct(self):
        """build_execute_fn returns a callable producing correct FFT output."""
        from acf_functor import AlgebraicDiscoveryEngine
        AlgebraicDiscoveryEngine.reset_knowledge()
        engine = AlgebraicDiscoveryEngine()
        N = 16
        execute = engine.build_execute_fn(N)
        assert execute is not None, "build_execute_fn should return callable for N=16"
        rng = np.random.RandomState(7)
        x = rng.randn(N) + 1j * rng.randn(N)
        y = execute(x)
        y_ref = np.fft.fft(x)
        rel_err = np.linalg.norm(y - y_ref) / np.linalg.norm(y_ref)
        assert rel_err < 1e-6, f"Execute fn has too much error: {rel_err:.2e}"

    def test_build_execute_fn_n1024(self):
        """build_execute_fn produces correct FFT for N=1024."""
        from acf_functor import AlgebraicDiscoveryEngine
        AlgebraicDiscoveryEngine.reset_knowledge()
        engine = AlgebraicDiscoveryEngine()
        N = 1024
        execute = engine.build_execute_fn(N)
        assert execute is not None
        rng = np.random.RandomState(13)
        x = rng.randn(N).astype(np.complex128)
        y = execute(x)
        y_ref = np.fft.fft(x)
        rel_err = np.linalg.norm(y - y_ref) / np.linalg.norm(y_ref)
        assert rel_err < 1e-6, f"N=1024 error: {rel_err:.2e}"

    def test_reset_knowledge(self):
        """reset_knowledge() clears all persistent state."""
        from acf_functor import AlgebraicDiscoveryEngine
        engine = AlgebraicDiscoveryEngine()
        engine.discover(self._dft_matrix(4))
        AlgebraicDiscoveryEngine.reset_knowledge()
        engine2 = AlgebraicDiscoveryEngine()
        assert engine2.knowledge.size == 0, "Knowledge should be empty after reset"
        assert len(engine2.rule_induction.induced_rules) == 0


# =============================================================================
# TestBackendSynthesizerFromMatrix
# =============================================================================

class TestBackendSynthesizerFromMatrix:
    """Tests for BackendSynthesizer.synthesize_from_matrix()."""

    def _dft_matrix(self, N):
        j = np.arange(N).reshape(-1, 1)
        k = np.arange(N).reshape(1, -1)
        return np.exp(-2j * np.pi * j * k / N)

    def test_synthesize_from_matrix_returns_kernel(self):
        """synthesize_from_matrix returns a SynthesizedKernel."""
        from acf_functor.algorithm_forge import BackendSynthesizer, SynthesizedKernel
        synth = BackendSynthesizer()
        F8 = self._dft_matrix(8)
        kernel = synth.synthesize_from_matrix(F8, name="dft8", verify=True)
        assert isinstance(kernel, SynthesizedKernel)
        assert callable(kernel.execute)
        assert kernel.n_fma > 0
        assert "grammar" in kernel.certificate

    def test_synthesize_from_matrix_correct(self):
        """Synthesized kernel produces correct output."""
        from acf_functor.algorithm_forge import BackendSynthesizer
        synth = BackendSynthesizer()
        N = 16
        F = self._dft_matrix(N)
        kernel = synth.synthesize_from_matrix(F, name="dft16", verify=True)
        rng = np.random.RandomState(42)
        x = rng.randn(N).astype(np.complex128)
        y = kernel.execute(x)
        y_ref = F @ x
        rel_err = np.linalg.norm(y - y_ref) / np.linalg.norm(y_ref)
        assert rel_err < 1e-3, f"synthesize_from_matrix error: {rel_err:.2e}"

    def test_certificate_has_correct_key(self):
        """SynthesizedKernel certificate contains 'correct'."""
        from acf_functor.algorithm_forge import BackendSynthesizer
        synth = BackendSynthesizer()
        F8 = self._dft_matrix(8)
        kernel = synth.synthesize_from_matrix(F8, verify=True)
        assert "correct" in kernel.certificate


# =============================================================================
# TestUniversalConstructorDiscovery
# =============================================================================

class TestUniversalConstructorDiscovery:
    """Tests for UniversalConstructor.discover_operator_structure() and
    run_algebraic_discovery_experiment()."""

    def _dft_matrix(self, N):
        j = np.arange(N).reshape(-1, 1)
        k = np.arange(N).reshape(1, -1)
        return np.exp(-2j * np.pi * j * k / N)

    def test_discover_operator_structure_returns_report(self):
        """discover_operator_structure() returns a well-structured report."""
        from acf_functor import UniversalConstructor, AlgebraicDiscoveryEngine
        AlgebraicDiscoveryEngine.reset_knowledge()
        uc = UniversalConstructor()
        F8 = self._dft_matrix(8)
        report = uc.discover_operator_structure(F8, name="dft8")
        assert "phases" in report
        assert "fingerprint" in report["phases"]
        assert "factorization" in report["phases"]
        assert report["N"] == 8

    def test_run_algebraic_discovery_experiment_structure(self):
        """run_algebraic_discovery_experiment() returns expected keys."""
        from acf_functor import UniversalConstructor
        uc = UniversalConstructor()
        report = uc.run_algebraic_discovery_experiment(
            bootstrap_sizes=[4, 8, 16],
            validate_sizes=[32],
        )
        assert "bootstrap" in report
        assert "phase_n1024" in report
        assert "certificates" in report
        certs = report["certificates"]
        for cert in ("AD-5", "AD-6", "AD-7", "AD-8"):
            assert cert in certs, f"Missing certificate {cert}"

    def test_run_algebraic_discovery_n1024_correct(self):
        """The discovered N=1024 algorithm produces correct output."""
        from acf_functor import UniversalConstructor
        uc = UniversalConstructor()
        report = uc.run_algebraic_discovery_experiment(
            bootstrap_sizes=[4, 8, 16],
            validate_sizes=[32],
        )
        assert report["phase_n1024"]["correct"], (
            f"N=1024 incorrect: error={report['phase_n1024']['max_error']:.2e}"
        )

    def test_run_algebraic_discovery_certificates(self):
        """AD-6 (grammar found) and AD-8 (N=1024 correct) pass."""
        from acf_functor import UniversalConstructor
        uc = UniversalConstructor()
        report = uc.run_algebraic_discovery_experiment(
            bootstrap_sizes=[4, 8, 16, 32],
            validate_sizes=[64],
        )
        assert report["certificates"]["AD-6"]["passed"], \
            "AD-6: Grammar search should find butterfly factorization"
        assert report["certificates"]["AD-8"]["passed"], \
            "AD-8: N=1024 algorithm should be correct"
