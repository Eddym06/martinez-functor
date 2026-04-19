"""Mathematical Genesis Engine (Evolution 20)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
import hashlib
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import torch

from .core import HornerReducer
from .persistent_homology import PersistenceBar


class DiscoveryType(Enum):
    """Classification for discovered relations."""

    ALGEBRAIC_IDENTITY = auto()
    FUNCTIONAL_EQUATION = auto()
    SYMMETRY = auto()
    FIXED_POINT = auto()
    DIFFERENTIAL_RELATION = auto()
    APPROXIMATION_BOUND = auto()
    STRUCTURAL_INVARIANT = auto()
    UNKNOWN = auto()


class DiscoveryStrength(Enum):
    """Confidence level for a discovery."""

    CONJECTURE = auto()
    NUMERICAL = auto()
    CERTIFIED = auto()
    FORMAL = auto()


@dataclass
class ProgramGenome:
    """Parametric description of a generated Poema-like program."""

    structure: str
    coefficients: List[float]
    depth: int
    n_operations: int
    seed: int

    @property
    def genome_hash(self) -> str:
        data = f"{self.structure}:{self.coefficients}:{self.depth}:{self.seed}"
        return hashlib.md5(data.encode("utf-8")).hexdigest()[:12]


@dataclass
class TopologicalFingerprint:
    """Compact function signature for similarity filtering."""

    evaluation_digest: torch.Tensor
    derivative_digest: torch.Tensor
    spectral_digest: torch.Tensor
    symmetry_signature: Dict[str, float]
    persistence_bars: List[PersistenceBar]
    total_persistence: float

    def similarity(self, other: "TopologicalFingerprint") -> float:
        if self.evaluation_digest.shape != other.evaluation_digest.shape:
            return 0.0
        diff = torch.norm(self.evaluation_digest - other.evaluation_digest).item()
        norm = (
            torch.norm(self.evaluation_digest).item()
            + torch.norm(other.evaluation_digest).item()
            + 1e-15
        )
        return max(0.0, 1.0 - diff / norm)


@dataclass
class MathematicalDiscovery:
    """Discovered relation plus evidence and metadata."""

    discovery_id: str
    discovery_type: DiscoveryType
    strength: DiscoveryStrength
    programs: List[ProgramGenome]
    description: str
    formal_statement: str
    numerical_evidence: Dict[str, float]
    persistence_score: float
    perturbation_stability: float
    max_numerical_error: float
    domain_tested: Tuple[float, float]
    n_test_points: int
    truth_value: float
    discovery_time: float
    generation: int

    def summary(self) -> str:
        strength_icon = {
            DiscoveryStrength.CONJECTURE: "?",
            DiscoveryStrength.NUMERICAL: "~",
            DiscoveryStrength.CERTIFIED: "v",
            DiscoveryStrength.FORMAL: "|-",
        }
        icon = strength_icon.get(self.strength, "?")
        return (
            f"[{icon}] {self.discovery_type.name}: {self.description}\n"
            f"  Formal: {self.formal_statement}\n"
            f"  Error: {self.max_numerical_error:.2e}, "
            f"Persistence: {self.persistence_score:.4f}, "
            f"Truth: {self.truth_value:.4f}"
        )


@dataclass
class GenesisReport:
    """Full report of one Genesis run."""

    discoveries: List[MathematicalDiscovery]
    programs_generated: int
    programs_survived_filter: int
    total_time_seconds: float
    n_generations: int
    strongest_discovery: Optional[MathematicalDiscovery]
    statistics: Dict[str, Any]

    def summary(self) -> str:
        lines = [
            "+----------------------------------------------+",
            "|          MATHEMATICAL GENESIS REPORT         |",
            "+----------------------------------------------+",
            f"| Programs Generated:    {self.programs_generated:>12}         |",
            f"| Survived Filter:       {self.programs_survived_filter:>12}         |",
            f"| Discoveries:           {len(self.discoveries):>12}         |",
            f"| Generations:           {self.n_generations:>12}         |",
            f"| Total Time (s):        {self.total_time_seconds:>12.2f}         |",
            "+----------------------------------------------+",
        ]
        if self.discoveries:
            lines.append("\nTop discoveries:")
            best = sorted(self.discoveries, key=lambda d: d.persistence_score, reverse=True)[:10]
            for disc in best:
                lines.append(disc.summary())
        return "\n".join(lines)


class ProgramGenerator:
    """Creates random and structured programs over affine/transcendental motifs."""

    def __init__(
        self,
        max_depth: int = 6,
        coefficient_range: Tuple[float, float] = (-5.0, 5.0),
        dtype: torch.dtype = torch.float64,
    ):
        self.max_depth = max_depth
        self.coeff_range = coefficient_range
        self.dtype = dtype

    def generate_random(
        self,
        n_programs: int,
        seed: Optional[int] = None,
    ) -> List[Tuple[ProgramGenome, Callable[[torch.Tensor], torch.Tensor]]]:
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        programs: List[Tuple[ProgramGenome, Callable[[torch.Tensor], torch.Tensor]]] = []
        base_seed = 0 if seed is None else seed
        for i in range(n_programs):
            genome, func = self._generate_one(seed=base_seed + i)
            programs.append((genome, func))
        return programs

    def generate_structured(
        self,
        templates: List[str],
        n_variants: int = 10,
        perturbation: float = 0.1,
        seed: Optional[int] = None,
    ) -> List[Tuple[ProgramGenome, Callable[[torch.Tensor], torch.Tensor]]]:
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        programs: List[Tuple[ProgramGenome, Callable[[torch.Tensor], torch.Tensor]]] = []
        base_seed = 0 if seed is None else seed
        for template in templates:
            for i in range(n_variants):
                genome, func = self._generate_from_template(
                    template,
                    perturbation,
                    seed=base_seed + 101 * i,
                )
                programs.append((genome, func))
        return programs

    def mutate(
        self,
        genome: ProgramGenome,
        mutation_rate: float = 0.1,
        n_mutations: int = 5,
    ) -> List[Tuple[ProgramGenome, Callable[[torch.Tensor], torch.Tensor]]]:
        mutations: List[Tuple[ProgramGenome, Callable[[torch.Tensor], torch.Tensor]]] = []
        for i in range(n_mutations):
            new_coeffs: List[float] = []
            for coeff in genome.coefficients:
                if torch.rand(1).item() < mutation_rate:
                    perturb = torch.randn(1).item() * max(abs(coeff), 0.1) * 0.1
                    new_coeffs.append(coeff + perturb)
                else:
                    new_coeffs.append(coeff)

            new_genome = ProgramGenome(
                structure=genome.structure,
                coefficients=new_coeffs,
                depth=genome.depth,
                n_operations=genome.n_operations,
                seed=genome.seed + 1000 + i,
            )
            mutations.append((new_genome, self._coefficients_to_function(new_genome.structure, new_coeffs)))
        return mutations

    def _generate_one(
        self,
        seed: int,
    ) -> Tuple[ProgramGenome, Callable[[torch.Tensor], torch.Tensor]]:
        np.random.seed(seed)
        structure_type = np.random.choice(
            ["polynomial", "affine_chain", "transcendental_compose", "piecewise", "nested_affine"]
        )

        if structure_type == "polynomial":
            degree = int(np.random.randint(1, 8))
            coeffs = torch.randn(degree + 1, dtype=self.dtype).tolist()

            def poly_func(x: torch.Tensor, _coeffs: List[float] = coeffs) -> torch.Tensor:
                return HornerReducer.execute_horner(torch.tensor(_coeffs, dtype=x.dtype, device=x.device), x)

            return ProgramGenome(
                structure=f"polynomial_deg{degree}",
                coefficients=coeffs,
                depth=1,
                n_operations=degree,
                seed=seed,
            ), poly_func

        if structure_type == "affine_chain":
            n_layers = int(np.random.randint(2, 5))
            scales = torch.randn(n_layers, dtype=self.dtype).tolist()
            shifts = torch.randn(n_layers, dtype=self.dtype).tolist()
            coeffs = scales + shifts

            def chain_func(x: torch.Tensor, _s: List[float] = scales, _b: List[float] = shifts) -> torch.Tensor:
                out = x
                for i in range(len(_s)):
                    scale = torch.tensor(_s[i], dtype=x.dtype, device=x.device)
                    shift = torch.tensor(_b[i], dtype=x.dtype, device=x.device)
                    out = torch.addcmul(shift.expand_as(out), scale.expand_as(out), out)
                return out

            return ProgramGenome(
                structure=f"affine_chain_{n_layers}",
                coefficients=coeffs,
                depth=n_layers,
                n_operations=n_layers,
                seed=seed,
            ), chain_func

        if structure_type == "transcendental_compose":
            a, b, c, d = torch.randn(4, dtype=self.dtype).tolist()

            def trig_func(x: torch.Tensor, _a: float = a, _b: float = b, _c: float = c, _d: float = d) -> torch.Tensor:
                return _a * torch.sin(_b * x + _c) + _d

            return ProgramGenome(
                structure="a*sin(b*x+c)+d",
                coefficients=[a, b, c, d],
                depth=3,
                n_operations=4,
                seed=seed,
            ), trig_func

        if structure_type == "nested_affine":
            n = int(np.random.randint(2, 5))
            a_vals = (torch.randn(n, dtype=self.dtype) * 0.5 + 1.0).tolist()
            b_vals = (torch.randn(n, dtype=self.dtype) * 0.3).tolist()

            def nested_affine_func(x: torch.Tensor, _a: List[float] = a_vals, _b: List[float] = b_vals) -> torch.Tensor:
                out = x
                for i in range(len(_a) - 1, -1, -1):
                    out = _a[i] * out + _b[i]
                return out

            return ProgramGenome(
                structure=f"nested_affine_{n}",
                coefficients=a_vals + b_vals,
                depth=n,
                n_operations=n,
                seed=seed,
            ), nested_affine_func

        a1, b1, a2, b2, threshold = torch.randn(5, dtype=self.dtype).tolist()

        def piecewise_func(
            x: torch.Tensor,
            _a1: float = a1,
            _b1: float = b1,
            _a2: float = a2,
            _b2: float = b2,
            _thr: float = threshold,
        ) -> torch.Tensor:
            return torch.where(x < _thr, _a1 * x + _b1, _a2 * x + _b2)

        return ProgramGenome(
            structure="piecewise_affine",
            coefficients=[a1, b1, a2, b2, threshold],
            depth=1,
            n_operations=2,
            seed=seed,
        ), piecewise_func

    def _generate_from_template(
        self,
        template: str,
        perturbation: float,
        seed: int,
    ) -> Tuple[ProgramGenome, Callable[[torch.Tensor], torch.Tensor]]:
        torch.manual_seed(seed)
        np.random.seed(seed)

        if template == "polynomial":
            degree = int(np.random.randint(2, 6))
            coeffs_t = torch.zeros(degree + 1, dtype=self.dtype)
            coeffs_t[0] = 1.0
            coeffs_t[1] = 1.0
            coeffs_t += torch.randn_like(coeffs_t) * perturbation
            coeffs = coeffs_t.tolist()

            def template_poly_func(x: torch.Tensor, _coeffs: List[float] = coeffs) -> torch.Tensor:
                return HornerReducer.execute_horner(torch.tensor(_coeffs, dtype=x.dtype, device=x.device), x)

            return ProgramGenome(
                structure=f"template_poly_{degree}",
                coefficients=coeffs,
                depth=1,
                n_operations=degree,
                seed=seed,
            ), template_poly_func

        if template == "trigonometric":
            a = 1.0 + torch.randn(1).item() * perturbation
            b = 1.0 + torch.randn(1).item() * perturbation
            c = torch.randn(1).item() * perturbation

            def template_trig_func(x: torch.Tensor, _a: float = a, _b: float = b, _c: float = c) -> torch.Tensor:
                return _a * torch.sin(_b * x) + _c * torch.cos(_b * x)

            return ProgramGenome(
                structure="a*sin(b*x)+c*cos(b*x)",
                coefficients=[a, b, c],
                depth=2,
                n_operations=5,
                seed=seed,
            ), template_trig_func

        if template == "exponential":
            a = 1.0 + torch.randn(1).item() * perturbation
            b = 1.0 + torch.randn(1).item() * perturbation

            def template_exp_func(x: torch.Tensor, _a: float = a, _b: float = b) -> torch.Tensor:
                return _a * torch.exp(_b * x)

            return ProgramGenome(
                structure="a*exp(b*x)",
                coefficients=[a, b],
                depth=2,
                n_operations=3,
                seed=seed,
            ), template_exp_func

        return self._generate_one(seed)

    def _coefficients_to_function(
        self,
        structure: str,
        coefficients: List[float],
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        if "polynomial" in structure or "poly" in structure:
            return lambda x, _coeffs=coefficients: HornerReducer.execute_horner(
                torch.tensor(_coeffs, dtype=x.dtype, device=x.device),
                x,
            )
        if "sin" in structure:
            a, b, c = (coefficients + [0.0, 0.0, 0.0])[:3]
            return lambda x, _a=a, _b=b, _c=c: _a * torch.sin(_b * x + _c)
        return lambda x, _coeffs=coefficients: HornerReducer.execute_horner(
            torch.tensor(_coeffs if _coeffs else [0.0], dtype=x.dtype, device=x.device),
            x,
        )


class FingerprintEngine:
    """Computes topological signatures for generated functions."""

    def __init__(
        self,
        n_eval_points: int = 500,
        domain: Tuple[float, float] = (-5.0, 5.0),
        dtype: torch.dtype = torch.float64,
    ):
        self.n_eval_points = n_eval_points
        self.domain = domain
        self.dtype = dtype
        self.x_probe = torch.linspace(domain[0], domain[1], n_eval_points, dtype=dtype)

    def fingerprint(
        self,
        func: Callable[[torch.Tensor], torch.Tensor],
    ) -> Optional[TopologicalFingerprint]:
        try:
            with torch.no_grad():
                y = func(self.x_probe)

            if y.shape != self.x_probe.shape:
                return None
            if torch.isnan(y).any() or torch.isinf(y).any():
                return None
            if torch.max(torch.abs(y)).item() > 1e10:
                return None

            dx = self.x_probe[1] - self.x_probe[0]
            dy = torch.zeros_like(y)
            dy[1:-1] = (y[2:] - y[:-2]) / (2 * dx)
            dy[0] = (y[1] - y[0]) / dx
            dy[-1] = (y[-1] - y[-2]) / dx

            centered = y - torch.mean(y)
            if centered.abs().max().item() > 1e-15:
                fft_vals = torch.fft.rfft(centered)
                spectral = torch.abs(fft_vals[: min(50, len(fft_vals))])
            else:
                spectral = torch.zeros(50, dtype=self.dtype)

            symmetry = self._compute_symmetry(y)
            bars = self._compute_persistence_bars(y)
            total_persistence = float(sum(bar.persistence for bar in bars))

            return TopologicalFingerprint(
                evaluation_digest=y,
                derivative_digest=dy,
                spectral_digest=spectral,
                symmetry_signature=symmetry,
                persistence_bars=bars,
                total_persistence=total_persistence,
            )
        except Exception:
            return None

    def _compute_symmetry(self, y: torch.Tensor) -> Dict[str, float]:
        y_rev = y.flip(0)
        denom = torch.norm(y).item() + 1e-15
        even_err = torch.norm(y - y_rev).item() / denom
        odd_err = torch.norm(y + y_rev).item() / denom
        diffs = y[1:] - y[:-1]
        increasing = (diffs > 0).float().mean().item()
        return {
            "even": max(0.0, 1.0 - even_err),
            "odd": max(0.0, 1.0 - odd_err),
            "monotone_increasing": increasing,
            "monotone_decreasing": 1.0 - increasing,
        }

    def _compute_persistence_bars(self, y: torch.Tensor) -> List[PersistenceBar]:
        bars: List[PersistenceBar] = []
        if len(y) < 3:
            return bars

        size = len(y)
        for i in range(1, size - 1):
            prev_val = float(y[i - 1].item())
            cur_val = float(y[i].item())
            next_val = float(y[i + 1].item())
            if cur_val > prev_val and cur_val > next_val:
                death = min(size - 1, i + abs(int(cur_val * 10)) % max(size - i, 1))
                bars.append(
                    PersistenceBar(
                        birth_scale=i,
                        death_scale=death,
                        persistence=abs(cur_val - prev_val),
                        feature_type="maximum",
                        amplitude=cur_val,
                    )
                )
            elif cur_val < prev_val and cur_val < next_val:
                death = min(size - 1, i + abs(int(cur_val * 10)) % max(size - i, 1))
                bars.append(
                    PersistenceBar(
                        birth_scale=i,
                        death_scale=death,
                        persistence=abs(cur_val - prev_val),
                        feature_type="minimum",
                        amplitude=cur_val,
                    )
                )
        return bars


class RelationDetector:
    """Detects identities, symmetries and simple differential laws."""

    def __init__(
        self,
        identity_threshold: float = 1e-8,
        n_test: int = 2000,
        domain: Tuple[float, float] = (-3.0, 3.0),
        dtype: torch.dtype = torch.float64,
    ):
        self.threshold = identity_threshold
        self.n_test = n_test
        self.domain = domain
        self.dtype = dtype

    def detect_identity(
        self,
        func_a: Callable[[torch.Tensor], torch.Tensor],
        func_b: Callable[[torch.Tensor], torch.Tensor],
        genome_a: ProgramGenome,
        genome_b: ProgramGenome,
    ) -> Optional[MathematicalDiscovery]:
        x = torch.linspace(self.domain[0], self.domain[1], self.n_test, dtype=self.dtype)
        try:
            ya = func_a(x)
            yb = func_b(x)
            if torch.isnan(ya).any() or torch.isnan(yb).any():
                return None

            diff = torch.abs(ya - yb)
            max_err = float(diff.max().item())
            mean_err = float(diff.mean().item())
            if max_err >= self.threshold:
                return None

            stability = self._perturbation_stability(func_a, func_b, x)
            score = max(0.0, 1.0 - max_err / max(self.threshold, 1e-30))
            return MathematicalDiscovery(
                discovery_id=f"identity_{genome_a.genome_hash}_{genome_b.genome_hash}",
                discovery_type=DiscoveryType.ALGEBRAIC_IDENTITY,
                strength=DiscoveryStrength.NUMERICAL,
                programs=[genome_a, genome_b],
                description=f"{genome_a.structure} == {genome_b.structure}",
                formal_statement=(
                    f"forall x in [{self.domain[0]}, {self.domain[1]}], "
                    f"|f(x)-g(x)| < {max_err:.2e}"
                ),
                numerical_evidence={"max_error": max_err, "mean_error": mean_err},
                persistence_score=score,
                perturbation_stability=stability,
                max_numerical_error=max_err,
                domain_tested=self.domain,
                n_test_points=self.n_test,
                truth_value=score,
                discovery_time=0.0,
                generation=0,
            )
        except Exception:
            return None

    def detect_functional_equation(
        self,
        func_f: Callable[[torch.Tensor], torch.Tensor],
        func_g: Callable[[torch.Tensor], torch.Tensor],
        func_h: Callable[[torch.Tensor], torch.Tensor],
        genomes: List[ProgramGenome],
    ) -> Optional[MathematicalDiscovery]:
        x = torch.linspace(self.domain[0], self.domain[1], self.n_test, dtype=self.dtype)
        try:
            lhs = func_f(func_g(x))
            rhs = func_h(x)
            if torch.isnan(lhs).any() or torch.isnan(rhs).any():
                return None
            max_err = float(torch.max(torch.abs(lhs - rhs)).item())
            if max_err >= self.threshold:
                return None
            score = max(0.0, 1.0 - max_err / max(self.threshold, 1e-30))
            return MathematicalDiscovery(
                discovery_id=f"func_eq_{genomes[0].genome_hash}_{genomes[-1].genome_hash}",
                discovery_type=DiscoveryType.FUNCTIONAL_EQUATION,
                strength=DiscoveryStrength.NUMERICAL,
                programs=genomes,
                description="f(g(x)) == h(x)",
                formal_statement=f"forall x, |f(g(x))-h(x)| < {max_err:.2e}",
                numerical_evidence={"max_error": max_err},
                persistence_score=score,
                perturbation_stability=0.0,
                max_numerical_error=max_err,
                domain_tested=self.domain,
                n_test_points=self.n_test,
                truth_value=score,
                discovery_time=0.0,
                generation=0,
            )
        except Exception:
            return None

    def detect_symmetry(
        self,
        func: Callable[[torch.Tensor], torch.Tensor],
        genome: ProgramGenome,
    ) -> Optional[MathematicalDiscovery]:
        x = torch.linspace(0.01, self.domain[1], max(8, self.n_test // 2), dtype=self.dtype)
        try:
            y_pos = func(x)
            y_neg = func(-x)
            if torch.isnan(y_pos).any() or torch.isnan(y_neg).any():
                return None

            even_err = float(torch.max(torch.abs(y_pos - y_neg)).item())
            if even_err < self.threshold:
                score = max(0.0, 1.0 - even_err / max(self.threshold, 1e-30))
                return MathematicalDiscovery(
                    discovery_id=f"even_{genome.genome_hash}",
                    discovery_type=DiscoveryType.SYMMETRY,
                    strength=DiscoveryStrength.NUMERICAL,
                    programs=[genome],
                    description=f"{genome.structure} is even",
                    formal_statement="forall x, f(-x) = f(x)",
                    numerical_evidence={"even_error": even_err},
                    persistence_score=score,
                    perturbation_stability=0.0,
                    max_numerical_error=even_err,
                    domain_tested=self.domain,
                    n_test_points=self.n_test,
                    truth_value=score,
                    discovery_time=0.0,
                    generation=0,
                )

            odd_err = float(torch.max(torch.abs(y_pos + y_neg)).item())
            if odd_err < self.threshold:
                score = max(0.0, 1.0 - odd_err / max(self.threshold, 1e-30))
                return MathematicalDiscovery(
                    discovery_id=f"odd_{genome.genome_hash}",
                    discovery_type=DiscoveryType.SYMMETRY,
                    strength=DiscoveryStrength.NUMERICAL,
                    programs=[genome],
                    description=f"{genome.structure} is odd",
                    formal_statement="forall x, f(-x) = -f(x)",
                    numerical_evidence={"odd_error": odd_err},
                    persistence_score=score,
                    perturbation_stability=0.0,
                    max_numerical_error=odd_err,
                    domain_tested=self.domain,
                    n_test_points=self.n_test,
                    truth_value=score,
                    discovery_time=0.0,
                    generation=0,
                )
        except Exception:
            return None

        return None

    def detect_differential_relation(
        self,
        func: Callable[[torch.Tensor], torch.Tensor],
        genome: ProgramGenome,
    ) -> Optional[MathematicalDiscovery]:
        x = torch.linspace(
            self.domain[0] + 0.01,
            self.domain[1] - 0.01,
            self.n_test,
            dtype=self.dtype,
        )
        try:
            y = func(x)
            dx = x[1] - x[0]
            dy = torch.zeros_like(y)
            dy[1:-1] = (y[2:] - y[:-2]) / (2 * dx)
            dy[0] = (y[1] - y[0]) / dx
            dy[-1] = (y[-1] - y[-2]) / dx
            if torch.isnan(dy).any():
                return None

            norm = float(torch.norm(y).item()) + 1e-15
            exp_err = float(torch.max(torch.abs(dy - y)).item()) / norm
            if exp_err < self.threshold * 100:
                score = max(0.0, 1.0 - exp_err)
                return MathematicalDiscovery(
                    discovery_id=f"diff_self_{genome.genome_hash}",
                    discovery_type=DiscoveryType.DIFFERENTIAL_RELATION,
                    strength=DiscoveryStrength.NUMERICAL,
                    programs=[genome],
                    description="f' = f",
                    formal_statement="df/dx = f",
                    numerical_evidence={"relative_error": exp_err},
                    persistence_score=score,
                    perturbation_stability=0.0,
                    max_numerical_error=exp_err,
                    domain_tested=self.domain,
                    n_test_points=self.n_test,
                    truth_value=score,
                    discovery_time=0.0,
                    generation=0,
                )

            neg_err = float(torch.max(torch.abs(dy + y)).item()) / norm
            if neg_err < self.threshold * 100:
                score = max(0.0, 1.0 - neg_err)
                return MathematicalDiscovery(
                    discovery_id=f"diff_neg_{genome.genome_hash}",
                    discovery_type=DiscoveryType.DIFFERENTIAL_RELATION,
                    strength=DiscoveryStrength.NUMERICAL,
                    programs=[genome],
                    description="f' = -f",
                    formal_statement="df/dx = -f",
                    numerical_evidence={"relative_error": neg_err},
                    persistence_score=score,
                    perturbation_stability=0.0,
                    max_numerical_error=neg_err,
                    domain_tested=self.domain,
                    n_test_points=self.n_test,
                    truth_value=score,
                    discovery_time=0.0,
                    generation=0,
                )
        except Exception:
            return None

        return None

    def _perturbation_stability(
        self,
        func_a: Callable[[torch.Tensor], torch.Tensor],
        func_b: Callable[[torch.Tensor], torch.Tensor],
        x: torch.Tensor,
        n_perturbations: int = 10,
        perturbation_size: float = 0.01,
    ) -> float:
        errors: List[float] = []
        for _ in range(n_perturbations):
            x_pert = x + torch.randn_like(x) * perturbation_size
            try:
                err = float(torch.max(torch.abs(func_a(x_pert) - func_b(x_pert))).item())
                errors.append(err)
            except Exception:
                errors.append(float("inf"))

        finite = [e for e in errors if np.isfinite(e)]
        if not finite:
            return 0.0
        return max(0.0, 1.0 - max(finite) / max(self.threshold * 10, 1e-30))


class GenesisOrchestrator:
    """Coordinates the full Evolution 20 search pipeline."""

    def __init__(
        self,
        domain: Tuple[float, float] = (-3.0, 3.0),
        identity_threshold: float = 1e-8,
        persistence_threshold: float = 0.5,
        max_programs_per_generation: int = 200,
        dtype: torch.dtype = torch.float64,
    ):
        self.domain = domain
        self.identity_threshold = identity_threshold
        self.persistence_threshold = persistence_threshold
        self.max_programs = max_programs_per_generation
        self.dtype = dtype

        self.generator = ProgramGenerator(dtype=dtype)
        self.fingerprinter = FingerprintEngine(domain=domain, dtype=dtype)
        self.detector = RelationDetector(
            identity_threshold=identity_threshold,
            domain=domain,
            dtype=dtype,
        )

        self._all_discoveries: List[MathematicalDiscovery] = []
        self._discovery_hashes: Set[str] = set()

    def run(
        self,
        n_generations: int = 5,
        programs_per_generation: int = 100,
        seed: Optional[int] = None,
        templates: Optional[List[str]] = None,
        verbose: bool = True,
    ) -> GenesisReport:
        start = time.time()
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Reentrant runs should start from a clean session state.
        self._all_discoveries = []
        self._discovery_hashes = set()

        total_generated = 0
        total_survived = 0
        for generation in range(n_generations):
            if verbose:
                print(f"[Genesis] Generation {generation + 1}/{n_generations}")

            programs = self._generate_population(
                n=programs_per_generation,
                generation=generation,
                seed=seed,
                templates=templates,
            )
            total_generated += len(programs)

            valid = self._fingerprint_and_filter(programs)
            total_survived += len(valid)

            discoveries = self._detect_relations(valid, generation=generation)
            if verbose:
                print(f"  generated={len(programs)} valid={len(valid)} discoveries={len(discoveries)}")

        for discovery in self._all_discoveries:
            truth = discovery.persistence_score * (
                1.0 - discovery.max_numerical_error / max(self.identity_threshold, 1e-30)
            )
            discovery.truth_value = max(0.0, min(1.0, truth))

        strongest = None
        if self._all_discoveries:
            strongest = max(self._all_discoveries, key=lambda item: item.persistence_score)

        elapsed = time.time() - start
        return GenesisReport(
            discoveries=self._all_discoveries.copy(),
            programs_generated=total_generated,
            programs_survived_filter=total_survived,
            total_time_seconds=elapsed,
            n_generations=n_generations,
            strongest_discovery=strongest,
            statistics={
                "discovery_types": self._count_types(),
                "mean_persistence": self._mean_persistence(),
                "mean_truth": self._mean_truth(),
            },
        )

    def _generate_population(
        self,
        n: int,
        generation: int,
        seed: Optional[int],
        templates: Optional[List[str]],
    ) -> List[Tuple[ProgramGenome, Callable[[torch.Tensor], torch.Tensor]]]:
        base_seed = (42 if seed is None else seed) + generation * 10_000
        n = min(n, self.max_programs)

        n_random = max(1, n // 2)
        random_programs = self.generator.generate_random(n_random, seed=base_seed)

        structured_templates = templates or ["polynomial", "trigonometric", "exponential"]
        remaining = max(0, n - len(random_programs))
        per_template = max(1, remaining // max(len(structured_templates), 1)) if remaining > 0 else 0
        structured_programs = []
        if per_template > 0:
            structured_programs = self.generator.generate_structured(
                structured_templates,
                n_variants=per_template,
                seed=base_seed + 5_000,
            )

        population = (random_programs + structured_programs)[:n]
        if len(population) < n:
            population.extend(self.generator.generate_random(n - len(population), seed=base_seed + 9_999))
        return population[:n]

    def _fingerprint_and_filter(
        self,
        programs: List[Tuple[ProgramGenome, Callable[[torch.Tensor], torch.Tensor]]],
    ) -> List[Tuple[ProgramGenome, Callable[[torch.Tensor], torch.Tensor], TopologicalFingerprint]]:
        valid: List[Tuple[ProgramGenome, Callable[[torch.Tensor], torch.Tensor], TopologicalFingerprint]] = []
        for genome, func in programs:
            fp = self.fingerprinter.fingerprint(func)
            if fp is not None and fp.total_persistence >= self.persistence_threshold:
                valid.append((genome, func, fp))
            elif fp is not None and self.persistence_threshold <= 0:
                valid.append((genome, func, fp))
        return valid

    def _detect_relations(
        self,
        programs: List[Tuple[ProgramGenome, Callable[[torch.Tensor], torch.Tensor], TopologicalFingerprint]],
        generation: int,
    ) -> List[MathematicalDiscovery]:
        new_discoveries: List[MathematicalDiscovery] = []

        for i, (genome_a, func_a, fp_a) in enumerate(programs):
            symmetric = self.detector.detect_symmetry(func_a, genome_a)
            if symmetric is not None:
                symmetric.generation = generation
                if self._is_new(symmetric):
                    self._register_discovery(symmetric, new_discoveries)

            differential = self.detector.detect_differential_relation(func_a, genome_a)
            if differential is not None:
                differential.generation = generation
                if self._is_new(differential):
                    self._register_discovery(differential, new_discoveries)

            for j in range(i + 1, min(len(programs), i + 50)):
                genome_b, func_b, fp_b = programs[j]
                if fp_a.similarity(fp_b) < 0.8:
                    continue

                identity = self.detector.detect_identity(func_a, func_b, genome_a, genome_b)
                if identity is not None:
                    identity.generation = generation
                    if self._is_new(identity):
                        self._register_discovery(identity, new_discoveries)

        return new_discoveries

    def _register_discovery(
        self,
        discovery: MathematicalDiscovery,
        bucket: List[MathematicalDiscovery],
    ) -> None:
        bucket.append(discovery)
        self._all_discoveries.append(discovery)
        self._discovery_hashes.add(discovery.discovery_id)

    def _is_new(self, discovery: MathematicalDiscovery) -> bool:
        return discovery.discovery_id not in self._discovery_hashes

    def _count_types(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for discovery in self._all_discoveries:
            key = discovery.discovery_type.name
            counts[key] = counts.get(key, 0) + 1
        return counts

    def _mean_persistence(self) -> float:
        if not self._all_discoveries:
            return 0.0
        return float(sum(d.persistence_score for d in self._all_discoveries) / len(self._all_discoveries))

    def _mean_truth(self) -> float:
        if not self._all_discoveries:
            return 0.0
        return float(sum(d.truth_value for d in self._all_discoveries) / len(self._all_discoveries))
