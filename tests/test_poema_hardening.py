import random
from typing import Callable, List, Tuple

import pytest
import torch

from poema import Poem, PoemCompiler


@pytest.fixture
def poem():
    return Poem()


@pytest.fixture
def compiler():
    return PoemCompiler(target="pytorch", precision="fp64", verbose=False)


def _poly_eval(coeffs: List[float], x: torch.Tensor) -> torch.Tensor:
    y = torch.zeros_like(x)
    for c in reversed(coeffs):
        y = y * x + c
    return y


def _interval_from_fn(
    fn: Callable[[torch.Tensor], torch.Tensor],
    domain: Tuple[float, float],
    samples: int = 1025,
) -> Tuple[float, float]:
    x = torch.linspace(domain[0], domain[1], samples, dtype=torch.float64)
    y = fn(x)
    return float(torch.min(y).item()), float(torch.max(y).item())


def _build_random_safe_chain(
    poem: Poem,
    rng: random.Random,
    base_domain: Tuple[float, float],
    depth: int,
):
    coeffs = [
        rng.uniform(-0.08, 0.08),
        rng.uniform(0.25, 0.55),
        rng.uniform(-0.06, 0.06),
        rng.uniform(-0.03, 0.03),
        rng.uniform(-0.01, 0.01),
    ]

    ast = poem.polynomial(coeffs)
    ref_fn = lambda x, _c=coeffs: _poly_eval(_c, x)
    current_interval = _interval_from_fn(ref_fn, base_domain)

    for _ in range(depth):
        options = [
            (poem.sin, torch.sin),
            (poem.cos, torch.cos),
            (poem.tanh, torch.tanh),
        ]
        # Include exp only when the incoming interval is narrow enough to keep it stable.
        if max(abs(current_interval[0]), abs(current_interval[1])) < 1.1:
            options.append((poem.exp, torch.exp))

        builder, torch_fn = rng.choice(options)
        margin = 0.25
        domain = (current_interval[0] - margin, current_interval[1] + margin)
        outer = builder(domain=domain, degree=56)

        prev_ref = ref_fn
        ref_fn = lambda x, _p=prev_ref, _f=torch_fn: _f(_p(x))
        ast = poem.compose(outer, ast)
        current_interval = _interval_from_fn(ref_fn, base_domain)

    return ast, ref_fn


def test_fuzz_semantic_equivalence_deep_compose_scientific(  # 180 deep random compositions
    poem,
    compiler,
):
    rng = random.Random(20260323)
    torch.manual_seed(20260323)

    base_domain = (-1.0, 1.0)
    x = torch.linspace(base_domain[0], base_domain[1], 4097, dtype=torch.float64)

    errors: List[float] = []
    n_cases = 180
    for _ in range(n_cases):
        depth = rng.choice([2, 3])
        ast, ref_fn = _build_random_safe_chain(poem, rng, base_domain, depth)

        exe, report = compiler.compile(ast, domain=base_domain)
        y = exe(x)
        ref = ref_fn(x)

        err = torch.max(torch.abs(y - ref)).item()
        errors.append(err)

        assert torch.isfinite(y).all().item()
        assert report.domain_guard_checks >= depth
        assert report.domain_guard_violations == 0

    q = torch.quantile(torch.tensor(errors, dtype=torch.float64), torch.tensor([0.5, 0.9, 0.99], dtype=torch.float64))
    p50, p90, p99 = [float(v.item()) for v in q]

    print(f"semantic fuzz n={n_cases} p50={p50:.3e} p90={p90:.3e} p99={p99:.3e} max={max(errors):.3e}")

    assert p90 < 2e-5
    assert p99 < 2e-4
    assert max(errors) < 1e-3


def test_domain_guard_overshoot_percentiles_ci_scientific(  # 320 adversarial deep compositions
    poem,
    compiler,
):
    rng = random.Random(91077)
    torch.manual_seed(91077)

    base_domain = (-1.0, 1.0)
    n_cases = 320
    overshoots: List[float] = []

    for _ in range(n_cases):
        coeffs = [
            rng.uniform(-0.25, 0.25),
            rng.uniform(1.8, 2.6),
            rng.uniform(-0.30, 0.30),
            rng.uniform(-0.12, 0.12),
        ]
        inner = poem.polynomial(coeffs)

        # First compose intentionally too narrow to force a guaranteed guard hit.
        deep = poem.compose(poem.sin(domain=(-0.75, 0.75), degree=48), inner)

        # Additional deep level with potentially tight certified domain.
        if rng.random() < 0.5:
            deep = poem.compose(poem.exp(domain=(-1.0, 1.0), degree=48), deep)
        else:
            deep = poem.compose(poem.tanh(domain=(-0.9, 0.9), degree=48), deep)

        _, report = compiler.compile(deep, domain=base_domain)

        overshoots.append(report.domain_guard_max_overshoot)
        assert report.domain_guard_checks >= 2
        assert report.domain_guard_violations >= 1
        assert any("domain guard" in w for w in report.warnings)

    over = torch.tensor(overshoots, dtype=torch.float64)
    q = torch.quantile(over, torch.tensor([0.5, 0.9, 0.99], dtype=torch.float64))
    p50, p90, p99 = [float(v.item()) for v in q]

    print(f"overshoot fuzz n={n_cases} p50={p50:.3e} p90={p90:.3e} p99={p99:.3e} max={float(torch.max(over).item()):.3e}")

    assert float(torch.min(over).item()) > 0.0
    assert 0.0 < p50 <= p90 <= p99
    assert p90 > 0.25
    assert p99 > 0.4


def test_auto_domain_repair_high_complexity_activation(poem):
    torch.manual_seed(20260323)

    # Deliberately narrow certified domains to force out-of-domain compose behavior.
    poly = poem.polynomial([0.3, 1.25, -0.55, 0.22, -0.04])
    exp_narrow = poem.exp(domain=(-1.0, 1.0), degree=110)
    sin_narrow = poem.sin(domain=(-2.0, 2.0), degree=120)

    ast = poem.compose(sin_narrow, poem.compose(exp_narrow, poly))
    x = torch.linspace(-2.5, 2.5, 25001, dtype=torch.float64)
    ref = torch.sin(torch.exp(0.3 + 1.25 * x - 0.55 * x**2 + 0.22 * x**3 - 0.04 * x**4))

    compiler_unrepaired = PoemCompiler(
        target="pytorch",
        precision="fp64",
        verbose=False,
        auto_domain_repair=False,
    )
    compiler_repaired = PoemCompiler(
        target="pytorch",
        precision="fp64",
        verbose=False,
        auto_domain_repair=True,
    )

    exe_bad, _ = compiler_unrepaired.compile(ast, domain=(-2.5, 2.5))
    exe_ok, _ = compiler_repaired.compile(ast, domain=(-2.5, 2.5))

    y_bad = exe_bad(x)
    y_ok = exe_ok(x)

    # Stress condition: non-repaired path destabilizes in this adversarial regime.
    assert not torch.isfinite(y_bad).all().item()

    # Auto-repair must recover a stable and accurate output.
    err_ok = torch.abs(y_ok - ref)
    assert torch.isfinite(y_ok).all().item()
    assert torch.max(err_ok).item() < 1e-7
    assert torch.mean(err_ok).item() < 1e-9
