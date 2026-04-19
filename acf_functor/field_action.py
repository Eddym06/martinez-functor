"""Field theory action minimization for reductions (Evolution 14)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

import torch

from .core import ChebyshevReducer, HornerReducer, ReductionResult


@dataclass
class ActionComponents:
    kinetic: float
    potential: float
    regularization: float
    total: float
    lagrange_multiplier: float


@dataclass
class EulerLagrangeStep:
    degree: int
    epsilon: float
    action: float
    gradient: float
    is_stationary: bool


@dataclass
class PhaseTransition:
    critical_degree: int
    action_before: float
    action_after: float
    transition_type: str
    order_parameter: float


@dataclass
class FieldTheoryResult:
    optimal_degree: int
    optimal_epsilon: float
    optimal_action: float
    optimal_reduction: ReductionResult
    action_trajectory: List[EulerLagrangeStep]
    phase_transitions: List[PhaseTransition]
    free_energy_profile: Dict[str, List[float]]
    convergence_achieved: bool


class FieldAction:
    def __init__(
        self,
        degree_range: Tuple[int, int] = (2, 100),
        lambda_reg: float = 0.001,
        beta: float = 10.0,
        gradient_threshold: float = 1e-4,
        max_flow_steps: int = 50,
        n_probe: int = 5000,
        dtype: torch.dtype = torch.float64,
    ):
        self.degree_range = degree_range
        self.lambda_reg = lambda_reg
        self.beta = beta
        self.gradient_threshold = gradient_threshold
        self.max_flow_steps = max_flow_steps
        self.n_probe = n_probe
        self.dtype = dtype

    def minimize_action(
        self,
        f: Callable[[torch.Tensor], torch.Tensor],
        domain: Tuple[float, float],
    ) -> FieldTheoryResult:
        x = torch.linspace(domain[0], domain[1], self.n_probe, dtype=self.dtype)
        f_exact = f(x)

        landscape = self._compute_action_landscape(f, domain, x, f_exact)
        if not landscape:
            raise RuntimeError("Could not compute action landscape.")

        trajectory, optimal_step = self._gradient_flow(landscape)
        transitions = self._detect_phase_transitions(landscape)
        free_energy = self._compute_free_energy(landscape)

        optimal_degree = optimal_step.degree
        optimal_reduction = ChebyshevReducer.reduce(f, degree=optimal_degree, domain=domain, dtype=self.dtype)

        return FieldTheoryResult(
            optimal_degree=optimal_degree,
            optimal_epsilon=optimal_step.epsilon,
            optimal_action=optimal_step.action,
            optimal_reduction=optimal_reduction,
            action_trajectory=trajectory,
            phase_transitions=transitions,
            free_energy_profile=free_energy,
            convergence_achieved=optimal_step.is_stationary,
        )

    def compute_action(
        self,
        f: Callable[[torch.Tensor], torch.Tensor],
        domain: Tuple[float, float],
        degree: int,
    ) -> ActionComponents:
        x = torch.linspace(domain[0], domain[1], self.n_probe, dtype=self.dtype)
        f_exact = f(x)

        try:
            reduction = ChebyshevReducer.reduce(f, degree=degree, domain=domain, dtype=self.dtype)
            coeffs = reduction.metadata.get(
                "monomial_coefficients",
                reduction.metadata.get("coefficients", []),
            )
            y = HornerReducer.execute_horner(torch.as_tensor(coeffs, dtype=self.dtype), x)
            potential = float(torch.max(torch.abs(f_exact - y)).item())
        except Exception:
            potential = float("inf")

        kinetic = self._compute_kinetic(f, domain, degree, x)
        regularization = float(degree)
        total = kinetic + potential + self.lambda_reg * regularization

        return ActionComponents(
            kinetic=kinetic,
            potential=potential,
            regularization=regularization,
            total=total,
            lagrange_multiplier=self.lambda_reg,
        )

    def _compute_action_landscape(
        self,
        f: Callable[[torch.Tensor], torch.Tensor],
        domain: Tuple[float, float],
        x: torch.Tensor,
        f_exact: torch.Tensor,
    ) -> List[Dict[str, Any]]:
        landscape: List[Dict[str, Any]] = []
        prev_y = None

        for degree in range(self.degree_range[0], self.degree_range[1] + 1):
            try:
                reduction = ChebyshevReducer.reduce(f, degree=degree, domain=domain, dtype=self.dtype)
                coeffs = reduction.metadata.get(
                    "monomial_coefficients",
                    reduction.metadata.get("coefficients", []),
                )
                y = HornerReducer.execute_horner(torch.as_tensor(coeffs, dtype=self.dtype), x)
                potential = float(torch.max(torch.abs(f_exact - y)).item())
                if prev_y is not None:
                    kinetic = float(torch.max(torch.abs(y - prev_y)).item())
                else:
                    kinetic = 0.0
                regularization = float(degree)
                action = kinetic + potential + self.lambda_reg * regularization
                landscape.append(
                    {
                        "degree": degree,
                        "potential": potential,
                        "kinetic": kinetic,
                        "regularization": regularization,
                        "action": action,
                        "reduction": reduction,
                    }
                )
                prev_y = y
            except Exception:
                continue

        return landscape

    def _gradient_flow(self, landscape: List[Dict[str, Any]]) -> Tuple[List[EulerLagrangeStep], EulerLagrangeStep]:
        trajectory: List[EulerLagrangeStep] = []
        for i, entry in enumerate(landscape):
            if i > 0 and i < len(landscape) - 1:
                gradient = (landscape[i + 1]["action"] - landscape[i - 1]["action"]) / 2.0
            elif i > 0:
                gradient = entry["action"] - landscape[i - 1]["action"]
            else:
                gradient = 0.0

            trajectory.append(
                EulerLagrangeStep(
                    degree=int(entry["degree"]),
                    epsilon=float(entry["potential"]),
                    action=float(entry["action"]),
                    gradient=float(gradient),
                    is_stationary=(abs(float(gradient)) < self.gradient_threshold),
                )
            )

        optimal = min(trajectory, key=lambda step: step.action)
        return trajectory, optimal

    def _detect_phase_transitions(self, landscape: List[Dict[str, Any]]) -> List[PhaseTransition]:
        transitions: List[PhaseTransition] = []
        if len(landscape) < 3:
            return transitions

        actions = [float(entry["action"]) for entry in landscape]
        for i in range(1, len(actions) - 1):
            second = actions[i + 1] - 2.0 * actions[i] + actions[i - 1]
            if i >= 2:
                prev_second = actions[i] - 2.0 * actions[i - 1] + actions[i - 2]
                if prev_second * second < 0 and abs(second) > 0.01:
                    transitions.append(
                        PhaseTransition(
                            critical_degree=int(landscape[i]["degree"]),
                            action_before=actions[i - 1],
                            action_after=actions[i + 1],
                            transition_type=("crystallization" if second > 0 else "melting"),
                            order_parameter=abs(second),
                        )
                    )
        return transitions

    def _compute_free_energy(self, landscape: List[Dict[str, Any]]) -> Dict[str, List[float]]:
        degrees: List[float] = []
        energies: List[float] = []
        entropies: List[float] = []
        free_energies: List[float] = []

        for entry in landscape:
            d = float(entry["degree"])
            energy = float(entry["potential"])
            entropy = float(torch.log(torch.tensor(d + 1.0)).item())
            free_energy = energy - (1.0 / self.beta) * entropy

            degrees.append(d)
            energies.append(energy)
            entropies.append(entropy)
            free_energies.append(free_energy)

        return {
            "degrees": degrees,
            "energy": energies,
            "entropy": entropies,
            "free_energy": free_energies,
        }

    def _compute_kinetic(
        self,
        f: Callable[[torch.Tensor], torch.Tensor],
        domain: Tuple[float, float],
        degree: int,
        x: torch.Tensor,
    ) -> float:
        try:
            r1 = ChebyshevReducer.reduce(f, degree=max(degree - 1, 2), domain=domain, dtype=self.dtype)
            r2 = ChebyshevReducer.reduce(f, degree=degree, domain=domain, dtype=self.dtype)
            c1 = r1.metadata.get("monomial_coefficients", r1.metadata.get("coefficients", []))
            c2 = r2.metadata.get("monomial_coefficients", r2.metadata.get("coefficients", []))
            y1 = HornerReducer.execute_horner(torch.as_tensor(c1, dtype=self.dtype), x)
            y2 = HornerReducer.execute_horner(torch.as_tensor(c2, dtype=self.dtype), x)
            return float(torch.max(torch.abs(y2 - y1)).item())
        except Exception:
            return 0.0
