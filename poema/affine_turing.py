"""
Affine Turing Machine (Evolution 18).

This module provides a constructive computational model over affine
operations and a bridge back to Poema AST nodes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import torch

from .ast_nodes import (
    ASTNode,
    AffineNode,
    ComposeNode,
    ConstantNode,
    IdentityNode,
    InputNode,
    PolynomialNode,
    ScaleNode,
    ShiftNode,
    StratifiedNode,
)


@dataclass
class MTAState:
    name: str
    index: int
    is_initial: bool = False
    is_accepting: bool = False


@dataclass
class MTATransition:
    source: MTAState
    target: MTAState
    condition: Callable[[torch.Tensor], torch.Tensor]
    affine_action: ASTNode
    description: str = ""


@dataclass
class MTAProgram:
    states: List[MTAState]
    transitions: List[MTATransition]
    initial_state: MTAState
    accepting_states: List[MTAState]
    tape_dimension: int = 1

    @property
    def n_states(self) -> int:
        return len(self.states)

    @property
    def n_transitions(self) -> int:
        return len(self.transitions)


@dataclass
class MTAExecution:
    steps: List[Tuple[MTAState, torch.Tensor]]
    final_state: MTAState
    final_tape: torch.Tensor
    accepted: bool
    n_steps: int


class AffineTuringMachine:
    def __init__(self, dtype: torch.dtype = torch.float64):
        self.dtype = dtype

    def execute(
        self,
        program: MTAProgram,
        initial_tape: torch.Tensor,
        max_steps: int = 10000,
    ) -> MTAExecution:
        state = program.initial_state
        tape = initial_tape.clone().to(self.dtype)
        steps = [(state, tape.clone())]

        for _ in range(max_steps):
            matched = False
            for trans in program.transitions:
                if trans.source.index != state.index:
                    continue
                cond = trans.condition(tape)
                if isinstance(cond, torch.Tensor):
                    ok = bool(cond.any().item())
                else:
                    ok = bool(cond)
                if not ok:
                    continue

                tape = self._apply_action(trans.affine_action, tape)
                state = trans.target
                steps.append((state, tape.clone()))
                matched = True
                break

            if not matched:
                break
            if state in program.accepting_states:
                break

        return MTAExecution(
            steps=steps,
            final_state=state,
            final_tape=tape,
            accepted=(state in program.accepting_states),
            n_steps=max(0, len(steps) - 1),
        )

    def _apply_action(self, action: ASTNode, tape: torch.Tensor) -> torch.Tensor:
        if isinstance(action, AffineNode):
            return action.scale_factor.to(self.dtype) * tape + action.shift_value.to(self.dtype)
        if isinstance(action, ScaleNode):
            return action.factor.to(self.dtype) * tape
        if isinstance(action, ShiftNode):
            return tape + action.value.to(self.dtype)
        if isinstance(action, IdentityNode):
            return tape
        if isinstance(action, ConstantNode):
            return action.value.to(self.dtype).expand_as(tape)
        return tape

    @staticmethod
    def build_counter(
        initial_value: float = 0.0,
        increment: float = 1.0,
        limit: float = 10.0,
    ) -> MTAProgram:
        _ = initial_value
        q_count = MTAState("count", 0, is_initial=True)
        q_done = MTAState("done", 1, is_accepting=True)
        return MTAProgram(
            states=[q_count, q_done],
            transitions=[
                MTATransition(
                    source=q_count,
                    target=q_count,
                    condition=lambda x, _l=limit: x < _l,
                    affine_action=ShiftNode(value=torch.tensor(increment, dtype=torch.float64)),
                    description="increment",
                ),
                MTATransition(
                    source=q_count,
                    target=q_done,
                    condition=lambda x, _l=limit: x >= _l,
                    affine_action=IdentityNode(),
                    description="halt",
                ),
            ],
            initial_state=q_count,
            accepting_states=[q_done],
        )

    @staticmethod
    def build_multiplier(factor: float, n_times: int) -> MTAProgram:
        states = [MTAState("mul_0", 0, is_initial=True)]
        for i in range(1, n_times + 1):
            states.append(MTAState("mul_{0}".format(i), i, is_accepting=(i == n_times)))

        transitions: List[MTATransition] = []
        for i in range(n_times):
            transitions.append(
                MTATransition(
                    source=states[i],
                    target=states[i + 1],
                    condition=lambda x: torch.ones_like(x, dtype=torch.bool),
                    affine_action=ScaleNode(factor=torch.tensor(factor, dtype=torch.float64)),
                    description="multiply",
                )
            )

        return MTAProgram(
            states=states,
            transitions=transitions,
            initial_state=states[0],
            accepting_states=[states[-1]],
        )

    @staticmethod
    def build_conditional(
        threshold: float,
        action_above: ASTNode,
        action_below: ASTNode,
    ) -> MTAProgram:
        q_start = MTAState("start", 0, is_initial=True)
        q_above = MTAState("above", 1, is_accepting=True)
        q_below = MTAState("below", 2, is_accepting=True)
        return MTAProgram(
            states=[q_start, q_above, q_below],
            transitions=[
                MTATransition(
                    source=q_start,
                    target=q_above,
                    condition=lambda x, _t=threshold: x >= _t,
                    affine_action=action_above,
                    description="if_above",
                ),
                MTATransition(
                    source=q_start,
                    target=q_below,
                    condition=lambda x, _t=threshold: x < _t,
                    affine_action=action_below,
                    description="if_below",
                ),
            ],
            initial_state=q_start,
            accepting_states=[q_above, q_below],
        )

    @staticmethod
    def build_horner_evaluator(coefficients: List[float]) -> MTAProgram:
        n = len(coefficients)
        states = [MTAState("horner_{0}".format(i), i, is_initial=(i == 0), is_accepting=(i == n)) for i in range(n + 1)]

        transitions: List[MTATransition] = []
        transitions.append(
            MTATransition(
                source=states[0],
                target=states[1],
                condition=lambda x: torch.ones_like(x, dtype=torch.bool),
                affine_action=AffineNode(
                    scale_factor=torch.tensor(0.0, dtype=torch.float64),
                    shift_value=torch.tensor(coefficients[-1], dtype=torch.float64),
                ),
                description="init",
            )
        )

        for i in range(1, n):
            idx = n - 1 - i
            transitions.append(
                MTATransition(
                    source=states[i],
                    target=states[i + 1],
                    condition=lambda x: torch.ones_like(x, dtype=torch.bool),
                    affine_action=AffineNode(
                        scale_factor=torch.tensor(1.0, dtype=torch.float64),
                        shift_value=torch.tensor(coefficients[idx], dtype=torch.float64),
                    ),
                    description="horner_step",
                )
            )

        return MTAProgram(
            states=states,
            transitions=transitions,
            initial_state=states[0],
            accepting_states=[states[-1]],
        )

    def compile_to_ast(self, program: MTAProgram) -> ASTNode:
        if not program.transitions:
            return IdentityNode()

        has_branching = False
        for state in program.states:
            outgoing = [t for t in program.transitions if t.source.index == state.index]
            if len(outgoing) > 1:
                has_branching = True
                break

        if not has_branching:
            actions = [t.affine_action for t in program.transitions]
            if len(actions) == 1:
                return actions[0]
            out = actions[0]
            for act in actions[1:]:
                out = ComposeNode(outer=act, inner=out)
            return out

        initial_transitions = [
            t for t in program.transitions if t.source.index == program.initial_state.index
        ]
        branches = []
        for trans in initial_transitions:
            branches.append(
                StratifiedNode.Branch(
                    selector_ast=InputNode("selector"),
                    body_ast=trans.affine_action,
                    domain=(-float("inf"), float("inf")),
                    tear_type="mta_branch",
                )
            )
        if branches:
            return StratifiedNode(branches=branches)
        return IdentityNode()
