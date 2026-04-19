"""
The free algebra for Poema (Evolution 16).

This module provides normalization over affine generators with explicit
rewrite tracing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Tuple

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
)


class RelationType(Enum):
    SCALE_SHIFT_COMMUTATION = auto()  # R1
    SCALE_MULTIPLICATION = auto()  # R2
    SHIFT_ADDITION = auto()  # R3
    SCALE_IDENTITY = auto()  # R4
    SHIFT_IDENTITY = auto()  # R5
    RIGHT_IDENTITY = auto()  # R6
    LEFT_IDENTITY = auto()  # R7


@dataclass
class RewriteRule:
    name: str
    relation: RelationType
    pattern_description: str
    applications: int = 0


@dataclass
class NormalizationTrace:
    steps: List[Tuple[RewriteRule, str, str]] = field(default_factory=list)
    total_rewrites: int = 0

    def record(self, rule: RewriteRule, before: str, after: str) -> None:
        self.steps.append((rule, before, after))
        self.total_rewrites += 1
        rule.applications += 1

    def summary(self) -> str:
        lines = ["Normalization rewrites: {0}".format(self.total_rewrites)]
        for rule, before, after in self.steps:
            lines.append("  {0}: {1} -> {2}".format(rule.name, before, after))
        return "\n".join(lines)


class FreeAlgebra:
    """Knuth-Bendix-like normalization for affine Poema terms."""

    def __init__(self) -> None:
        self.rules = self._build_rules()
        self._normalization_cache: Dict[int, ASTNode] = {}

    def _build_rules(self) -> List[RewriteRule]:
        return [
            RewriteRule(
                "R1 scale_shift_to_affine",
                RelationType.SCALE_SHIFT_COMMUTATION,
                "scale(a) o shift(b) -> affine(a, a*b)",
            ),
            RewriteRule(
                "R2 merge_scales",
                RelationType.SCALE_MULTIPLICATION,
                "scale(a) o scale(b) -> scale(a*b)",
            ),
            RewriteRule(
                "R3 merge_shifts",
                RelationType.SHIFT_ADDITION,
                "shift(a) o shift(b) -> shift(a+b)",
            ),
            RewriteRule(
                "R4 scale_identity",
                RelationType.SCALE_IDENTITY,
                "scale(1) -> id",
            ),
            RewriteRule(
                "R5 shift_identity",
                RelationType.SHIFT_IDENTITY,
                "shift(0) -> id",
            ),
            RewriteRule(
                "R6 right_identity",
                RelationType.RIGHT_IDENTITY,
                "compose(f, id) -> f",
            ),
            RewriteRule(
                "R7 left_identity",
                RelationType.LEFT_IDENTITY,
                "compose(id, f) -> f",
            ),
        ]

    def normalize(
        self,
        node: ASTNode,
        max_iterations: int = 100,
    ) -> Tuple[ASTNode, NormalizationTrace]:
        trace = NormalizationTrace()
        current = node
        for _ in range(max_iterations):
            rewritten, changed = self._apply_rules_once(current, trace)
            current = rewritten
            if not changed:
                break
        return current, trace

    def _apply_rules_once(
        self,
        node: ASTNode,
        trace: NormalizationTrace,
    ) -> Tuple[ASTNode, bool]:
        any_child_changed = False
        new_children = []
        for child in node.children:
            if isinstance(child, ASTNode):
                nchild, changed = self._apply_rules_once(child, trace)
                new_children.append(nchild)
                any_child_changed = any_child_changed or changed
            else:
                new_children.append(child)

        if any_child_changed:
            node = self._rebuild_with_children(node, new_children)

        rewritten, did_rewrite = self._try_rewrite(node, trace)
        return rewritten, any_child_changed or did_rewrite

    def _try_rewrite(self, node: ASTNode, trace: NormalizationTrace) -> Tuple[ASTNode, bool]:
        before = repr(node)

        if isinstance(node, ScaleNode):
            one = torch.tensor(1.0, dtype=node.factor.dtype)
            zero = torch.tensor(0.0, dtype=node.factor.dtype)
            if torch.allclose(node.factor, one):
                result = node.children[0] if node.children else IdentityNode(node.geometric_type)
                trace.record(self.rules[3], before, repr(result))
                return result, True
            if torch.allclose(node.factor, zero):
                result = ConstantNode(0.0, node.geometric_type)
                trace.record(self.rules[3], before, repr(result))
                return result, True
            if node.children and isinstance(node.children[0], ScaleNode):
                inner = node.children[0]
                result = ScaleNode(
                    factor=node.factor * inner.factor,
                    child=inner.children[0] if inner.children else None,
                    geometric_type=node.geometric_type,
                )
                trace.record(self.rules[1], before, repr(result))
                return result, True

        if isinstance(node, ShiftNode):
            zero = torch.tensor(0.0, dtype=node.value.dtype)
            if torch.allclose(node.value, zero):
                result = node.children[0] if node.children else IdentityNode(node.geometric_type)
                trace.record(self.rules[4], before, repr(result))
                return result, True
            if node.children and isinstance(node.children[0], ShiftNode):
                inner = node.children[0]
                result = ShiftNode(
                    value=node.value + inner.value,
                    child=inner.children[0] if inner.children else None,
                    geometric_type=node.geometric_type,
                )
                trace.record(self.rules[2], before, repr(result))
                return result, True

        if isinstance(node, ComposeNode):
            outer = node.outer
            inner = node.inner

            if isinstance(inner, IdentityNode):
                trace.record(self.rules[5], before, repr(outer))
                return outer, True
            if isinstance(outer, IdentityNode):
                trace.record(self.rules[6], before, repr(inner))
                return inner, True
            if isinstance(outer, ScaleNode) and isinstance(inner, ShiftNode):
                result = AffineNode(
                    scale_factor=outer.factor,
                    shift_value=outer.factor * inner.value,
                    child=inner.children[0] if inner.children else None,
                    geometric_type=node.geometric_type,
                )
                trace.record(self.rules[0], before, repr(result))
                return result, True
            if isinstance(outer, AffineNode) and isinstance(inner, AffineNode):
                result = AffineNode(
                    scale_factor=outer.scale_factor * inner.scale_factor,
                    shift_value=outer.scale_factor * inner.shift_value + outer.shift_value,
                    child=inner.children[0] if inner.children else None,
                    geometric_type=node.geometric_type,
                )
                trace.record(self.rules[0], before, repr(result))
                return result, True

        if isinstance(node, AffineNode):
            one = torch.tensor(1.0, dtype=node.scale_factor.dtype)
            zero = torch.tensor(0.0, dtype=node.scale_factor.dtype)
            if torch.allclose(node.scale_factor, one) and torch.allclose(node.shift_value, zero):
                result = node.children[0] if node.children else IdentityNode(node.geometric_type)
                trace.record(self.rules[3], before, repr(result))
                return result, True
            if torch.allclose(node.scale_factor, zero):
                result = ConstantNode(node.shift_value, node.geometric_type)
                trace.record(self.rules[3], before, repr(result))
                return result, True

        return node, False

    def _rebuild_with_children(self, node: ASTNode, new_children: List[ASTNode]) -> ASTNode:
        if isinstance(node, ComposeNode) and len(new_children) >= 2:
            outer, inner = new_children[0], new_children[1]
            if isinstance(outer, ASTNode) and isinstance(inner, ASTNode):
                return ComposeNode(outer=outer, inner=inner)
        node.children = new_children
        return node

    def to_word(self, node: ASTNode) -> List[str]:
        if isinstance(node, IdentityNode):
            return ["id"]
        if isinstance(node, ConstantNode):
            return ["const({0:.6f})".format(float(node.value.item()))]
        if isinstance(node, ScaleNode):
            out = ["scale({0:.6f})".format(float(node.factor.item()))]
            for child in node.children:
                if isinstance(child, ASTNode):
                    out.extend(self.to_word(child))
            return out
        if isinstance(node, ShiftNode):
            out = ["shift({0:.6f})".format(float(node.value.item()))]
            for child in node.children:
                if isinstance(child, ASTNode):
                    out.extend(self.to_word(child))
            return out
        if isinstance(node, AffineNode):
            return [
                "affine({0:.6f},{1:.6f})".format(
                    float(node.scale_factor.item()), float(node.shift_value.item())
                )
            ]
        if isinstance(node, ComposeNode):
            return self.to_word(node.outer) + ["o"] + self.to_word(node.inner)
        if isinstance(node, PolynomialNode):
            return ["horner({0})".format(int(node.coefficients.numel()))]
        if isinstance(node, InputNode):
            return ["x:{0}".format(node.name)]
        return ["<{0}>".format(node.tag.name)]

    def word_length(self, node: ASTNode) -> int:
        word = self.to_word(node)
        return len([w for w in word if w not in ("id", "o")])

    def is_in_normal_form(self, node: ASTNode) -> bool:
        _, trace = self.normalize(node, max_iterations=1)
        return trace.total_rewrites == 0

    def enumerate_equivalence_class(self, node: ASTNode, max_depth: int = 3) -> List[ASTNode]:
        # max_depth is currently reserved for future inverse-rule expansion.
        _ = max_depth
        canonical, _ = self.normalize(node)
        out = [canonical]
        if isinstance(canonical, AffineNode):
            scale = float(canonical.scale_factor.item())
            if abs(scale) > 1e-30:
                out.append(
                    ComposeNode(
                        outer=ScaleNode(canonical.scale_factor.clone(), geometric_type=canonical.geometric_type),
                        inner=ShiftNode(
                            canonical.shift_value / canonical.scale_factor,
                            geometric_type=canonical.geometric_type,
                        ),
                    )
                )
            out.append(
                ComposeNode(
                    outer=ShiftNode(canonical.shift_value.clone(), geometric_type=canonical.geometric_type),
                    inner=ScaleNode(canonical.scale_factor.clone(), geometric_type=canonical.geometric_type),
                )
            )
        return out
