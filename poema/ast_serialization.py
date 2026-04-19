"""
AST Serialization for Poema.

Native serialization/deserialization of AST to JSON.
Preserves structure, geometric types, error bounds, and compilation metadata.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import torch

from .ast_nodes import (
    ASTNode, InputNode, ConstantNode, IdentityNode,
    ScaleNode, ShiftNode, AffineNode, ComposeNode,
    PolynomialNode, TranscendentalNode, StratifiedNode,
    ParameterNode, ConstraintNode, Scalar
)
from .frontend import _CompoundAddNode, _CompoundMulNode


class ASTSerializer:
    """
    Native serialization/deserialization of AST to JSON.
    
    Preserves:
    - Complete AST structure
    - Geometric types
    - Certified error bounds
    - Compilation metadata
    """
    
    VERSION = "1.0"
    
    @staticmethod
    def to_dict(node: ASTNode) -> Dict[str, Any]:
        """Convert AST to serializable dictionary."""
        base = {
            '__type__': type(node).__name__,
            '__version__': ASTSerializer.VERSION,
        }
        
        if isinstance(node, InputNode):
            return {**base, 'name': node.name}
        
        if isinstance(node, ConstantNode):
            return {**base, 'value': float(node.value.item())}
        
        if isinstance(node, IdentityNode):
            return {**base}
        
        if isinstance(node, ScaleNode):
            return {
                **base,
                'factor': float(node.factor.item()),
                'child': ASTSerializer.to_dict(node.children[0]) if node.children else None,
            }
        
        if isinstance(node, ShiftNode):
            return {
                **base,
                'value': float(node.value.item()),
                'child': ASTSerializer.to_dict(node.children[0]) if node.children else None,
            }
        
        if isinstance(node, AffineNode):
            return {
                **base,
                'scale_factor': float(node.scale_factor.item()),
                'shift_value': float(node.shift_value.item()),
                'child': ASTSerializer.to_dict(node.children[0]) if node.children else None,
            }
        
        if isinstance(node, ComposeNode):
            return {
                **base,
                'outer': ASTSerializer.to_dict(node.outer),
                'inner': ASTSerializer.to_dict(node.inner),
            }
        
        if isinstance(node, PolynomialNode):
            return {
                **base,
                'coefficients': node.coefficients.tolist() if hasattr(node.coefficients, 'tolist') else list(node.coefficients),
            }
        
        if isinstance(node, TranscendentalNode):
            return {
                **base,
                'fn_name': node.name,
                'domain': list(node.original_domain) if node.original_domain else None,
                'degree': len(node.polynomial.coefficients) if hasattr(node.polynomial, 'coefficients') else None,
                'epsilon_certified': node.certified_epsilon,
                'polynomial': ASTSerializer.to_dict(node.polynomial),
            }
        

        
        if isinstance(node, _CompoundAddNode):
            return {
                **base,
                'left': ASTSerializer.to_dict(node.left),
                'right': ASTSerializer.to_dict(node.right),
            }
        
        if isinstance(node, _CompoundMulNode):
            return {
                **base,
                'left': ASTSerializer.to_dict(node.left),
                'right': ASTSerializer.to_dict(node.right),
            }
        
        if isinstance(node, TranscendentalNode):
            return {
                **base,
                'name': node.name,
                'polynomial': ASTSerializer.to_dict(node.polynomial),
                'certified_epsilon': node.certified_epsilon,
                'original_domain': list(node.original_domain),
                'evaluation_mode': node.evaluation_mode,
                'chebyshev_coefficients': node.chebyshev_coefficients.tolist() if node.chebyshev_coefficients is not None else None,
            }
            
        if isinstance(node, StratifiedNode):
            return {
                **base,
                'branches': [{
                    'selector_ast': ASTSerializer.to_dict(b.selector_ast),
                    'body_ast': ASTSerializer.to_dict(b.body_ast),
                    'domain': list(b.domain),
                    'tear_type': b.tear_type,
                } for b in node.branches]
            }
            
        if isinstance(node, ParameterNode):
            return {
                **base,
                'name': node.name,
                'initial_value': node.param_value.tolist(),
                'requires_grad': node.requires_grad,
            }
            
        if isinstance(node, ConstraintNode):
            return {
                **base,
                'constraint_type': node.constraint_type,
                'value': node.constraint_value,
            }
        raise NotImplementedError(f"Serialization not implemented for {type(node).__name__}")
    
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> ASTNode:
        """Reconstruct AST from dictionary."""
        node_type = d.get('__type__')
        
        if node_type == 'InputNode':
            return InputNode(d['name'])
        
        if node_type == 'ConstantNode':
            return ConstantNode(torch.tensor(d['value'], dtype=torch.float64))
        
        if node_type == 'IdentityNode':
            return IdentityNode()
        
        if node_type == 'ScaleNode':
            child = ASTSerializer.from_dict(d['child']) if d.get('child') else None
            return ScaleNode(torch.tensor(d['factor'], dtype=torch.float64), child=child)
        
        if node_type == 'ShiftNode':
            child = ASTSerializer.from_dict(d['child']) if d.get('child') else None
            return ShiftNode(torch.tensor(d['value'], dtype=torch.float64), child=child)
        
        if node_type == 'AffineNode':
            child = ASTSerializer.from_dict(d['child']) if d.get('child') else None
            return AffineNode(
                torch.tensor(d['scale_factor'], dtype=torch.float64),
                torch.tensor(d['shift_value'], dtype=torch.float64),
                child=child,
            )
        
        if node_type == 'ComposeNode':
            return ComposeNode(
                outer=ASTSerializer.from_dict(d['outer']),
                inner=ASTSerializer.from_dict(d['inner']),
            )
        
        if node_type == 'PolynomialNode':
            coeffs = torch.tensor(d['coefficients'], dtype=torch.float64)
            return PolynomialNode(coeffs)
        
        if node_type == 'TranscendentalNode':
            poly = ASTSerializer.from_dict(d['polynomial'])
            from .ast_nodes import TranscendentalNode, Scalar
            node = TranscendentalNode(
                name=d['fn_name'],
                polynomial=poly,
                certified_epsilon=d.get('epsilon_certified', 0.0),
                original_domain=tuple(d['domain']) if d.get('domain') else None,
                geometric_type=Scalar(),
            )
            return node
        

        
        if node_type == '_CompoundAddNode':
            return _CompoundAddNode(
                ASTSerializer.from_dict(d['left']),
                ASTSerializer.from_dict(d['right']),
            )
        
        if node_type == '_CompoundMulNode':
            return _CompoundMulNode(
                ASTSerializer.from_dict(d['left']),
                ASTSerializer.from_dict(d['right']),
            )
        
        if node_type == 'TranscendentalNode':
            cheby_coeffs = d.get('chebyshev_coefficients')
            if cheby_coeffs is not None:
                cheby_coeffs = torch.tensor(cheby_coeffs, dtype=torch.float64)
            return TranscendentalNode(
                name=d['name'],
                polynomial=ASTSerializer.from_dict(d['polynomial']),
                certified_epsilon=d['certified_epsilon'],
                original_domain=tuple(d['original_domain']),
                chebyshev_coefficients=cheby_coeffs,
                evaluation_mode=d.get('evaluation_mode', 'horner')
            )
            
        if node_type == 'StratifiedNode':
            branches = []
            for b in d['branches']:
                branches.append(StratifiedNode.Branch(
                    selector_ast=ASTSerializer.from_dict(b['selector_ast']),
                    body_ast=ASTSerializer.from_dict(b['body_ast']),
                    domain=tuple(b['domain']),
                    tear_type=b.get('tear_type')
                ))
            return StratifiedNode(branches)
            
        if node_type == 'ParameterNode':
            return ParameterNode(
                name=d['name'],
                initial_value=torch.tensor(d['initial_value'], dtype=torch.float64),
                requires_grad=d['requires_grad']
            )
            
        if node_type == 'ConstraintNode':
            return ConstraintNode(
                constraint_type=d['constraint_type'],
                value=d['value']
            )
        raise NotImplementedError(f"Deserialization not implemented for {node_type}")
    
    @staticmethod
    def save(node: ASTNode, path: str, metadata: Optional[Dict] = None) -> None:
        """Save AST to JSON file with optional metadata."""
        data = {
            'ast': ASTSerializer.to_dict(node),
            'metadata': metadata or {},
            'poema_version': '5.0.0',
            'timestamp': datetime.now().isoformat(),
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @staticmethod
    def load(path: str) -> Tuple[ASTNode, Dict]:
        """Load AST from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return ASTSerializer.from_dict(data['ast']), data.get('metadata', {})


# Integration methods for ASTNode
def ast_to_json(node: ASTNode) -> str:
    """Convert AST to JSON string."""
    return json.dumps(ASTSerializer.to_dict(node), indent=2)


def ast_from_json(json_str: str) -> ASTNode:
    """Reconstruct AST from JSON string."""
    return ASTSerializer.from_dict(json.loads(json_str))


def ast_save(node: ASTNode, path: str, metadata: Optional[Dict] = None) -> None:
    """Save AST to JSON file."""
    ASTSerializer.save(node, path, metadata)


def ast_load(path: str) -> Tuple[ASTNode, Dict]:
    """Load AST from JSON file."""
    return ASTSerializer.load(path)
