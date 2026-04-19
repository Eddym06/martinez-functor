"""
ONNX export for Poema programs.

Exports Poema AST to ONNX format for interoperability with other ML frameworks.
Maps Poema nodes to ONNX operators with certified metadata embedded in the model.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
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
    TranscendentalNode,
    ParameterNode,
    ConstraintNode,
)
from .frontend import _CompoundAddNode, _CompoundMulNode
from .compiler import FMALinearizer, FMAInstruction


class PoemONNXExporter:
    """
    Exporta programas Poema a formato ONNX para interoperabilidad.
    
    Mapeo:
    - ScaleNode → ONNX Mul
    - ShiftNode → ONNX Add  
    - ComposeNode → secuencia de nodos ONNX
    - PolynomialNode → ONNX serie de FMA
    - TranscendentalNode → nodo personalizado con certificado embebido
    - AffineNode → ONNX FMA (Mul + Add)
    """
    
    def __init__(self, opset_version: int = 17):
        self.opset_version = opset_version
        self._node_counter = 0
        self.inputs = []
        self.params = {}
        self._nodes = []
        self._initializers = []
        self._value_infos = []
    
    def _new_name(self, prefix: str) -> str:
        self._node_counter += 1
        return f"{prefix}_{self._node_counter}"
    
    def _add_initializer(self, name: str, value: np.ndarray) -> None:
        # Convert numpy array to list for JSON serialization
        self._initializers.append({'name': name, 'value': value.tolist()})
    
    def _add_node(self, op_type: str, inputs: List[str], outputs: List[str], 
                  attributes: Optional[Dict[str, Any]] = None,
                  domain: str = "") -> None:
        node = {
            'op_type': op_type,
            'inputs': inputs,
            'outputs': outputs,
            'attributes': attributes or {},
            'domain': domain,
        }
        self._nodes.append(node)
    
    def export(self, ast: ASTNode, 
               input_shape: Tuple[int, ...] = (1,),
               output_path: Optional[str] = None,
               domain: Optional[Tuple[float, float]] = None) -> Dict[str, Any]:
        """
        Exporta AST a representación ONNX.
        
        Returns dict con estructura ONNX que puede ser guardada con onnx.save().
        """
        self._nodes = []
        self._initializers = []
        self._value_infos = []
        self._node_counter = 0
        self.inputs = []
        self.params = {}
        
        input_name = 'input'
        self.inputs = [input_name]
        output_name = self._ast_to_onnx(ast, input_name)
        
        # Construir representación ONNX
        model = {
            'graph': {
                'name': 'poema_function',
                'inputs': [{
                    'name': input_name,
                    'shape': list(input_shape),
                    'dtype': 'double',
                }],
                'outputs': [{
                    'name': output_name,
                    'shape': list(input_shape),
                    'dtype': 'double',
                }],
                'nodes': self._nodes,
                'initializers': self._initializers,
                'value_infos': self._value_infos,
            },
            'opset_imports': [{'version': self.opset_version}],
            'metadata': {
                'poema_epsilon_certified': self._get_epsilon(ast, domain),
                'poema_certificate_source': self._get_certificate_source(ast, domain),
                'poema_version': '5.0.0',
            },
        }
        
        if output_path:
            self._save_onnx(model, output_path)
        
        return model
    
    def _get_epsilon(self, ast: ASTNode, domain: Optional[Tuple[float, float]] = None) -> float:
        """Obtener epsilon certificado del AST."""
        if isinstance(ast, TranscendentalNode):
            return ast.certified_epsilon
        if isinstance(ast, PolynomialNode):
            return 0.0
        if isinstance(ast, ComposeNode):
            return max(
                self._get_epsilon(ast.outer, domain),
                self._get_epsilon(ast.inner, domain)
            )
        return 0.0
    
    def _get_certificate_source(self, ast: ASTNode, domain: Optional[Tuple[float, float]] = None) -> str:
        """Obtener fuente del certificado."""
        if isinstance(ast, TranscendentalNode):
            if ast.original_domain and domain:
                if (ast.original_domain[0] <= domain[0] and 
                    ast.original_domain[1] >= domain[1]):
                    return "lean_synchronized"
            return "local_estimate"
        if isinstance(ast, ComposeNode):
            outer_src = self._get_certificate_source(ast.outer, domain)
            inner_src = self._get_certificate_source(ast.inner, domain)
            if outer_src == "lean_synchronized" and inner_src == "lean_synchronized":
                return "lean_synchronized"
            return "mixed"
        return "none"
    
    def _ast_to_onnx(self, node: ASTNode, input_name: str) -> str:
        """Convierte nodo AST a nodos ONNX, retorna nombre de salida."""
        
        if isinstance(node, InputNode):
            return input_name
        
        if isinstance(node, IdentityNode):
            return input_name
        
        if isinstance(node, ConstantNode):
            name = self._new_name('const')
            value = np.array([node.value.item()], dtype=np.float64)
            self._add_initializer(name, value)
            
            # Broadcast constant to match input shape
            output = self._new_name('const_out')
            self._add_node('ConstantOfShape', [name], [output])
            return output
        
        if isinstance(node, ScaleNode):
            alpha_name = self._new_name('alpha')
            alpha_value = np.array([node.factor.item()], dtype=np.float64)
            self._add_initializer(alpha_name, alpha_value)
            
            child_input = self._ast_to_onnx(
                node.children[0] if node.children else InputNode("x"),
                input_name
            )
            output = self._new_name('scale_out')
            self._add_node('Mul', [child_input, alpha_name], [output])
            return output
        
        if isinstance(node, ShiftNode):
            beta_name = self._new_name('beta')
            beta_value = np.array([node.value.item()], dtype=np.float64)
            self._add_initializer(beta_name, beta_value)
            
            child_input = self._ast_to_onnx(
                node.children[0] if node.children else InputNode("x"),
                input_name
            )
            output = self._new_name('shift_out')
            self._add_node('Add', [child_input, beta_name], [output])
            return output
        
        if isinstance(node, AffineNode):
            child_input = self._ast_to_onnx(
                node.children[0] if node.children else InputNode("x"),
                input_name
            )
            
            # FMA: y = a*x + b
            alpha_name = self._new_name('affine_a')
            alpha_value = np.array([node.scale_factor.item()], dtype=np.float64)
            self._add_initializer(alpha_name, alpha_value)
            
            beta_name = self._new_name('affine_b')
            beta_value = np.array([node.shift_value.item()], dtype=np.float64)
            self._add_initializer(beta_name, beta_value)
            
            mul_out = self._new_name('affine_mul')
            self._add_node('Mul', [child_input, alpha_name], [mul_out])
            
            output = self._new_name('affine_out')
            self._add_node('Add', [mul_out, beta_name], [output])
            return output
        
        if isinstance(node, ComposeNode):
            inner_out = self._ast_to_onnx(node.inner, input_name)
            return self._ast_to_onnx(node.outer, inner_out)
        
        if isinstance(node, PolynomialNode):
            coeffs = node.coefficients.numpy() if hasattr(node.coefficients, 'numpy') else np.array(node.coefficients)
            n = len(coeffs)
            if n == 0:
                name = self._new_name('poly_zero')
                self._add_initializer(name, np.array([0.0], dtype=np.float64))
                return name
            
            current = self._new_name('poly_c_n')
            self._add_initializer(current, np.array([coeffs[-1]], dtype=np.float64))
            
            for i in range(n-2, -1, -1):
                c_name = self._new_name(f'poly_c_{i}')
                self._add_initializer(c_name, np.array([coeffs[i]], dtype=np.float64))
                
                mul_out = self._new_name('poly_mul')
                self._add_node('Mul', [current, input_name], [mul_out])
                
                add_out = self._new_name('poly_add')
                self._add_node('Add', [mul_out, c_name], [add_out])
                current = add_out
            
            return current
        

        
        if isinstance(node, _CompoundAddNode):
            left_out = self._ast_to_onnx(node.left, input_name)
            right_out = self._ast_to_onnx(node.right, input_name)
            output = self._new_name('add_out')
            self._add_node('Add', [left_out, right_out], [output])
            return output
        
        if isinstance(node, _CompoundMulNode):
            left_out = self._ast_to_onnx(node.left, input_name)
            right_out = self._ast_to_onnx(node.right, input_name)
            output = self._new_name('mul_out')
            self._add_node('Mul', [left_out, right_out], [output])
            return output
        
        if isinstance(node, TranscendentalNode):
            # For ONNX export we can rely on its polynomial evaluated by horner sequence
            return self._ast_to_onnx(node.polynomial, input_name)
            
        if isinstance(node, ParameterNode):
            if node.name not in self.inputs:
                self.params[node.name] = node.param_value.detach().cpu().numpy()
            return node.name
            
        if isinstance(node, ConstraintNode):
            # A Constraint doesn't influence forward computation in ONNX, it acts as identity
            return input_name
            
        if isinstance(node, StratifiedNode):
            # Evaluate all branches conceptually and use `Where` combined with geometric domain overlaps
            n_branches = len(node.branches)
            if n_branches == 0:
                return input_name
            
            # Start logic with the last branch as the base background condition 
            # Alternatively start with 0.0, but starting with the last branch prevents having to handle unfilled gaps if domain covers all
            current_out = self._ast_to_onnx(node.branches[-1].body_ast, input_name)
            
            # Iterate backwards through the branches so earlier branches have priority (like PyTorch sequential checks)
            for branch in reversed(node.branches[:-1]):
                b_out = self._ast_to_onnx(branch.body_ast, input_name)
                
                # Check (x >= domain_a) AND (x < domain_b)
                a, b = branch.domain
                a_name = self._new_name('strat_a')
                b_name = self._new_name('strat_b')
                
                self._add_initializer(a_name, np.array([a], dtype=np.float64))
                self._add_initializer(b_name, np.array([b], dtype=np.float64))
                
                greater_eq_name = self._new_name('ge')
                self._add_node('GreaterOrEqual', [input_name, a_name], [greater_eq_name])
                
                less_name = self._new_name('lt')
                self._add_node('Less', [input_name, b_name], [less_name])
                
                mask_name = self._new_name('mask')
                self._add_node('And', [greater_eq_name, less_name], [mask_name])
                
                next_out = self._new_name('where_out')
                self._add_node('Where', [mask_name, b_out, current_out], [next_out])
                current_out = next_out

            return current_out
        raise NotImplementedError(f"ONNX export not implemented for {type(node).__name__}")
    
    def _save_onnx(self, model: Dict[str, Any], path: str) -> None:
        """Save ONNX model to file."""
        try:
            import onnx
            from onnx import helper, TensorProto, numpy_helper
            
            # Create initializers
            onnx_initializers = []
            for init in model['graph']['initializers']:
                import numpy as np
                tensor_array = np.array(init['value']) if isinstance(init['value'], list) else init['value']
                tensor = numpy_helper.from_array(tensor_array, name=init['name'])
                onnx_initializers.append(tensor)
            
            # Create nodes
            onnx_nodes = []
            for node in model['graph']['nodes']:
                attrs = []
                for key, val in node.get('attributes', {}).items():
                    if isinstance(val, float):
                        attrs.append(helper.make_attribute(key, val))
                    elif isinstance(val, str):
                        attrs.append(helper.make_attribute(key, val))
                
                onnx_node = helper.make_node(
                    node['op_type'],
                    node['inputs'],
                    node['outputs'],
                    domain=node.get('domain', ''),
                )
                for attr in attrs:
                    onnx_node.attribute.append(attr)
                onnx_nodes.append(onnx_node)
            
            # Create graph
            graph = helper.make_graph(
                onnx_nodes,
                model['graph']['name'],
                [helper.make_tensor_value_info(
                    model['graph']['inputs'][0]['name'],
                    TensorProto.DOUBLE,
                    model['graph']['inputs'][0]['shape']
                )],
                [helper.make_tensor_value_info(
                    model['graph']['outputs'][0]['name'],
                    TensorProto.DOUBLE,
                    model['graph']['outputs'][0]['shape']
                )],
                initializer=onnx_initializers,
            )
            
            # Create model
            onnx_model = helper.make_model(
                graph,
                opset_imports=[helper.make_opsetid("", model['opset_imports'][0]['version'])]
            )
            
            # Add metadata
            for key, val in model.get('metadata', {}).items():
                meta = onnx_model.metadata_props.add()
                meta.key = key
                meta.value = str(val)
            
            onnx.save(onnx_model, path)
            
        except ImportError:
            # Fallback: save as JSON
            import json
            json_path = path.replace('.onnx', '.json')
            with open(json_path, 'w') as f:
                json.dump(model, f, indent=2)


def export_to_onnx(ast: ASTNode, 
                   output_path: str,
                   input_shape: Tuple[int, ...] = (1,),
                   domain: Optional[Tuple[float, float]] = None) -> Dict[str, Any]:
    """
    Convenience function to export Poema AST to ONNX.
    
    Args:
        ast: Poema AST node
        output_path: Path to save ONNX file
        input_shape: Shape of input tensor
        domain: Domain for certificate validation
    
    Returns:
        ONNX model dict
    """
    exporter = PoemONNXExporter()
    return exporter.export(ast, input_shape=input_shape, output_path=output_path, domain=domain)
