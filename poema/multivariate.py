"""
Soporte multivariable real en Poema.
Implementa diferenciación automática simbólica para funciones
de múltiples variables, compilable a secuencias FMA.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import torch

from .ast_nodes import (
    ASTNode, InputNode, ConstantNode, ScaleNode, ShiftNode,
    ComposeNode, TranscendentalNode, PolynomialNode,
    Scalar,
)
from .frontend import _CompoundAddNode, _CompoundMulNode, Poem


def _symbolic_diff(node: ASTNode, var: str, frontend: Poem, domain: Tuple[float, float], degree: int) -> ASTNode:
    """Compute symbolic derivative using the parser's derivative logic."""
    # Base cases
    if isinstance(node, InputNode):
        if node.name == var:
            return ConstantNode(torch.tensor(1.0, dtype=frontend.dtype))
        else:
            return ConstantNode(torch.tensor(0.0, dtype=frontend.dtype))
    
    if isinstance(node, ConstantNode):
        return ConstantNode(torch.tensor(0.0, dtype=frontend.dtype))
    
    if isinstance(node, ScaleNode):
        child_deriv = _symbolic_diff(
            node.children[0] if node.children else InputNode("x"),
            var, frontend, domain, degree
        )
        return ScaleNode(node.factor, child=child_deriv)
    
    if isinstance(node, ShiftNode):
        child_deriv = _symbolic_diff(
            node.children[0] if node.children else InputNode("x"),
            var, frontend, domain, degree
        )
        return child_deriv
    
    if isinstance(node, TranscendentalNode):
        name = node.name.lower()
        inner_node = node.polynomial.input_node or InputNode("x")
        
        if name == 'sin':
            deriv = frontend.cos(inner_node, domain=domain, degree=degree)
        elif name == 'cos':
            sin_node = frontend.sin(inner_node, domain=domain, degree=degree)
            deriv = ScaleNode(torch.tensor(-1.0, dtype=frontend.dtype), child=sin_node)
        elif name == 'exp':
            deriv = frontend.exp(inner_node, domain=domain, degree=degree)
        elif name == 'log':
            inv = ScaleNode(torch.tensor(-1.0, dtype=frontend.dtype), child=inner_node)
            deriv = frontend._ast_pow(inner_node, inv)
        elif name == 'tanh':
            tanh_node = frontend.tanh(inner_node, domain=domain, degree=degree)
            tanh_sq = frontend._ast_mul(tanh_node, tanh_node)
            one = ConstantNode(torch.tensor(1.0, dtype=frontend.dtype))
            deriv = frontend._ast_sub(one, tanh_sq)
        elif name == 'sigmoid':
            sig_node = frontend.sigmoid(inner_node, domain=domain, degree=degree)
            one = ConstantNode(torch.tensor(1.0, dtype=frontend.dtype))
            one_minus_sig = frontend._ast_sub(one, sig_node)
            deriv = frontend._ast_mul(sig_node, one_minus_sig)
        else:
            raise ValueError(f"Derivative not supported for function: {name}")
        
        # Chain rule
        if not (isinstance(inner_node, InputNode) and inner_node.name == var):
            inner_deriv = _symbolic_diff(inner_node, var, frontend, domain, degree)
            deriv = frontend._ast_mul(deriv, inner_deriv)
        
        return deriv
    
    if isinstance(node, _CompoundAddNode):
        left_deriv = _symbolic_diff(node.left, var, frontend, domain, degree)
        right_deriv = _symbolic_diff(node.right, var, frontend, domain, degree)
        return _CompoundAddNode(left_deriv, right_deriv)
    
    if isinstance(node, _CompoundMulNode):
        # Product rule: d(f*g) = f'*g + f*g'
        left_deriv = _symbolic_diff(node.left, var, frontend, domain, degree)
        right_deriv = _symbolic_diff(node.right, var, frontend, domain, degree)
        term1 = frontend._ast_mul(left_deriv, node.right)
        term2 = frontend._ast_mul(node.left, right_deriv)
        return _CompoundAddNode(term1, term2)
    
    # Default: return zero constant
    return ConstantNode(torch.tensor(0.0, dtype=frontend.dtype))


@dataclass
class MultivariateExpr:
    """
    Representación de una función multivariable.
    f: R^n → R^m
    """
    variables: List[str]         # ['x', 'y', 'z', ...]
    components: List[ASTNode]    # un AST por componente de salida
    _frontend: Optional[Poem] = field(default=None, repr=False)
    _domain: Tuple[float, float] = (-5.0, 5.0)
    _degree: int = 24
    
    def gradient(self, var: str) -> 'MultivariateExpr':
        """Gradiente respecto a una variable."""
        if self._frontend is None:
            self._frontend = Poem(dtype=torch.float64)
        return MultivariateExpr(
            variables=self.variables,
            components=[_symbolic_diff(comp, var, self._frontend, self._domain, self._degree) for comp in self.components],
            _frontend=self._frontend,
            _domain=self._domain,
            _degree=self._degree,
        )
    
    def jacobian(self) -> 'JacobianExpr':
        """Matriz Jacobiana: d(f_i)/d(x_j)."""
        if self._frontend is None:
            self._frontend = Poem(dtype=torch.float64)
        rows = []
        for comp in self.components:
            row = []
            for var in self.variables:
                row.append(_symbolic_diff(comp, var, self._frontend, self._domain, self._degree))
            rows.append(row)
        return JacobianExpr(rows, self.variables)


@dataclass  
class JacobianExpr:
    """Representación de la matriz Jacobiana."""
    entries: List[List[ASTNode]]  # entries[i][j] = df_i/dx_j
    variables: List[str]
    
    def compile(self, compiler, domain):
        """Compila cada entrada a una función FMA."""
        compiled = []
        for row in self.entries:
            compiled_row = []
            for entry in row:
                fn, report = compiler.compile(entry, domain=domain)
                compiled_row.append((fn, report))
            compiled.append(compiled_row)
        return compiled


def parse_multivariate(expr_str: str, variables: List[str], domain: Tuple[float, float] = (-5.0, 5.0), degree: int = 24) -> MultivariateExpr:
    """
    Parsea una expresión multivariable.
    
    Ejemplo:
        parse_multivariate("x^2 + y^2", ["x", "y"])
        parse_multivariate("[x*cos(y), x*sin(y)]", ["x", "y"])
    """
    from .frontend import _RecursiveDescentParser
    
    # Crear un Poem temporal para el parsing
    P = Poem(dtype=torch.float64)
    for var in variables:
        P.input(var)
    
    # Detectar si es expresión vectorial [f1, f2, ...]
    expr_str = expr_str.strip()
    if expr_str.startswith('[') and expr_str.endswith(']'):
        # Parsear cada componente
        inner = expr_str[1:-1]
        components_str = _split_top_level_commas(inner)
    else:
        components_str = [expr_str]
    
    components = []
    for comp_str in components_str:
        parser = _RecursiveDescentParser(comp_str, P, domain, degree)
        ast = parser.parse()
        components.append(ast)
    
    return MultivariateExpr(
        variables=variables,
        components=components,
        _frontend=P,
        _domain=domain,
        _degree=degree,
    )


def _split_top_level_commas(s: str) -> List[str]:
    """Divide por comas en nivel superior (ignora comas dentro de paréntesis)."""
    parts = []
    depth = 0
    current = []
    for ch in s:
        if ch == '(':
            depth += 1
            current.append(ch)
        elif ch == ')':
            depth -= 1
            current.append(ch)
        elif ch == ',' and depth == 0:
            parts.append(''.join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        parts.append(''.join(current).strip())
    return parts
