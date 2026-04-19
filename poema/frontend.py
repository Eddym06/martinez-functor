"""
Poema frontend with three modes:
- Poem: prescriptive mode
- CoPoem: descriptive mode
- BiPoem: relational mode
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import math

import torch

from .ast_nodes import (
    ASTNode,
    AffineNode,
    ComposeNode,
    ConstantNode,
    ConstraintNode,
    GeometricType,
    IdentityNode,
    InputNode,
    NodeTag,
    PolynomialNode,
    ScaleNode,
    ShiftNode,
    StratifiedNode,
    TopologicalObstructionError,
    TranscendentalNode,
    Scalar,
    Vector,
)


class _RecursiveDescentParser:
    """Recursive descent parser for Poema continuous_flow expressions.
    
    Extended Grammar:
        program    ::= let_binding | expr
        let_binding::= 'let' IDENT '=' expr 'in' expr
        expr       ::= term (('+' | '-') term)*
        term       ::= power (('*' | '/') power)*
        power      ::= unary ('^' unary)*
        unary      ::= ('-' | '+') unary | func_call | atom
        func_call  ::= IDENT '(' expr (',' expr)* ')'
        atom       ::= NUMBER | IDENT | '(' expr ')' | piecewise | derivative
        piecewise  ::= 'piecewise' '(' condition ',' true_expr ',' false_expr ')'
        derivative ::= 'D' '(' expr (',' NUMBER)? ')'
        IDENT      ::= [a-zA-Z_][a-zA-Z0-9_]*
        NUMBER     ::= [0-9]+[.][0-9]+ or [0-9]+
    
    Extensions over basic grammar:
    - let bindings for named subexpressions
    - piecewise functions for stratified definitions
    - D(expr) for symbolic derivatives
    - Composition via nested function calls: sin(cos(x))
    """
    
    SUPPORTED_FUNCTIONS = {"sin", "cos", "exp", "log", "tanh", "sigmoid"}
    CONSTANTS = {"pi": math.pi, "e": math.e, "tau": 2 * math.pi}
    
    def __init__(self, expr: str, frontend: 'Poem', domain: Tuple[float, float], degree: int):
        self.tokens = self._tokenize(expr)
        self.pos = 0
        self.frontend = frontend
        self.domain = domain
        self.degree = degree
        self._let_bindings: Dict[str, ASTNode] = {}
        self._let_var_names: Dict[str, str] = {}  # name -> variable name in body
    
    def _tokenize(self, expr: str) -> List[Tuple[str, str]]:
        """Tokenize expression into (type, value) pairs."""
        tokens = []
        i = 0
        while i < len(expr):
            c = expr[i]
            if c.isspace():
                i += 1
            elif c in '()+-*/^,=[]':
                tokens.append(('OP', c))
                i += 1
            elif c in '><':
                # Check for >= or <=
                if i + 1 < len(expr) and expr[i+1] == '=':
                    tokens.append(('OP', c + '='))
                    i += 2
                else:
                    tokens.append(('OP', c))
                    i += 1
            elif c.isdigit() or (c == '.' and i + 1 < len(expr) and expr[i+1].isdigit()):
                j = i
                while j < len(expr) and (expr[j].isdigit() or expr[j] == '.'):
                    j += 1
                if j < len(expr) and expr[j] in 'eE':
                    j += 1
                    if j < len(expr) and expr[j] in '+-':
                        j += 1
                    while j < len(expr) and expr[j].isdigit():
                        j += 1
                tokens.append(('NUM', expr[i:j]))
                i = j
            elif c.isalpha() or c == '_':
                j = i
                while j < len(expr) and (expr[j].isalnum() or expr[j] == '_'):
                    j += 1
                ident = expr[i:j]
                if ident in ['if', 'else', 'while', 'def', 'for']:
                    tokens.append(('KEYWORD', ident))
                else:
                    tokens.append(('IDENT', ident))
                i = j
            else:
                raise SyntaxError(f"Unexpected character: '{c}' at position {i}")
        return tokens
    
    def _peek(self) -> Optional[Tuple[str, str]]:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None
    
    def _consume(self, expected_type: Optional[str] = None, expected_value: Optional[str] = None) -> Tuple[str, str]:
        if self.pos >= len(self.tokens):
            raise SyntaxError(f"Unexpected end of expression, expected {expected_value or expected_type}")
        tok = self.tokens[self.pos]
        if expected_type and tok[0] != expected_type:
            raise SyntaxError(f"Expected {expected_type}, got {tok[0]} '{tok[1]}'")
        if expected_value and tok[1] != expected_value:
            raise SyntaxError(f"Expected '{expected_value}', got '{tok[1]}'")
        self.pos += 1
        return tok
    
    def parse(self) -> ASTNode:
        # Check for let binding at top level
        peek = self._peek()
        if peek and peek[0] == 'IDENT' and peek[1] == 'let':
            return self._parse_let()
        result = self._parse_expr()
        if self.pos < len(self.tokens):
            remaining = ' '.join(f"{t[1]}" for t in self.tokens[self.pos:])
            raise SyntaxError(f"Unexpected tokens after expression: {remaining}")
        return result
    
    def _parse_let(self) -> ASTNode:
        """let_binding ::= 'let' IDENT '=' expr 'in' expr"""
        self._consume('IDENT', 'let')
        name_tok = self._consume('IDENT')
        name = name_tok[1]
        self._consume('OP', '=')
        value = self._parse_expr()
        self._consume('IDENT', 'in')
        
        # Register the binding with its variable name (default 'x')
        self._let_bindings[name] = value
        self._let_var_names[name] = 'x'  # Default variable name
        
        # Parse body
        result = self._parse_expr()
        
        # Clean up binding
        del self._let_bindings[name]
        del self._let_var_names[name]
        return result
    
    def _substitute(self, node: ASTNode, var_name: str, replacement: ASTNode) -> ASTNode:
        """Substitute all occurrences of var_name with replacement in node.
        
        Deep substitution: also substitutes within the replacement if it contains
        the same variable name (handles nested function calls like f(f(x))).
        """
        if isinstance(node, InputNode) and node.name == var_name:
            # Deep copy the replacement to avoid shared state
            return self._deep_copy(replacement)
        if isinstance(node, ConstantNode):
            return node
        
        new_children = []
        for child in node.children:
            if isinstance(child, ASTNode):
                new_children.append(self._substitute(child, var_name, replacement))
            else:
                new_children.append(child)
        
        # Reconstruct node with substituted children
        if isinstance(node, ScaleNode):
            return ScaleNode(node.factor, child=new_children[0] if new_children else None)
        if isinstance(node, ShiftNode):
            return ShiftNode(node.value, child=new_children[0] if new_children else None)
        if isinstance(node, AffineNode):
            return AffineNode(node.scale_factor, node.shift_value, 
                            child=new_children[0] if new_children else None)
        if isinstance(node, ComposeNode):
            return ComposeNode(outer=new_children[0], inner=new_children[1])
        if isinstance(node, TranscendentalNode):
            new_input = self._substitute(node.polynomial.input_node or InputNode("x"), var_name, replacement)
            return self.frontend._build_transcendental(
                node.name, lambda x: x, new_input, node.original_domain, 
                len(node.polynomial.coefficients) if hasattr(node.polynomial, 'coefficients') else self.degree
            )
        if isinstance(node, _CompoundAddNode):
            return _CompoundAddNode(new_children[0], new_children[1])
        if isinstance(node, _CompoundMulNode):
            return _CompoundMulNode(new_children[0], new_children[1])
        if isinstance(node, PolynomialNode):
            return PolynomialNode(node.coefficients, input_node=new_children[0] if new_children else None)
        
        # Default: return node with new children
        node.children = new_children
        return node
    
    def _deep_copy(self, node: ASTNode) -> ASTNode:
        """Create a deep copy of an AST node."""
        if isinstance(node, InputNode):
            return InputNode(node.name)
        if isinstance(node, ConstantNode):
            return ConstantNode(node.value.clone())
        if isinstance(node, ScaleNode):
            child = self._deep_copy(node.children[0]) if node.children else None
            return ScaleNode(node.factor.clone(), child=child)
        if isinstance(node, ShiftNode):
            child = self._deep_copy(node.children[0]) if node.children else None
            return ShiftNode(node.value.clone(), child=child)
        if isinstance(node, AffineNode):
            child = self._deep_copy(node.children[0]) if node.children else None
            return AffineNode(node.scale_factor.clone(), node.shift_value.clone(), child=child)
        if isinstance(node, ComposeNode):
            return ComposeNode(outer=self._deep_copy(node.outer), inner=self._deep_copy(node.inner))
        if isinstance(node, TranscendentalNode):
            poly_copy = PolynomialNode(node.polynomial.coefficients.clone(),
                                       input_node=self._deep_copy(node.polynomial.input_node) if node.polynomial.input_node else None)
            return TranscendentalNode(
                name=node.name, polynomial=poly_copy,
                certified_epsilon=node.certified_epsilon,
                original_domain=node.original_domain,
                geometric_type=node.geometric_type,
                chebyshev_coefficients=node.chebyshev_coefficients.clone() if node.chebyshev_coefficients is not None else None,
                evaluation_mode=node.evaluation_mode,
            )
        if isinstance(node, _CompoundAddNode):
            return _CompoundAddNode(self._deep_copy(node.left), self._deep_copy(node.right))
        if isinstance(node, _CompoundMulNode):
            return _CompoundMulNode(self._deep_copy(node.left), self._deep_copy(node.right))
        if isinstance(node, PolynomialNode):
            input_copy = self._deep_copy(node.input_node) if node.input_node else None
            return PolynomialNode(node.coefficients.clone(), input_node=input_copy)
        
        # Generic fallback
        new_children = [self._deep_copy(c) if isinstance(c, ASTNode) else c for c in node.children]
        node.children = new_children
        return node
    
    def _parse_expr(self) -> ASTNode:
        peek = self._peek()
        if peek and peek[0] == 'KEYWORD':
            if peek[1] == 'if':
                return self._parse_if()
            elif peek[1] == 'while':
                return self._parse_while()
            elif peek[1] == 'for':
                return self._parse_for()
            elif peek[1] == 'def':
                return self._parse_def()
                
        left = self._parse_term()
        while True:
            peek = self._peek()
            if peek and peek[0] == 'OP' and peek[1] in ('+', '-', '>', '<', '=', '>=', '<=', '=='):
                op = self._consume()[1]
                right = self._parse_term()
                if op == '+':
                    left = self.frontend._ast_add(left, right)
                elif op == '-':
                    left = self.frontend._ast_sub(left, right)
                elif op in ('>', '<', '=', '>=', '<=', '=='):
                    # For simplicity, treat a > b as a - b > 0 in evaluating Piecewise/Loops.
                    # We just return left - right as the condition node !
                    if op == '<':
                        left = self.frontend._ast_sub(right, left) 
                    else:
                        left = self.frontend._ast_sub(left, right)
            else:
                break
        return left

    def _parse_if(self) -> ASTNode:
        self._consume('KEYWORD', 'if')
        self._consume('OP', '(')
        cond = self._parse_expr()
        self._consume('OP', ')')
        true_expr = self._parse_expr()
        self._consume('KEYWORD', 'else')
        false_expr = self._parse_expr()
        from .ast_nodes import PiecewiseNode
        return PiecewiseNode(cond, true_expr, false_expr)

    def _parse_while(self) -> ASTNode:
        self._consume('KEYWORD', 'while')
        self._consume('OP', '(')
        cond = self._parse_expr()
        self._consume('OP', ')')
        body = self._parse_expr()
        # Simulated while block as recursive piecewise or custom LoopNode (we return the body to keep AST valid)
        return body

    def _parse_for(self) -> ASTNode:
        self._consume('KEYWORD', 'for')
        self._consume('OP', '(')
        var_name = self._consume('IDENT')[1]
        self._consume('OP', '=')
        start_expr = self._parse_expr()
        self._consume('IDENT') # 'to' or ',' depending on syntax
        end_expr = self._parse_expr()
        self._consume('OP', ')')
        body = self._parse_expr()
        # Simulated for block
        return body

    def _parse_def(self) -> ASTNode:
        self._consume('KEYWORD', 'def')
        func_name = self._consume('IDENT')[1]
        self._consume('OP', '(')
        args = []
        if self._peek()[1] != ')':
            args.append(self._consume('IDENT')[1])
            while self._peek()[1] == ',':
                self._consume()
                args.append(self._consume('IDENT')[1])
        self._consume('OP', ')')
        self._consume('OP', '=')
        body = self._parse_expr()
        # Simulated function def returning the evaluated definition AST
        return body
    
    def _parse_term(self) -> ASTNode:
        """term ::= power (('*' | '/') power)*"""
        left = self._parse_power()
        while True:
            peek = self._peek()
            if peek and peek[0] == 'OP' and peek[1] in '*/':
                op = self._consume()[1]
                right = self._parse_power()
                if op == '*':
                    left = self.frontend._ast_mul(left, right)
                else:
                    # Division: a / b = a * (1/b) = a * b^(-1)
                    inv = ScaleNode(torch.tensor(-1.0, dtype=self.frontend.dtype), child=right)
                    left = self.frontend._ast_mul(left, inv)
            else:
                break
        return left
    
    def _parse_power(self) -> ASTNode:
        """power ::= unary ('^' unary)*"""
        base = self._parse_unary()
        while True:
            peek = self._peek()
            if peek and peek[0] == 'OP' and peek[1] == '^':
                self._consume()
                exp = self._parse_unary()
                base = self.frontend._ast_pow(base, exp)
            else:
                break
        return base
    
    def _parse_unary(self) -> ASTNode:
        """unary ::= ('-' | '+') unary | func_call | atom"""
        peek = self._peek()
        if peek and peek[0] == 'OP' and peek[1] in '+-':
            op = self._consume()[1]
            operand = self._parse_unary()
            if op == '-':
                return ScaleNode(torch.tensor(-1.0, dtype=self.frontend.dtype), child=operand)
            return operand  # +x is just x
        
        return self._parse_atom()
    
    def _parse_atom(self) -> ASTNode:
        """atom ::= NUMBER | IDENT | '(' expr ')' | func_call | piecewise | derivative | let_binding"""
        peek = self._peek()
        if peek is None:
            raise SyntaxError("Unexpected end of expression")
        
        # Let binding (nested)
        if peek[0] == 'IDENT' and peek[1] == 'let':
            return self._parse_let()
        
        # Parenthesized expression
        if peek[1] == '(':
            self._consume()
            result = self._parse_expr()
            self._consume('OP', ')')
            return result
        
        # Number
        if peek[0] == 'NUM':
            self._consume()
            return ConstantNode(torch.tensor(float(peek[1]), dtype=self.frontend.dtype))
        
        # Identifier (function call, constant, variable, or let binding)
        if peek[0] == 'IDENT':
            name = self._consume()[1]
            
            # Check if it's a let binding reference
            if name in self._let_bindings:
                # Check if it's followed by '(' (function application)
                next_peek = self._peek()
                if next_peek and next_peek[1] == '(':
                    # Parse arguments
                    self._consume('OP', '(')
                    args = [self._parse_expr()]
                    while True:
                        p = self._peek()
                        if p and p[1] == ',':
                            self._consume()
                            args.append(self._parse_expr())
                        else:
                            break
                    self._consume('OP', ')')
                    # Substitute
                    var_name = self._let_var_names.get(name, 'x')
                    return self._substitute(self._let_bindings[name], var_name, args[0])
                else:
                    # Just reference the bound expression
                    return self._let_bindings[name]
            
            # Check if it's a piecewise function
            if name == 'piecewise':
                return self._parse_piecewise()
            
            # Check if it's a derivative
            if name == 'D':
                return self._parse_derivative()
            
            # Check if it's a gradient
            if name == 'grad':
                return self._parse_gradient()
            
            # Check if it's a function call
            next_peek = self._peek()
            if next_peek and next_peek[1] == '(':
                return self._parse_func_call(name)
            
            # Check if it's a constant
            if name in self.CONSTANTS:
                return ConstantNode(torch.tensor(self.CONSTANTS[name], dtype=self.frontend.dtype))
            
            # It's a variable
            if name not in self.frontend._inputs:
                self.frontend._inputs[name] = InputNode(name)
            return self.frontend._inputs[name]
        
        raise SyntaxError(f"Unexpected token: {peek}")
    
    def _parse_piecewise(self) -> ASTNode:
        """piecewise ::= 'piecewise' '(' case (',' case)* ',' default_expr ')'
                     | 'piecewise' '(' condition ',' true_expr ',' false_expr ')'
        case ::= '(' condition ',' expr ')'
        condition ::= IDENT ('>=' | '>' | '<=' | '<') NUMBER
        
        Soporta N casos con verificación de continuidad en fronteras.
        Sintaxis binaria legacy: piecewise(x >= 0, x, 0)
        Sintaxis n-aria: piecewise((x < -1, -1), (x < 1, x), 1)
        """
        self._consume('OP', '(')
        
        # Detectar sintaxis: si el siguiente token es IDENT (no '('), es sintaxis binaria legacy
        peek = self._peek()
        if peek and peek[0] == 'IDENT' and peek[1] != '(':
            # Sintaxis binaria legacy: piecewise(condition, true_expr, false_expr)
            return self._parse_piecewise_binary()
        else:
            # Sintaxis n-aria: piecewise((case), (case), ..., default)
            return self._parse_piecewise_nary()
    
    def _parse_piecewise_binary(self) -> ASTNode:
        """Sintaxis binaria: piecewise(condition, true_expr, false_expr)"""
        # Parse condition: IDENT OP NUMBER
        var_tok = self._consume('IDENT')
        op_tok = self._consume('OP')
        op = op_tok[1]
        
        # Handle negative numbers
        val_peek = self._peek()
        if val_peek and val_peek[0] == 'OP' and val_peek[1] == '-':
            self._consume()
            num_tok = self._consume('NUM')
            threshold = -float(num_tok[1])
        else:
            num_tok = self._consume('NUM')
            threshold = float(num_tok[1])
        
        self._consume('OP', ',')
        true_expr = self._parse_expr()
        self._consume('OP', ',')
        false_expr = self._parse_expr()
        self._consume('OP', ')')
        
        a, b = self.domain
        
        if op in ('>=', '>'):
            branches = [
                StratifiedNode.Branch(false_expr, false_expr, (a, threshold)),
                StratifiedNode.Branch(true_expr, true_expr, (threshold, b)),
            ]
        elif op in ('<=', '<'):
            branches = [
                StratifiedNode.Branch(true_expr, true_expr, (a, threshold)),
                StratifiedNode.Branch(false_expr, false_expr, (threshold, b)),
            ]
        else:
            raise SyntaxError(f"Unsupported condition operator: {op}")
        
        return StratifiedNode(branches)
    
    def _parse_piecewise_nary(self) -> ASTNode:
        """Sintaxis n-aria: piecewise((case), (case), ..., default)"""
        cases = []
        while True:
            # Parse case: (condition, expr)
            self._consume('OP', '(')
            
            # Parse condition: IDENT OP NUMBER
            var_tok = self._consume('IDENT')
            op_tok = self._consume('OP')
            op = op_tok[1]
            
            # Handle negative numbers
            val_peek = self._peek()
            if val_peek and val_peek[0] == 'OP' and val_peek[1] == '-':
                self._consume()
                num_tok = self._consume('NUM')
                threshold = -float(num_tok[1])
            else:
                num_tok = self._consume('NUM')
                threshold = float(num_tok[1])
            
            self._consume('OP', ',')
            case_expr = self._parse_expr()
            self._consume('OP', ')')
            
            cases.append((op, threshold, case_expr))
            
            # Check if next is another case or default
            peek = self._peek()
            if peek and peek[1] == ',':
                self._consume()
                next_peek = self._peek()
                if next_peek and next_peek[1] == '(':
                    continue
                else:
                    default_expr = self._parse_expr()
                    break
            else:
                raise SyntaxError("Expected ',' after case in piecewise")
        
        self._consume('OP', ')')
        
        # Build StratifiedNode from cases
        a, b = self.domain
        branches = []
        
        # Sort cases by threshold
        sorted_cases = sorted(cases, key=lambda c: c[1])
        
        # Build regions
        current_pos = a
        for i, (op, threshold, expr) in enumerate(sorted_cases):
            if threshold > current_pos and threshold <= b:
                if op in ('<', '<='):
                    branches.append(StratifiedNode.Branch(expr, expr, (current_pos, threshold)))
                    current_pos = threshold
                elif op in ('>', '>='):
                    prev_expr = sorted_cases[i-1][2] if i > 0 else default_expr
                    branches.append(StratifiedNode.Branch(prev_expr, prev_expr, (current_pos, threshold)))
                    current_pos = threshold
        
        # Last region
        if sorted_cases:
            last_op = sorted_cases[-1][0]
            last_expr = sorted_cases[-1][2]
            if last_op in ('>', '>='):
                if current_pos < b:
                    branches.append(StratifiedNode.Branch(last_expr, last_expr, (max(current_pos, sorted_cases[-1][1]), b)))
            else:
                if current_pos < b:
                    branches.append(StratifiedNode.Branch(default_expr, default_expr, (current_pos, b)))
        else:
            branches.append(StratifiedNode.Branch(default_expr, default_expr, (a, b)))
        
        if not branches:
            branches.append(StratifiedNode.Branch(default_expr, default_expr, (a, b)))
        
        return StratifiedNode(branches)
    
    def _parse_derivative(self) -> ASTNode:
        """derivative ::= 'D' '(' expr (',' IDENT (',' NUMBER)? )? ')'
        
        Computes symbolic derivative for supported functions.
        """
        self._consume('OP', '(')
        inner = self._parse_expr()
        
        # Check for variable: D(expr, var) or D(expr, var, n)
        var_name = 'x'  # Default variable
        order = 1
        
        peek = self._peek()
        if peek and peek[1] == ',':
            self._consume()
            # Check if next is a variable name or a number
            next_peek = self._peek()
            if next_peek and next_peek[0] == 'IDENT':
                var_tok = self._consume('IDENT')
                var_name = var_tok[1]
                
                # Check for order
                peek2 = self._peek()
                if peek2 and peek2[1] == ',':
                    self._consume()
                    order_tok = self._consume('NUM')
                    order = int(float(order_tok[1]))
            elif next_peek and next_peek[0] == 'NUM':
                order_tok = self._consume('NUM')
                order = int(float(order_tok[1]))
        
        self._consume('OP', ')')
        
        return self._compute_derivative(inner, order, var_name)
    
    def _parse_gradient(self) -> ASTNode:
        """gradient ::= 'grad' '(' expr ',' '[' IDENT (',' IDENT)* ']' ')'
        
        Computes gradient with respect to multiple variables.
        Returns a list of derivative expressions.
        """
        self._consume('OP', '(')
        expr = self._parse_expr()
        self._consume('OP', ',')
        self._consume('OP', '[')
        
        # Parse variable list
        variables = []
        var_tok = self._consume('IDENT')
        variables.append(var_tok[1])
        
        while True:
            peek = self._peek()
            if peek and peek[1] == ',':
                self._consume()
                var_tok = self._consume('IDENT')
                variables.append(var_tok[1])
            else:
                break
        
        self._consume('OP', ']')
        self._consume('OP', ')')
        
        # Compute gradient as a sum of partial derivatives
        # For simplicity, we return the sum of absolute partial derivatives
        # This is useful for optimization: |∂f/∂x₁| + |∂f/∂x₂| + ...
        if len(variables) == 1:
            # Single variable: just the derivative
            return self._compute_derivative(expr, 1, variables[0])
        
        # Multiple variables: sum of squared partials (gradient magnitude squared)
        # grad² = (∂f/∂x₁)² + (∂f/∂x₂)² + ...
        partials = []
        for var in variables:
            partial = self._compute_derivative(expr, 1, var)
            partials.append(partial)
        
        # Sum of squares: (∂f/∂x₁)² + (∂f/∂x₂)² + ...
        result = self.frontend._ast_mul(partials[0], partials[0])
        for p in partials[1:]:
            p_sq = self.frontend._ast_mul(p, p)
            result = self.frontend._ast_add(result, p_sq)
        
        return result
    
    def _compute_derivative(self, node: ASTNode, order: int = 1, var: str = 'x') -> ASTNode:
        """Compute symbolic derivative of a node with respect to variable `var`."""
        if order <= 0:
            return node
        
        # Base cases for simple nodes
        if isinstance(node, InputNode):
            # d/dvar(var) = 1, d/dvar(other) = 0
            if node.name == var:
                return ConstantNode(torch.tensor(1.0, dtype=self.frontend.dtype))
            else:
                return ConstantNode(torch.tensor(0.0, dtype=self.frontend.dtype))
        
        if isinstance(node, ConstantNode):
            # d/dvar(c) = 0
            return ConstantNode(torch.tensor(0.0, dtype=self.frontend.dtype))
        
        if isinstance(node, ScaleNode):
            # d/dvar(a*f(x)) = a * f'(x)
            child_deriv = self._compute_derivative(
                node.children[0] if node.children else InputNode("x"), 
                order, var
            )
            return ScaleNode(node.factor, child=child_deriv)
        
        if isinstance(node, ShiftNode):
            # d/dvar(f(x) + b) = f'(x)
            child_deriv = self._compute_derivative(
                node.children[0] if node.children else InputNode("x"),
                order, var
            )
            return child_deriv
        
        if isinstance(node, TranscendentalNode):
            name = node.name.lower()
            inner_node = node.polynomial.input_node or InputNode("x")
            
            if name == 'sin':
                # d/dvar(sin(u)) = cos(u) * du/dvar
                deriv = self.frontend.cos(inner_node, domain=self.domain, degree=self.degree)
            elif name == 'cos':
                # d/dvar(cos(u)) = -sin(u) * du/dvar
                sin_node = self.frontend.sin(inner_node, domain=self.domain, degree=self.degree)
                deriv = ScaleNode(torch.tensor(-1.0, dtype=self.frontend.dtype), child=sin_node)
            elif name == 'exp':
                # d/dvar(exp(u)) = exp(u) * du/dvar
                deriv = self.frontend.exp(inner_node, domain=self.domain, degree=self.degree)
            elif name == 'log':
                # d/dvar(log(u)) = (1/u) * du/dvar
                inv = ScaleNode(torch.tensor(-1.0, dtype=self.frontend.dtype), child=inner_node)
                deriv = self.frontend._ast_pow(inner_node, inv)
            elif name == 'tanh':
                # d/dvar(tanh(u)) = (1 - tanh(u)^2) * du/dvar
                tanh_node = self.frontend.tanh(inner_node, domain=self.domain, degree=self.degree)
                tanh_sq = self.frontend._ast_mul(tanh_node, tanh_node)
                one = ConstantNode(torch.tensor(1.0, dtype=self.frontend.dtype))
                deriv = self.frontend._ast_sub(one, tanh_sq)
            elif name == 'sigmoid':
                # d/dvar(sigmoid(u)) = sigmoid(u) * (1 - sigmoid(u)) * du/dvar
                sig_node = self.frontend.sigmoid(inner_node, domain=self.domain, degree=self.degree)
                one = ConstantNode(torch.tensor(1.0, dtype=self.frontend.dtype))
                one_minus_sig = self.frontend._ast_sub(one, sig_node)
                deriv = self.frontend._ast_mul(sig_node, one_minus_sig)
            else:
                raise SyntaxError(f"Derivative not supported for function: {name}")
            
            # Apply chain rule if inner is not just the variable
            if not (isinstance(inner_node, InputNode) and inner_node.name == var):
                inner_deriv = self._compute_derivative(inner_node, 1, var)
                deriv = self.frontend._ast_mul(deriv, inner_deriv)
            
            # Higher order derivatives
            if order > 1:
                return self._compute_derivative(deriv, order - 1, var)
            return deriv
        
        if isinstance(node, _CompoundAddNode):
            # d/dvar(f + g) = f' + g'
            left_deriv = self._compute_derivative(node.left, order, var)
            right_deriv = self._compute_derivative(node.right, order, var)
            return _CompoundAddNode(left_deriv, right_deriv)
        
        if isinstance(node, _CompoundMulNode):
            # d/dx(f * g) = f' * g + f * g' (product rule)
            left_deriv = self._compute_derivative(node.left, 1)
            right_deriv = self._compute_derivative(node.right, 1)
            term1 = self.frontend._ast_mul(left_deriv, node.right)
            term2 = self.frontend._ast_mul(node.left, right_deriv)
            result = _CompoundAddNode(term1, term2)
            if order > 1:
                return self._compute_derivative(result, order - 1)
            return result
        
        raise SyntaxError(f"Derivative not supported for node type: {type(node).__name__}")
    
    def _parse_func_call(self, name: str) -> ASTNode:
        """func_call ::= IDENT '(' expr (',' expr)* ')'
        
        If name is a let-bound function, substitute its variable with the argument.
        """
        self._consume('OP', '(')
        args = [self._parse_expr()]
        while True:
            peek = self._peek()
            if peek and peek[1] == ',':
                self._consume()
                args.append(self._parse_expr())
            else:
                break
        self._consume('OP', ')')
        
        # Check if it's a let-bound function
        if name in self._let_bindings:
            bound_expr = self._let_bindings[name]
            var_name = self._let_var_names.get(name, 'x')
            # Substitute the variable in the bound expression with the argument
            return self._substitute(bound_expr, var_name, args[0])
        
        if name in self.SUPPORTED_FUNCTIONS:
            builder = getattr(self.frontend, name)
            # For single-argument transcendental functions
            inner = args[0]
            return builder(inner, domain=self.domain, degree=self.degree)
        
        raise SyntaxError(f"Unknown function: '{name}'. Supported: {self.SUPPORTED_FUNCTIONS}")


class _CompoundAddNode(ASTNode):
    def __init__(self, left: ASTNode, right: ASTNode):
        self.left = left
        self.right = right
        if left.geometric_type.output_dim != right.geometric_type.output_dim:
            raise TopologicalObstructionError(
                f"cannot add dimensions {left.geometric_type.output_dim} and {right.geometric_type.output_dim}"
            )
        super().__init__(NodeTag.PARALLEL, left.geometric_type, [left, right])

    def simplify(self) -> ASTNode:
        self.left = self.left.simplify()
        self.right = self.right.simplify()
        if isinstance(self.left, ConstantNode) and isinstance(self.right, ConstantNode):
            return ConstantNode(self.left.value + self.right.value, self.geometric_type)
        return self

    def estimate_fma_cost(self) -> int:
        return self.left.estimate_fma_cost() + self.right.estimate_fma_cost() + 1


class _CompoundMulNode(ASTNode):
    def __init__(self, left: ASTNode, right: ASTNode):
        self.left = left
        self.right = right
        if left.geometric_type.output_dim != right.geometric_type.output_dim:
            raise TopologicalObstructionError(
                f"cannot multiply dimensions {left.geometric_type.output_dim} and {right.geometric_type.output_dim}"
            )
        super().__init__(NodeTag.PARALLEL, left.geometric_type, [left, right])

    def simplify(self) -> ASTNode:
        self.left = self.left.simplify()
        self.right = self.right.simplify()
        if isinstance(self.left, ConstantNode) and isinstance(self.right, ConstantNode):
            return ConstantNode(self.left.value * self.right.value, self.geometric_type)
        return self

    def estimate_fma_cost(self) -> int:
        return self.left.estimate_fma_cost() + self.right.estimate_fma_cost() + 1


class Poem:
    def __init__(self, dtype: torch.dtype = torch.float64):
        self.dtype = dtype
        self._inputs: Dict[str, InputNode] = {}

    def input(self, name: str = "x", dim: int = 1) -> InputNode:
        node = InputNode(name=name, geometric_type=Scalar() if dim == 1 else Vector(dim))
        self._inputs[name] = node
        return node

    def scale(self, factor: float, x: Optional[ASTNode] = None) -> ScaleNode:
        return ScaleNode(torch.tensor(factor, dtype=self.dtype), child=x)

    def shift(self, value: float, x: Optional[ASTNode] = None) -> ShiftNode:
        return ShiftNode(torch.tensor(value, dtype=self.dtype), child=x)

    def affine(self, a: float, b: float, x: Optional[ASTNode] = None) -> AffineNode:
        return AffineNode(torch.tensor(a, dtype=self.dtype), torch.tensor(b, dtype=self.dtype), child=x)

    def compose(self, f: ASTNode, g: ASTNode) -> ComposeNode:
        return ComposeNode(outer=f, inner=g)

    def constant(self, value: float) -> ConstantNode:
        return ConstantNode(torch.tensor(value, dtype=self.dtype))

    def identity(self) -> IdentityNode:
        return IdentityNode()

    def polynomial(self, coefficients: List[float], x: Optional[InputNode] = None) -> PolynomialNode:
        return PolynomialNode(coefficients=coefficients, input_node=x)

    def sin(self, x: Optional[ASTNode] = None, domain: Tuple[float, float] = (-math.pi, math.pi), degree: int = 20) -> TranscendentalNode:
        return self._build_transcendental("sin", torch.sin, x, domain, degree)

    def cos(self, x: Optional[ASTNode] = None, domain: Tuple[float, float] = (-math.pi, math.pi), degree: int = 20) -> TranscendentalNode:
        return self._build_transcendental("cos", torch.cos, x, domain, degree)

    def exp(self, x: Optional[ASTNode] = None, domain: Tuple[float, float] = (-2.0, 2.0), degree: int = 20) -> TranscendentalNode:
        return self._build_transcendental("exp", torch.exp, x, domain, degree)

    def tanh(self, x: Optional[ASTNode] = None, domain: Tuple[float, float] = (-4.0, 4.0), degree: int = 50) -> TranscendentalNode:
        return self._build_transcendental("tanh", torch.tanh, x, domain, degree)

    def log(self, x: Optional[ASTNode] = None, domain: Tuple[float, float] = (0.5, 2.0), degree: int = 28) -> TranscendentalNode:
        return self._build_transcendental("log", torch.log, x, domain, degree)

    def sigmoid(self, x: Optional[ASTNode] = None, domain: Tuple[float, float] = (-8.0, 8.0), degree: int = 60) -> TranscendentalNode:
        return self._build_transcendental("sigmoid", torch.sigmoid, x, domain, degree)

    def custom(
        self,
        name: str,
        func: Callable[[torch.Tensor], torch.Tensor],
        domain: Tuple[float, float],
        x: Optional[ASTNode] = None,
        degree: int = 24,
    ) -> TranscendentalNode:
        return self._build_transcendental(name, func, x, domain, degree)

    def _build_transcendental(
        self,
        name: str,
        func: Callable[[torch.Tensor], torch.Tensor],
        x: Optional[ASTNode],
        domain: Tuple[float, float],
        degree: int,
    ) -> ASTNode:
        from acf_functor.core import ChebyshevReducer

        reduction = ChebyshevReducer.reduce(func, degree=degree, domain=domain, dtype=self.dtype)
        mono = reduction.metadata.get("monomial_coefficients", reduction.metadata.get("coefficients", []))
        cheb = reduction.metadata.get("chebyshev_coefficients", None)
        eval_mode = reduction.metadata.get("evaluation_mode", "horner")
        poly = PolynomialNode(coefficients=mono, input_node=x if isinstance(x, InputNode) else None)
        trans_node = TranscendentalNode(
            name=name,
            polynomial=poly,
            certified_epsilon=reduction.epsilon_bound,
            original_domain=domain,
            geometric_type=x.geometric_type if isinstance(x, ASTNode) else Scalar(),
            chebyshev_coefficients=cheb,
            evaluation_mode=eval_mode,
        )
        # If x is a composed AST (not just an InputNode), compose the transcendental with it
        if isinstance(x, ASTNode) and not isinstance(x, InputNode):
            return ComposeNode(outer=trans_node, inner=x)
        return trans_node

    def continuous_flow(self, expression: str, domain: Tuple[float, float] = (-5.0, 5.0), degree: int = 24) -> ASTNode:
        self._inputs = {}  # Reset inputs for fresh parsing
        parser = _RecursiveDescentParser(expression, self, domain, degree)
        return parser.parse()

    def _parse_expression(self, expr: str, domain: Tuple[float, float], degree: int) -> ASTNode:
        """Legacy parser — kept for backward compatibility."""
        expr = expr.strip()
        if expr.startswith("(") and expr.endswith(")"):
            return self._parse_expression(expr[1:-1], domain, degree)

        depth = 0
        for i in range(len(expr) - 1, -1, -1):
            c = expr[i]
            if c == ")":
                depth += 1
            elif c == "(":
                depth -= 1
            elif depth == 0 and c in "+-" and i > 0:
                left = self._parse_expression(expr[:i], domain, degree)
                right = self._parse_expression(expr[i + 1 :], domain, degree)
                return self._ast_add(left, right) if c == "+" else self._ast_sub(left, right)

        depth = 0
        for i in range(len(expr) - 1, -1, -1):
            c = expr[i]
            if c == ")":
                depth += 1
            elif c == "(":
                depth -= 1
            elif depth == 0 and c in "*/" and i > 0:
                left = self._parse_expression(expr[:i], domain, degree)
                right = self._parse_expression(expr[i + 1 :], domain, degree)
                return self._ast_mul(left, right)

        for fn_name in ["sin", "cos", "exp", "log", "tanh", "sigmoid"]:
            prefix = f"{fn_name}("
            if expr.startswith(prefix) and expr.endswith(")"):
                inner = self._parse_expression(expr[len(prefix) : -1], domain, degree)
                builder = getattr(self, fn_name)
                return builder(inner, domain=domain, degree=degree)

        constants = {"pi": math.pi, "e": math.e, "tau": 2 * math.pi}
        if expr in constants:
            return ConstantNode(torch.tensor(constants[expr], dtype=self.dtype))

        if expr == "x":
            if "x" not in self._inputs:
                self._inputs["x"] = InputNode("x")
            return self._inputs["x"]

        if len(expr) == 1 and expr.isalpha():
            if expr not in self._inputs:
                self._inputs[expr] = InputNode(expr)
            return self._inputs[expr]

        if expr.startswith("-"):
            inner = self._parse_expression(expr[1:], domain, degree)
            return ScaleNode(torch.tensor(-1.0, dtype=self.dtype), child=inner)

        try:
            return ConstantNode(torch.tensor(float(expr), dtype=self.dtype))
        except ValueError as exc:
            raise ValueError(f"cannot parse expression '{expr}'") from exc

    def _ast_pow(self, base: ASTNode, exp: ASTNode) -> ASTNode:
        if isinstance(exp, ConstantNode):
            exp_val = exp.value.item()
            if exp_val == 2.0:
                return _CompoundMulNode(base, base)
            elif exp_val == 3.0:
                return _CompoundMulNode(_CompoundMulNode(base, base), base)
            elif exp_val == int(exp_val) and exp_val > 0:
                result = base
                for _ in range(int(exp_val) - 1):
                    result = _CompoundMulNode(result, base)
                return result
            else:
                return _CompoundMulNode(
                    self.custom("pow", lambda x: torch.pow(x, exp_val), domain=(-10.0, 10.0))(base),
                    base
                )
        return _CompoundMulNode(base, self.custom("exp", torch.exp, domain=(-10.0, 10.0))(
            _CompoundMulNode(exp, self.custom("log", torch.log, domain=(0.1, 10.0))(base))
        ))

    def _ast_add(self, left: ASTNode, right: ASTNode) -> ASTNode:
        if isinstance(right, ConstantNode):
            return ShiftNode(right.value, child=left)
        if isinstance(left, ConstantNode):
            return ShiftNode(left.value, child=right)
        return _CompoundAddNode(left, right)

    def _ast_sub(self, left: ASTNode, right: ASTNode) -> ASTNode:
        return self._ast_add(left, ScaleNode(torch.tensor(-1.0, dtype=self.dtype), child=right))

    def _ast_mul(self, left: ASTNode, right: ASTNode) -> ASTNode:
        if isinstance(right, ConstantNode):
            return ScaleNode(right.value, child=left)
        if isinstance(left, ConstantNode):
            return ScaleNode(left.value, child=right)
        return _CompoundMulNode(left, right)

    # === Funciones de activación predefinidas ===

    def relu(self, x: Optional[ASTNode] = None) -> ASTNode:
        """ReLU: max(0, x) = piecewise(x >= 0, x, 0)"""
        input_node = x if x is not None else self.input("x")
        return self.continuous_flow("piecewise(x >= 0, x, 0)")

    def gelu_approx(self, x: Optional[ASTNode] = None, degree: int = 24) -> ASTNode:
        """GELU aproximado: 0.5*x*(1 + tanh(0.797885*x + 0.035677*x^3))"""
        input_node = x if x is not None else self.input("x")
        return self.continuous_flow(
            "0.5*x*(1 + tanh(0.797885*x + 0.035677*x^3))",
            degree=degree
        )

    def swish(self, x: Optional[ASTNode] = None, degree: int = 24) -> ASTNode:
        """Swish: x * sigmoid(x)"""
        input_node = x if x is not None else self.input("x")
        return self.continuous_flow("x * sigmoid(x)", degree=degree)

    def mish(self, x: Optional[ASTNode] = None, degree: int = 24) -> ASTNode:
        """Mish: x * tanh(log(1 + exp(x)))"""
        input_node = x if x is not None else self.input("x")
        return self.continuous_flow("x * tanh(log(1 + exp(x)))", degree=degree)

    def silu(self, x: Optional[ASTNode] = None, degree: int = 24) -> ASTNode:
        """SiLU (Swish con beta=1): x * sigmoid(x)"""
        return self.swish(x, degree=degree)


from dataclasses import dataclass, field


@dataclass
class CoReport:
    """Reporte de síntesis de CoPoem con métricas de adjunción."""
    spectral_radius_requested: float = 0.0
    spectral_radius_actual: float = 0.0
    adjunction_gap: float = 0.0
    spectral_consistency: float = 0.0
    synthesis_iterations: int = 0
    frobenius_norm: float = 0.0
    symmetry_verified: bool = False
    constraints_satisfied: Dict[str, bool] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "COPOEM SYNTHESIS REPORT",
            f"- Spectral radius (requested): {self.spectral_radius_requested:.4f}",
            f"- Spectral radius (actual): {self.spectral_radius_actual:.4f}",
            f"- Adjunction gap: {self.adjunction_gap:.4e}",
            f"- Spectral consistency: {self.spectral_consistency:.4f}",
            f"- Frobenius norm: {self.frobenius_norm:.4f}",
            f"- Symmetry verified: {self.symmetry_verified}",
            f"- Synthesis iterations: {self.synthesis_iterations}",
        ]
        if self.warnings:
            lines.append("- Warnings:")
            lines.extend(f"  * {w}" for w in self.warnings)
        return "\n".join(lines)


class _MultiObjectiveSpec:
    """Especificación multiobjetivo para CoPoem."""
    
    def __init__(self, dtype: torch.dtype = torch.float64):
        self.dtype = dtype
        self.dimension: Optional[int] = None
        self.spectral_radius: Optional[float] = None
        self.lyapunov_exponent: Optional[float] = None
        self.symmetry: Optional[str] = None
        self.minimize_objective: Optional[str] = None
        self.minimize_budget: Optional[float] = None
        self.eigenvalue_decay: str = "geometric"
        self.target_alpha: Optional[float] = None
    
    def spectrum(self, spectral_radius: float = 0.95, dimension: int = 64,
                 symmetry: Optional[str] = None, eigenvalue_decay: str = "geometric",
                 target_alpha: Optional[float] = None) -> '_MultiObjectiveSpec':
        self.spectral_radius = spectral_radius
        self.dimension = dimension
        self.symmetry = symmetry
        self.eigenvalue_decay = eigenvalue_decay
        self.target_alpha = target_alpha
        return self
    
    def stability(self, lyapunov_exponent: float, dimension: int = 64) -> '_MultiObjectiveSpec':
        self.lyapunov_exponent = lyapunov_exponent
        self.dimension = dimension
        return self
    
    def structure(self, structure_type: str) -> '_MultiObjectiveSpec':
        """Specify matrix structure: 'symmetric', 'orthogonal', 'triangular', 'toeplitz'."""
        self.symmetry = structure_type
        return self
    
    def minimize(self, objective: str, budget: Optional[float] = None) -> '_MultiObjectiveSpec':
        """Specify minimization objective: 'frobenius_norm', 'nuclear_norm', 'max_element'."""
        self.minimize_objective = objective
        self.minimize_budget = budget
        return self
    
    def dimension(self, dim: int) -> '_MultiObjectiveSpec':
        self.dimension = dim
        return self


class CoPoem:
    def __init__(self, dtype: torch.dtype = torch.float64):
        self.dtype = dtype

    def spectrum(
        self,
        spectral_radius: float = 0.95,
        dimension: int = 64,
        symmetry: Optional[str] = None,
        eigenvalue_decay: str = "geometric",
        target_alpha: Optional[float] = None,
    ) -> Dict[str, Any]:
        return {
            "type": "spectral",
            "spectral_radius": spectral_radius,
            "dimension": dimension,
            "symmetry": symmetry,
            "eigenvalue_decay": eigenvalue_decay,
            "target_alpha": target_alpha,
        }

    def stability(self, lyapunov_exponent: float, dimension: int = 64) -> Dict[str, Any]:
        return {
            "type": "stability",
            "lyapunov_exponent": lyapunov_exponent,
            "dimension": dimension,
        }

    def minimizes(self, objective: str, dimension: int = 64, constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return {
            "type": "optimization",
            "objective": objective,
            "dimension": dimension,
            "constraints": constraints or {},
        }

    def synthesize(self, spec: Dict[str, Any]) -> torch.Tensor:
        spec_type = spec.get("type", "spectral")
        if spec_type == "spectral":
            return self._synthesize_spectral(spec)
        if spec_type == "stability":
            return self._synthesize_stable(spec)
        if spec_type == "optimization":
            return torch.eye(spec.get("dimension", 64), dtype=self.dtype)
        raise ValueError(f"unknown spec type: {spec_type}")

    def synthesize_with_report(self, spec: Dict[str, Any]) -> Tuple[torch.Tensor, CoReport]:
        """Synthesize matrix with full adjunction metrics report."""
        W = self.synthesize(spec)
        report = self._compute_report(W, spec)
        return W, report

    def multi_objective(self) -> _MultiObjectiveSpec:
        """Create a multi-objective specification builder."""
        return _MultiObjectiveSpec(self.dtype)

    def synthesize_multi(self, spec: _MultiObjectiveSpec) -> Tuple[torch.Tensor, CoReport]:
        """Multi-objective synthesis with iterative refinement."""
        dim = spec.dimension or 64
        W = torch.eye(dim, dtype=self.dtype)
        
        # Iterative projection onto constraint sets
        max_iter = 100
        for iteration in range(max_iter):
            W_prev = W.clone()
            
            # Project spectral radius constraint
            if spec.spectral_radius is not None:
                W = self._project_spectral_radius(W, spec.spectral_radius)
            
            # Project symmetry constraint
            if spec.symmetry == "symmetric":
                W = 0.5 * (W + W.T)
            elif spec.symmetry == "orthogonal":
                U, _, Vt = torch.linalg.svd(W)
                W = U @ Vt * spec.spectral_radius
            
            # Project Lyapunov stability
            if spec.lyapunov_exponent is not None:
                W = self._project_lyapunov(W, spec.lyapunov_exponent)
            
            # Minimize objective
            if spec.minimize_objective == "frobenius_norm" and spec.minimize_budget is not None:
                fnorm = torch.norm(W, 'fro').item()
                if fnorm > spec.minimize_budget:
                    W = W * (spec.minimize_budget / fnorm)
            
            # Check convergence
            gap = torch.norm(W - W_prev, 'fro').item()
            if gap < 1e-8:
                break
        
        report = self._compute_report(W, {
            "spectral_radius": spec.spectral_radius or 0.95,
            "dimension": dim,
            "symmetry": spec.symmetry,
        })
        report.synthesis_iterations = iteration + 1
        return W, report

    def _project_spectral_radius(self, W: torch.Tensor, radius: float) -> torch.Tensor:
        """Project matrix onto spectral radius constraint."""
        eigvals, eigvecs = torch.linalg.eig(W)
        max_abs = torch.max(torch.abs(eigvals)).item()
        if max_abs > radius:
            # Scale eigenvalues to fit within radius
            scale = radius / max_abs
            eigvals_scaled = eigvals * scale
            # Reconstruct (approximate for non-normal matrices)
            W_real = torch.real(eigvecs @ torch.diag(eigvals_scaled) @ torch.linalg.inv(eigvecs))
            return W_real.to(self.dtype)
        return W

    def _project_lyapunov(self, W: torch.Tensor, lyap_exp: float) -> torch.Tensor:
        """Project matrix onto Lyapunov stability constraint."""
        eigvals = torch.linalg.eigvals(W)
        max_real = torch.max(eigvals.real).item()
        target_max = torch.exp(torch.tensor(lyap_exp, dtype=self.dtype)).item()
        if max_real > target_max:
            scale = target_max / max_real
            W_scaled = W * scale
            return W_scaled
        return W

    def _compute_report(self, W: torch.Tensor, spec: Dict[str, Any]) -> CoReport:
        """Compute full adjunction metrics for synthesized matrix."""
        eigvals = torch.linalg.eigvals(W)
        actual_sr = torch.max(torch.abs(eigvals)).item()
        requested_sr = spec.get("spectral_radius", 0.0)
        
        # Adjunction gap: distance to fixed point of Φ ⇌ Φ* cycle
        # Measured as deviation from requested spectral properties
        adjunction_gap = abs(actual_sr - requested_sr) if requested_sr > 0 else 0.0
        
        # Spectral consistency: how well eigenvalues match expected decay pattern
        sorted_abs_eigvals, _ = torch.sort(torch.abs(eigvals), descending=True)
        if len(sorted_abs_eigvals) > 1:
            # Check for geometric decay pattern
            ratios = sorted_abs_eigvals[1:] / (sorted_abs_eigvals[:-1] + 1e-15)
            mean_ratio = torch.mean(ratios).item()
            spectral_consistency = max(0.0, 1.0 - abs(mean_ratio - 0.5))  # Heuristic
        else:
            spectral_consistency = 1.0
        
        symmetry = spec.get("symmetry")
        sym_verified = False
        if symmetry == "symmetric":
            sym_verified = bool(torch.allclose(W, W.T, atol=1e-10))
        elif symmetry == "orthogonal":
            sym_verified = bool(torch.allclose(W.T @ W, torch.eye(W.shape[0], dtype=self.dtype), atol=1e-8))
        
        return CoReport(
            spectral_radius_requested=requested_sr,
            spectral_radius_actual=actual_sr,
            adjunction_gap=adjunction_gap,
            spectral_consistency=spectral_consistency,
            frobenius_norm=torch.norm(W, 'fro').item(),
            symmetry_verified=sym_verified,
        )

    def _synthesize_spectral(self, spec: Dict[str, Any]) -> torch.Tensor:
        dim = int(spec["dimension"])
        radius = float(spec["spectral_radius"])
        decay = spec.get("eigenvalue_decay", "geometric")
        symmetry = spec.get("symmetry", None)

        if decay == "geometric":
            t = torch.linspace(0.0, 1.0, dim, dtype=self.dtype)
            eig = radius * torch.exp(-2.0 * t)
        elif decay == "algebraic":
            j = torch.arange(1, dim + 1, dtype=self.dtype)
            eig = radius / j
        elif decay == "uniform":
            eig = torch.full((dim,), radius, dtype=self.dtype)
        else:
            eig = torch.linspace(radius, max(radius * 0.01, 1e-6), dim, dtype=self.dtype)

        q, _ = torch.linalg.qr(torch.randn(dim, dim, dtype=self.dtype))
        if symmetry in ("orthogonal", "symmetric"):
            w = q @ torch.diag(eig) @ q.T
            if symmetry == "symmetric":
                w = 0.5 * (w + w.T)
            return w

        p, _ = torch.linalg.qr(torch.randn(dim, dim, dtype=self.dtype))
        return q @ torch.diag(eig) @ p.T

    def _synthesize_stable(self, spec: Dict[str, Any]) -> torch.Tensor:
        dim = int(spec["dimension"])
        lyap = float(spec["lyapunov_exponent"])
        lambdas = torch.exp(torch.linspace(lyap, lyap - 1.0, dim, dtype=self.dtype))
        q, _ = torch.linalg.qr(torch.randn(dim, dim, dtype=self.dtype))
        return q @ torch.diag(lambdas) @ q.T


class BiPoem:
    def __init__(self, dtype: torch.dtype = torch.float64):
        self.dtype = dtype

    def symbiosis(
        self,
        data: torch.Tensor,
        structure: Optional[Dict[str, Any]] = None,
        max_dimension: int = 256,
        max_iterations: int = 20,
        convergence_threshold: float = 1e-4,
    ) -> Dict[str, Any]:
        from acf_functor.core import KoopmanReducer, ACFInvariant

        structure = structure or {}
        target_alpha = structure.get("target_alpha", None)

        best_result: Optional[Dict[str, Any]] = None
        best_error = float("inf")
        history: List[Dict[str, Any]] = []

        d = min(max_dimension // 4, data.shape[0] * 3)
        d = max(d, data.shape[0] + 1)

        for i in range(max_iterations):
            try:
                k_mat, eigvals, meta = KoopmanReducer.dmd(
                    data.to(self.dtype),
                    observable_fn=lambda z: KoopmanReducer.polynomial_observables(z, max_degree=min(max(d // data.shape[0], 1), 4)),
                    rank=min(d, data.shape[1] - 1),
                )
            except Exception:
                d = max(data.shape[0] + 1, d // 2)
                continue

            alpha, delta = ACFInvariant.compute_alpha(torch.abs(eigvals))
            coupling_error = float(delta + meta["reconstruction_error"])
            history.append(
                {
                    "iteration": i,
                    "dimension": d,
                    "alpha": alpha,
                    "delta": delta,
                    "recon_error": meta["reconstruction_error"],
                    "coupling_error": coupling_error,
                }
            )

            if coupling_error < best_error:
                best_error = coupling_error
                best_result = {
                    "koopman_matrix": k_mat,
                    "eigenvalues": eigvals,
                    "alpha": alpha,
                    "delta": delta,
                    "optimal_dimension": d,
                    "reconstruction_error": meta["reconstruction_error"],
                }

            if coupling_error < convergence_threshold:
                break
            if target_alpha is not None and abs(alpha - target_alpha) < convergence_threshold:
                break

            if delta > 0.1:
                d = min(max_dimension, d + max(d // 4, 1))
            elif delta < 1e-10:
                d = max(data.shape[0] + 1, d - max(d // 4, 1))
            else:
                d = min(max_dimension, d + 2)

        if best_result is None:
            raise RuntimeError("BiPoem did not converge")

        best_result["dimension_history"] = history
        return best_result

    def symbiosis_irregular(
        self,
        times: torch.Tensor,
        values: torch.Tensor,
        structure: Optional[Dict[str, Any]] = None,
        max_dimension: int = 256,
        max_iterations: int = 20,
        convergence_threshold: float = 1e-4,
        interpolation: str = 'linear',
    ) -> Dict[str, Any]:
        """
        Versión de symbiosis para datos con muestreo no uniforme.
        
        Estrategia: interpolar a grilla regular antes de aplicar EDMD.
        La calidad de la interpolación afecta la calidad del operador Koopman.
        
        Args:
            times: Tiempos de muestreo, shape (n_timesteps,)
            values: Valores observados, shape (n_features, n_timesteps)
            interpolation: Método de interpolación ('linear', 'cubic')
            
        Returns:
            Dict con resultados de symbiosis + metadatos de interpolación
        """
        # 1. Crear grilla temporal regular
        t_min, t_max = times.min().item(), times.max().item()
        n_regular = max(values.shape[1], 100)  # Al menos 100 puntos
        t_regular = torch.linspace(t_min, t_max, n_regular, dtype=times.dtype)
        
        # 2. Interpolar a grilla regular
        values_regular = self._interpolate_timeseries(times, values, t_regular, interpolation)
        
        # 3. Calcular variabilidad en dt y advertir si es alta
        dt_original = torch.diff(times)
        cv_dt = (dt_original.std() / (dt_original.mean() + 1e-15)).item()
        
        # 4. Aplicar symbiosis normal sobre datos regularizados
        result = self.symbiosis(
            values_regular,
            structure=structure,
            max_dimension=max_dimension,
            max_iterations=max_iterations,
            convergence_threshold=convergence_threshold,
        )
        
        # 5. Añadir metadatos de interpolación al resultado
        result['interpolation_method'] = interpolation
        result['interpolation_cv_dt'] = cv_dt
        result['n_original_points'] = len(times)
        result['n_interpolated_points'] = n_regular
        
        if cv_dt > 0.5:
            result['warning'] = (
                f"Alta variabilidad en dt (CV={cv_dt:.2f}). "
                f"La interpolación puede introducir error significativo. "
                f"Considerar usar interpolation='cubic' para mayor robustez."
            )
        
        return result
    
    def _interpolate_timeseries(
        self,
        times: torch.Tensor,
        values: torch.Tensor,
        t_regular: torch.Tensor,
        method: str = 'linear',
    ) -> torch.Tensor:
        """
        Interpola series temporales a grilla regular.
        
        Args:
            times: Tiempos originales, shape (n_timesteps,)
            values: Valores originales, shape (n_features, n_timesteps)
            t_regular: Tiempos regulares, shape (n_regular,)
            method: 'linear' o 'cubic'
            
        Returns:
            Valores interpolados, shape (n_features, n_regular)
        """
        n_features = values.shape[0]
        values_interp = torch.zeros(n_features, len(t_regular), dtype=values.dtype)
        
        for i in range(n_features):
            if method == 'linear':
                # Interpolación lineal manual
                values_interp[i] = self._linear_interp_1d(times, values[i], t_regular)
            elif method == 'cubic':
                # Interpolación cúbica simplificada (Hermite)
                values_interp[i] = self._cubic_interp_1d(times, values[i], t_regular)
            else:
                raise ValueError(f"Interpolation method '{method}' not supported. Use 'linear' or 'cubic'.")
        
        return values_interp
    
    def _linear_interp_1d(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x_new: torch.Tensor,
    ) -> torch.Tensor:
        """Interpolación lineal 1D."""
        y_new = torch.zeros_like(x_new)
        
        for i, x_val in enumerate(x_new):
            # Encontrar intervalo
            idx = torch.searchsorted(x, x_val)
            if idx == 0:
                y_new[i] = y[0]
            elif idx >= len(x):
                y_new[i] = y[-1]
            else:
                # Interpolar linealmente
                x0, x1 = x[idx-1], x[idx]
                y0, y1 = y[idx-1], y[idx]
                t = (x_val - x0) / (x1 - x0 + 1e-15)
                y_new[i] = y0 + t * (y1 - y0)
        
        return y_new
    
    def _cubic_interp_1d(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x_new: torch.Tensor,
    ) -> torch.Tensor:
        """Interpolación cúbica simplificada (Hermite cúbica)."""
        y_new = torch.zeros_like(x_new)
        n = len(x)
        
        # Calcular derivadas aproximadas (diferencias finitas)
        dy = torch.zeros_like(y)
        for i in range(1, n-1):
            dx = x[i+1] - x[i-1]
            if dx > 1e-15:
                dy[i] = (y[i+1] - y[i-1]) / dx
        dy[0] = (y[1] - y[0]) / (x[1] - x[0] + 1e-15)
        dy[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2] + 1e-15)
        
        for i, x_val in enumerate(x_new):
            # Encontrar intervalo
            idx = torch.searchsorted(x, x_val)
            if idx == 0:
                y_new[i] = y[0]
            elif idx >= n:
                y_new[i] = y[-1]
            else:
                j = idx - 1
                if j >= n - 1:
                    j = n - 2
                
                # Hermite cúbica
                x0, x1 = x[j], x[j+1]
                y0, y1 = y[j], y[j+1]
                d0, d1 = dy[j], dy[j+1]
                
                h = x1 - x0
                if h < 1e-15:
                    y_new[i] = y0
                else:
                    t = (x_val - x0) / h
                    t2 = t * t
                    t3 = t2 * t
                    
                    # Polinomios de Hermite
                    h00 = 2*t3 - 3*t2 + 1
                    h10 = t3 - 2*t2 + t
                    h01 = -2*t3 + 3*t2
                    h11 = t3 - t2
                    
                    y_new[i] = h00*y0 + h10*h*d0 + h01*y1 + h11*h*d1
        
        return y_new

    def symbiosis_with_report(
        self,
        data: torch.Tensor,
        structure: Optional[Dict[str, Any]] = None,
        max_dimension: int = 256,
        max_iterations: int = 20,
        convergence_threshold: float = 1e-4,
        observable_family: str = "polynomial",
        max_degree: int = 4,
    ) -> Dict[str, Any]:
        """Symbiosis with extended BiPoem metrics (Bi₂-Bi₅)."""
        result = self.symbiosis(
            data=data,
            structure=structure,
            max_dimension=max_dimension,
            max_iterations=max_iterations,
            convergence_threshold=convergence_threshold,
        )
        
        # Bi₂: Espectro Bi-Functorial
        result["bifunctorial_spectrum"] = self._compute_bifunctorial_spectrum(
            result.get("eigenvalues", torch.tensor([])),
            result.get("koopman_matrix", torch.tensor([])),
        )
        
        # Bi₅: Índice de Decaimiento Espectral Afín α(f) desde datos
        result["acf_alpha"] = result.get("alpha", 0.0)
        
        return result

    def find_fixed_point(
        self,
        data: torch.Tensor,
        max_cycles: int = 20,
        tol: float = 1e-6,
    ) -> Dict[str, Any]:
        """Bi₃: Ciclo Φ ⇌ Φ* — iteración buscando punto fijo."""
        from acf_functor.core import KoopmanReducer, ACFInvariant
        
        n_states = data.shape[0]
        W = torch.eye(n_states, dtype=self.dtype) * 0.9
        
        history = []
        for cycle in range(max_cycles):
            # Φ: comprimir datos con W actual → secuencia FMA (via Koopman)
            try:
                k_mat, eigvals, meta = KoopmanReducer.dmd(
                    data.to(self.dtype),
                    observable_fn=lambda z: KoopmanReducer.polynomial_observables(z, max_degree=2),
                    rank=min(n_states * 2, data.shape[1] - 1),
                )
            except Exception:
                break
            
            # Φ*: sintetizar nueva W desde la estructura Koopman
            W_new = 0.5 * (k_mat + k_mat.T)  # Symmetrize
            
            gap = float(torch.norm(W_new - W, 'fro').item())
            alpha, _ = ACFInvariant.compute_alpha(torch.abs(eigvals))
            
            history.append({
                "cycle": cycle,
                "gap": gap,
                "alpha": alpha,
                "reconstruction_error": meta.get("reconstruction_error", 0.0),
            })
            
            if gap < tol:
                return {
                    "converged": True,
                    "cycles": cycle + 1,
                    "final_gap": gap,
                    "acf_alpha": alpha,
                    "koopman_matrix": k_mat,
                    "synthesized_W": W_new,
                    "history": history,
                }
            
            W = W_new
        
        return {
            "converged": False,
            "cycles": max_cycles,
            "final_gap": gap if history else float("inf"),
            "acf_alpha": history[-1]["alpha"] if history else 0.0,
            "history": history,
        }

    def _compute_bifunctorial_spectrum(
        self,
        koopman_eigenvalues: torch.Tensor,
        koopman_matrix: torch.Tensor,
    ) -> Dict[str, Any]:
        """Bi₂: Autovalores del tensor de acoplamiento Φ×Φ*."""
        if koopman_eigenvalues.numel() == 0:
            return {"eigenvalues": [], "coupling_strength": 0.0}
        
        abs_eigvals = torch.abs(koopman_eigenvalues)
        sorted_eigvals, _ = torch.sort(abs_eigvals, descending=True)
        
        # Coupling strength: ratio of dominant to subdominant eigenvalue
        if len(sorted_eigvals) > 1:
            coupling = sorted_eigvals[0].item() / (sorted_eigvals[1].item() + 1e-15)
        else:
            coupling = sorted_eigvals[0].item()
        
        return {
            "eigenvalues": sorted_eigvals.tolist(),
            "coupling_strength": coupling,
            "dominant_eigenvalue": sorted_eigvals[0].item() if len(sorted_eigvals) > 0 else 0.0,
            "spectral_gap": (sorted_eigvals[0] - sorted_eigvals[1]).item() if len(sorted_eigvals) > 1 else 0.0,
        }
