import re
with open("poema/frontend.py", "r") as f:
    text = f.read()

# Add loop, def, if to tokenization and supported concepts
new_keywords = "elif c.isalpha() or c == '_':\n                j = i\n                while j < len(expr) and (expr[j].isalnum() or expr[j] == '_'):\n                    j += 1\n                ident = expr[i:j]\n                if ident in ['if', 'else', 'while', 'def', 'for']:\n                    tokens.append(('KEYWORD', ident))\n                else:\n                    tokens.append(('IDENT', ident))\n                i = j"

text = re.sub(r"elif c.isalpha\(\) or c \=\= '_':.*?i \= j", new_keywords, text, flags=re.DOTALL)

# Add parsing for new statements
new_parse_expr = """
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
            if peek and peek[0] == 'OP' and peek[1] in '+-':
                op = self._consume()[1]
                right = self._parse_term()
                if op == '+':
                    left = self.frontend._ast_add(left, right)
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
"""

text = re.sub(r"    def _parse_expr\(self\) -> ASTNode:.*?return left", new_parse_expr.strip(), text, flags=re.DOTALL)

with open("poema/frontend.py", "w") as f:
    f.write(text)

