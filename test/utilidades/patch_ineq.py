import re
with open("poema/frontend.py", "r") as f:
    text = f.read()

new_logic = """
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
"""

text = re.sub(r"    def _parse_expr\(self\) -> ASTNode:.*?return left", new_logic.strip(), text, flags=re.DOTALL)

with open("poema/frontend.py", "w") as f:
    f.write(text)
