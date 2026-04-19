import re
with open("poema/ast_nodes.py", "r") as f:
    text = f.read()

repl = """
    def simplify(self) -> 'ASTNode': return self
    def estimate_fma_cost(self) -> int: return 1

class LoopNode"""

text = text.replace("class LoopNode", repl)

repl2 = """
    def simplify(self) -> 'ASTNode': return self
    def estimate_fma_cost(self) -> int: return 10

class DefNode"""
text = text.replace("class DefNode", repl2)

repl3 = """
    def simplify(self) -> 'ASTNode': return self
    def estimate_fma_cost(self) -> int: return 1

@dataclass"""
text = text.replace("@dataclass", repl3)

with open("poema/ast_nodes.py", "w") as f:
    f.write(text)
