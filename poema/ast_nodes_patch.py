import re
with open("poema/ast_nodes.py", "r") as f:
    text = f.read()

new_nodes = """
class LoopNode(ASTNode):
    def __init__(self, init: ASTNode, cond: ASTNode, body: ASTNode):
        super().__init__("loop", [init, cond, body])
        self.init = init
        self.cond = cond
        self.body = body
        
    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        # Evaluate iteratively
        val = self.init.evaluate(x)
        max_iters = 1000
        for _ in range(max_iters):
            if torch.all(self.cond.evaluate(val) <= 0):
                break
            val = self.body.evaluate(val)
        return val

class DefNode(ASTNode):
    def __init__(self, name: str, args: list, body: ASTNode):
        super().__init__("def", [body])
        self.name = name
        self.args = args
        self.body = body
        
    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        # Directly evaluate body since args mapping is handled by parser/let bindings
        return self.body.evaluate(x)

@dataclass
"""
text = text.replace("@dataclass\nclass FMAInstruction:", new_nodes + "class FMAInstruction:")

with open("poema/ast_nodes.py", "w") as f:
    f.write(text)
