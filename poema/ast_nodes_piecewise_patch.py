import re
with open("poema/ast_nodes.py", "r") as f:
    text = f.read()

new_nodes = """
class PiecewiseNode(ASTNode):
    def __init__(self, cond: ASTNode, true_expr: ASTNode, false_expr: ASTNode):
        super().__init__("piecewise", [cond, true_expr, false_expr])
        self.cond = cond
        self.true_expr = true_expr
        self.false_expr = false_expr
        
    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        # A simple piecewise evaluation: cond > 0
        cond_val = self.cond.evaluate(x)
        mask = cond_val > 0
        res = torch.empty_like(cond_val, dtype=x.dtype)
        
        if mask.any():
            res[mask] = self.true_expr.evaluate(x)[mask]
        if (~mask).any():
            res[~mask] = self.false_expr.evaluate(x)[~mask]
        return res

"""
text = text.replace("class LoopNode(ASTNode):", new_nodes + "class LoopNode(ASTNode):")

with open("poema/ast_nodes.py", "w") as f:
    f.write(text)
