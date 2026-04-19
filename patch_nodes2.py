import re
with open("poema/ast_nodes.py", "r") as f:
    text = f.read()

# Replace super().__init__("...") with super().__init__(NodeTag.STRATIFIED, Scalar())
text = text.replace('super().__init__("piecewise")', "from .ast_nodes import NodeTag, Scalar\n        super().__init__(NodeTag.STRATIFIED, Scalar())")
text = text.replace('super().__init__("loop")', "super().__init__(NodeTag.SEQUENCE, Scalar())")
text = text.replace('super().__init__("def")', "super().__init__(NodeTag.FEEDBACK, Scalar())")

with open("poema/ast_nodes.py", "w") as f:
    f.write(text)
