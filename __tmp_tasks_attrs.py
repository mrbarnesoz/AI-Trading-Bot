import ast
from pathlib import Path
code = Path('tradingbotui/routes.py').read_text()
mod = ast.parse(code)
names=set()
class Visitor(ast.NodeVisitor):
    def visit_Attribute(self,node):
        if isinstance(node.value, ast.Name) and node.value.id=='tasks':
            names.add(node.attr)
        self.generic_visit(node)
Visitor().visit(mod)
print(sorted(names))
