# !python3
import toml
import ast
def find_vers(node, new_vers):

    node.body[-1].value.args[0].value=new_vers
if __name__ == "__main__":
    with open("pyproject.toml") as f:
        data=toml.load(f)
        with open("mmcore/__init__.py") as fl:

            astModule=ast.parse(fl.read())
        find_vers(astModule, data['tool']['poetry']['version'])
        with open("mmcore/__init__.py",'w') as fll:
            fll.write( ast.unparse(astModule))
