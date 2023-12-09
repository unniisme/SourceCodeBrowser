import ast
import os
# This code is specific to networkx. probably will reorganise later


def extract_functions_docstrings(file_path):
    with open(file_path, 'r') as file:
        tree = ast.parse(file.read())

    functions = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            docstring = ast.get_docstring(node)
            if not docstring:
                # If no docstring, just ignore it
                continue

            # Get rid of notes, see alsos and references.
            # They (usually) don't provide any information that is inferable from the function
            docstring = docstring.split("Notes\n---")[0].split("See also\n---")[0].split("References\n---")[0]

            node.body.pop(0)
            func_code = ast.unparse(node).strip()
            functions.append({'docstring': docstring, 'code': func_code})

            ## 
    
    return functions

def GenerateTrainingSet(directory):
    all_functions = []

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):  # Check for Python files
                file_path = os.path.join(root, file)
                functions = extract_functions_docstrings(file_path)
                all_functions.extend(functions)

    return all_functions


if __name__ == "__main__":
    import json
    import sys

    with open(f"./config/{sys.argv[1]}.json") as f:
        CONFIG = json.load(f)

    dirs = os.listdir(CONFIG["SourceCodePath"])

    for folder in dirs:
        functions = GenerateTrainingSet(f"{CONFIG["SourceCodePath"]}/{folder}")
        if functions:
            with open(f"{CONFIG["DataStorePath"]}/{folder}.json", "w+") as f:
                json.dump(functions, f, indent=1)
