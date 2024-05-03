from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import TerminalFormatter
from model.T5 import *
import logging
import sys

FORMAT = "[%(levelname)-8s][%(asctime)s][%(filename)s:%(lineno)s - %(funcName)13s()] %(message)s"
logging.basicConfig(format=FORMAT, stream=sys.stdout, encoding='utf-8', level=logging.INFO)


if len(sys.argv) == 1:
    print("Possible arguments: \
        \t T5 : loads the base codeT5 summarization model (default)\
        \t T5_python: loads the codeT5 python summarization model\
        \t T5+ : loads the codeT5+ python summarization model \
        \t <model_path> : loads the model at the given directory")
    sys.argv.append("T5")

if sys.argv[1] == "T5":
    model = T5Model()
elif sys.argv[1] == "T5_python":
    model = T5Model("python")
elif sys.argv[1] == "T5+":
    model = T5_plus()
else:
    model = T5Model_Pretrained(sys.argv[1])


incode = sys.stdin.read()
print()
print(highlight(incode, PythonLexer(), TerminalFormatter(bg="dark")))
result = model.predict("Summarize python: " + incode)

print()
print()
print('\"\"\"')
print(result)
print('\"\"\"')
