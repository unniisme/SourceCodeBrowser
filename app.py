from flask import Flask, render_template, request
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter
# from model.model import get_model
from model.T5 import *
import logging
import sys

FORMAT = "[%(levelname)-8s][%(asctime)s][%(filename)s:%(lineno)s - %(funcName)13s()] %(message)s"
logging.basicConfig(format=FORMAT, stream=sys.stdout, encoding='utf-8', level=logging.DEBUG)

app = Flask(__name__)

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
elif sys.argv == "T5+":
    model = T5_plus()
else:
    model = T5Model_Pretrained(sys.argv[1])



@app.route('/', methods=['GET', 'POST'])
def index():
    if 'code' in request.form:
        result = model.predict(request.form['code'])
        # result = highlight(result, PythonLexer(), HtmlFormatter())
        return render_template('index.html', result=result, submitted_code = request.form['code'])


    return render_template('index.html')

@app.route('/highlight', methods=['POST'])
def highlight_code():
    code = request.form['code']

    # Perform syntax highlighting using Pygments
    highlighted_code = highlight(code, PythonLexer(), HtmlFormatter())

    return highlighted_code

if __name__ == '__main__':
    app.run(debug=True)
