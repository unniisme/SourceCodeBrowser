from flask import Flask, render_template, request
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter
# from model.model import get_model
from model.T5 import T5Model
import logging
import sys

FORMAT = "[%(levelname)-8s][%(asctime)s][%(filename)s:%(lineno)s - %(funcName)13s()] %(message)s"
logging.basicConfig(format=FORMAT, stream=sys.stdout, encoding='utf-8', level=logging.DEBUG)

app = Flask(__name__)

# model = get_model("T5")
model = T5Model()
model.model.save_pretrained("pretrained")

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
