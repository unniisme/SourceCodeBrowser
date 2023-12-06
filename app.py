from flask import Flask, render_template, request
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter
from model.model import get_model

app = Flask(__name__)
model = get_model()

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
