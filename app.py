from flask import Flask, request
import pickle
import pandas, json
from pyscripts import TextPreprocessor
import os

os.system("python -m spacy download en_core_web_lg")
os.system("python -m textblob.download_corpora")

app = Flask(__name__)

@app.route('/', methods=['POST'])
def bcml(self):
    if request.method == 'POST':
        datadf = pandas.DataFrame(request.form['data'])
        params = request.form['parameters']
    else:
        "InvalidRequestMethod. API only accepts POST requests"
    if params['product'] == 'text':
        outdf = TextPreprocessor.TextTransformer(purpose=params['purpose']).transform(datadf)
        return outdf.to_json(orient='records')


if __name__ == '__main__':
    app.run()
