from flask import Flask, request
from flask_restful import reqparse, abort, Api, Resource
import pickle
import pandas, json
from pyscripts import TextPreprocessor

app = Flask(__name__)
api = Api(app)

# argument parsing
parser = reqparse.RequestParser()
parser.add_argument('data', required=True)
parser.add_argument('parameters', required=True)

class BCML(Resource):
    def post(self):
        # use parser and find the user's query
        #args = parser.parse_args()
        call = request.json
        datadf = pandas.DataFrame(call['data'])
        params = call['parameters']

        if params['product'] == 'text':
            outdf = TextPreprocessor.TextTransformer(purpose=params['purpose']).transform(datadf)
            return outdf.to_json(orient='records')

api.add_resource(BCML, '/')


if __name__ == '__main__':
    app.run()
