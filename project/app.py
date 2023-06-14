from flask import Flask, jsonify, Blueprint, request
from flask_restx import Api, Resource, fields, fields, abort
from .model import model
import numpy as np

app_ins = Flask(__name__)

blueprint = Blueprint('api', __name__, url_prefix='/api/ml-model')
app = Api(blueprint, default="API", default_label="ML Model API")

app_ins.register_blueprint(blueprint)

carbon_emission_fields = app.model('Total Emission', {
    'total_emission': fields.Integer,
})
@app.route("/carbon_emission_comparison")
class MyResource(Resource):
    @app.expect(carbon_emission_fields, validate=True)
    def post(self):
        try:
            total_emission = float(request.json['total_emission'])
            prediction = model.predict(np.array([[total_emission]]))
            
            predicted_class = prediction.argmax()
            return jsonify(result=int(predicted_class))
        except ValueError:
            abort(400, "Invalid total_emission value")
        except Exception as e:
            abort(400, str(e))
