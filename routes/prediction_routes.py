from flask import Blueprint
from controllers.prediction_controller import PredictionController

prediction_bp = Blueprint("prediction_bp", __name__)

@prediction_bp.route("/predict", methods=["POST"])
def predict():
    return PredictionController.predict()
