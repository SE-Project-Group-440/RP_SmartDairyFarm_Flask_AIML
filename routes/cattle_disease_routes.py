from flask import Blueprint
from controllers.cattle_disease_controller import predict_cattle_disease

cattle_disease_bp = Blueprint("cattle_disease", __name__)

cattle_disease_bp.route(
    "/predict/cattle-disease",
    methods=["POST"]
)(predict_cattle_disease)
