from flask import Flask
from routes.prediction_routes import prediction_bp

app = Flask(__name__)

# register routes
app.register_blueprint(prediction_bp, url_prefix="/api")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
