from flask import Flask, request, jsonify
from flask_cors import CORS
from detector import predict_clickbait
import os
app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        url = data.get("url")

        if not url:
            return jsonify({"error": "No URL provided"}), 400

        result = predict_clickbait(url)
        return jsonify({"result": result}), 200

    except Exception as e:
        print("🔥 BACKEND ERROR:", e)
        return jsonify({
            "error": "Backend processing failed",
            "details": str(e)
        }), 500

@app.route("/")
def home():
    return "Backend is running 🚀"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
