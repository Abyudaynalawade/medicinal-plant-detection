from flask import Flask, request, jsonify
from plant_model import predict

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return {"status": "Medicinal Plant Detection API is running"}

@app.route("/predict", methods=["POST"])
def predict_route():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files["file"]
    temp_path = "temp.jpg"
    file.save(temp_path)

    label, confidence = predict(temp_path)
    return jsonify({"prediction": label, "confidence": confidence})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
