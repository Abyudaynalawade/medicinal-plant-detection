import os
from flask import Flask, request, render_template
from plant_model import predict

app = Flask(__name__)

# Folder for uploaded files
UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", prediction="No file uploaded", confidence=0, filename="")
        
        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", prediction="No file selected", confidence=0, filename="")
        
        # Save file
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        # Predict
        label, confidence = predict(filepath)

        return render_template(
            "index.html",
            prediction=label,
            confidence=confidence,
            filename=file.filename
        )

    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
