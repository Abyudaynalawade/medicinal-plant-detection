from flask import Flask, render_template, request
import os
from plant_model import predict

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'

        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        prediction, confidence = predict(filepath)

        return render_template('index.html', filename=file.filename, prediction=prediction, confidence=confidence)

    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return os.path.join(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
