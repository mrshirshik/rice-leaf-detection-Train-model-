from flask import Flask, request, render_template, send_file
from ultralytics import YOLO
import os

app = Flask(__name__)
model = YOLO("best.pt")

UPLOAD_FOLDER = "uploads"
PREDICT_FOLDER = "runs/detect/predict"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        filename: str = file.filename
        file_path: str = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        # Clear previous predictions to avoid conflict
        os.system("rm -rf runs/detect/predict")

        # Run detection with lowered confidence threshold
        results = model(file_path, save=True, conf=0.1)

        return render_template("index.html", result=True, filename=filename)

    return render_template("index.html", result=False)

@app.route("/result/<filename>")
def result(filename):
    path = os.path.join(PREDICT_FOLDER, filename)
    return send_file(path, mimetype="image/jpeg")

if __name__ == "__main__":
    app.run(debug=True)
