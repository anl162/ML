import sys
from pathlib import Path

# ensure project root is on sys.path so ai_detector (in ../) can be imported
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
from werkzeug.utils import secure_filename
app = Flask(__name__)
UPLOAD_FOLDER = str(Path(__file__).parent / 'uploads')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/")
def homepage():
   return render_template('index.html')

@app.route("/upload", methods = ["POST"])
def upload_photo():
   file = request.files.get('image_file')
   if not file or file.filename == "":
      return "No file uploaded", 400
   filename = secure_filename(file.filename)
   save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
   file.save(save_path)
   
   return redirect(url_for('predict', filename=filename))   # use secured filename


@app.route("/predict", methods=["GET", "POST"])
def predict():
    filename = request.args.get('filename') or request.form.get('image_path')
    if not filename:
        return jsonify({"error": "no filename provided"}), 400

    img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(img_path):
        return jsonify({"error": "file not found", "path": img_path}), 404

    # import after sys.path fix above
    try:
        from ai_detector import predict_image
    except Exception as e:
        return jsonify({"error": f"import ai_detector failed: {e}"}), 500

    result = predict_image(img_path)
    if isinstance(result, int):
        predicted_class = int(result)
    else:
        txt = str(result).lower()
        predicted_class = 1 if "ai" in txt or "ai-generated" in txt or "fake" in txt else 0

    return redirect(url_for('result', filename=filename, prediction=predicted_class, raw=str(result)))

@app.route("/result")
def result(): 
   filename = request.args.get('filename')
   prediction = request.args.get('prediction')
   raw = request.args.get('raw', '')
   if not filename or prediction is None:
       return "Missing parameters", 400
   try:
       prediction = int(prediction)
   except ValueError:
       return "Invalid prediction value", 400
   img_url = url_for('static', filename=f'uploads/{filename}')
   return render_template('result.html', img_url=img_url, prediction=prediction, raw=raw)


if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(host='0.0.0.0', port=3000)
