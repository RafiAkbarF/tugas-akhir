import os
import io
import torch
import cv2
import easyocr
from flask import Flask, request, jsonify, render_template
from PIL import Image
import base64
import numpy as np
import requests
from yolov5 import YOLOv5
import pathlib

# Fix for YOLOv5 path issue
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

app = Flask(__name__)

# Load the YOLO model
weights_path = "best(6).pt"
model = YOLOv5(weights_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.model.to(device)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def plot_detections(image_np, detections):
    for (x1, y1, x2, y2, conf, cls) in detections:
        cv2.rectangle(image_np, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    return image_np

def predict_on_image(image_stream):
    image = Image.open(image_stream).convert("RGB")
    image_np = np.array(image)
    if image_np is None or image_np.size == 0:
        raise ValueError("Error: Image is empty or not loaded correctly.")
    results = model.predict(image_np)
    detections = results.xyxy[0].cpu().numpy()
    img_with_boxes = plot_detections(image_np, detections)
    return img_with_boxes

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl1 = clahe.apply(gray)
    _, binary = cv2.threshold(cl1, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary

def extract_information_from_vin(image_path):
    image = preprocess_image(image_path)
    reader = easyocr.Reader(['id'])
    results = reader.readtext(image)
    combined_text = " ".join([text for (_, text, _) in results])
    for (bbox, text, prob) in results:
        print(f"{text} (Kepercayaan: {prob:.2f})")
    return combined_text, image

@app.route('/', methods=['GET'])
def deteksiYolo():
    if request.method == 'GET':
        image_url = request.args.get('image_url')
        if not image_url:
            return render_template('index.html', error='No image URL provided')

        try:
            resp = requests.get(image_url)
            if resp.status_code != 200:
                return render_template('index.html', error='Unable to fetch image from the provided URL')
            image_stream = io.BytesIO(resp.content)
            if allowed_file(image_url):
                predicted_image = predict_on_image(image_stream)
                if predicted_image is None or predicted_image.size == 0:
                    return render_template('index.html', error='Prediction failed or image is empty after prediction')
                predicted_image_pil = Image.fromarray(predicted_image)
                buffer = io.BytesIO()
                predicted_image_pil.save(buffer, format="PNG")
                detection_img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                image_stream.seek(0)
                original_img_base64 = base64.b64encode(image_stream.read()).decode('utf-8')
                return render_template('result.html', original_img_data=original_img_base64, detection_img_data=detection_img_base64)
            else:
                return render_template('index.html', error='File type is not allowed')
        except Exception as e:
            return render_template('index.html', error=f'Error processing the image: {str(e)}')

    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    print("Endpoint /predict dipanggil")
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and allowed_file(file.filename):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        extracted_text, preprocessed_image = extract_information_from_vin(file_path)
        _, buffer = cv2.imencode('.png', preprocessed_image)
        preprocessed_img_base64 = base64.b64encode(buffer).decode('utf-8')
        return jsonify({
                'extracted_text': extracted_text,
                'preprocessed_img_data': preprocessed_img_base64
            })

    return jsonify({'error': 'File type is not allowed'})

if __name__ == '__main__':
    app.run(debug=True)
