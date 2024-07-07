import os
from yolov5 import YOLOv5

yolo = YOLOv5("yolov5s.pt", device="cpu")


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_on_image(image_stream):
    image = cv2.imdecode(np.asarray(bytearray(image_stream.read()), dtype=np.uint8), cv2.IMREAD_COLOR)
    results = model.predict(image, classes=0, conf=0.5)
    in_bgr = results.plot(conf=False)
    for i, r in enumerate(results):
        pass
    return in_bgr

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')

        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error='No selected file')

        if file and allowed_file(file.filename):
            predicted_image = predict_on_image(file.stream)

            retval, buffer = cv2.imencode('.png', predicted_image)
            detection_img_base64 = base64.b64encode(buffer).decode('utf-8')

            file.stream.seek(0)
            original_img_base64 = base64.b64encode(file.stream.read()).decode('utf-8')

            return render_template('result.html', original_img_data=original_img_base64, detection_img_data=detection_img_base64)
    return render_template('index.html')

if __name__ == '__main__':
    os.environ.setdefault('FLASK_ENV', 'development')
    app.run(debug=True, port=5000, host='0.0.0.0')
