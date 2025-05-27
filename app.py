from flask import Flask, request, send_file, jsonify
from ultralytics import YOLO
import io
import cv2
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load your trained YOLOv8 model
model = YOLO("runs/detect/train2/weights/best.pt")  # Use correct path

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    img_bytes = file.read()

    # Convert image bytes to numpy array
    npimg = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Run prediction
    results = model.predict(img, conf=0.3)

    # Render bounding boxes on image
    annotated_img = results[0].plot()

    # Convert OpenCV image (BGR) to JPEG bytes
    _, buffer = cv2.imencode('.jpg', annotated_img)
    io_buf = io.BytesIO(buffer.tobytes())

    return send_file(io_buf, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)
