from flask import Flask, request, jsonify
from ultralytics import YOLO
import numpy as np
import cv2

app = Flask(__name__)

# Load trained YOLOv8 model
model = YOLO("runs/detect/train2/weights/best.pt")  # Adjust path if needed

# Define class names
class_names = ["window", "door"]

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

    # Build detection results
    detections = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        w = float(x2 - x1)
        h = float(y2 - y1)
        detection = {
            "label": class_names[int(box.cls[0])],
            "confidence": round(float(box.conf[0]), 2),
            "bbox": [round(float(x1), 2), round(float(y1), 2), round(w, 2), round(h, 2)]
        }
        detections.append(detection)

    return jsonify({"detections": detections})

if __name__ == '__main__':
    app.run(debug=True)
