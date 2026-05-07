import requests
import base64
import cv2

API_KEY = "ADEO1giAIKoEYpkAWMFW"
MODEL_ID = "manta-detector/3"


def run_inference(frame, api_key, model_id):
    """Call Roboflow hosted API directly."""
    _, buffer = cv2.imencode('.jpg', frame)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    model, version = model_id.split('/')
    url = f"https://detect.roboflow.com/{model}/{version}"

    response = requests.post(
        url,
        params={"api_key": api_key},
        data=img_base64,
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    response.raise_for_status()
    return response.json()


def detect_objects(frame, confidence=40, overlap=30):
    """
    Run object detection on a frame using Roboflow Inference SDK.
    Returns list of detections: [{'class': str, 'confidence': float, 'x': int, 'y': int, 'width': int, 'height': int}]
    """
    try:
        result = run_inference(frame, API_KEY, MODEL_ID)
        detections = []
        for pred in result.get('predictions', []):
            detections.append({
                'class': pred.get('class'),
                'confidence': pred.get('confidence'),
                'x': pred.get('x'),
                'y': pred.get('y'),
                'width': pred.get('width'),
                'height': pred.get('height')
            })
        return detections
    except Exception as e:
        print(f"Detection error: {e}")
        return []

def get_centroid(detection):
    """
    Calculate centroid from detection bounding box.
    """
    x, y, w, h = detection['x'], detection['y'], detection['width'], detection['height']
    return (int(x), int(y))  # Center of bbox

def filter_detections(detections, classes=['manta', 'boat'], min_conf=0.5):
    """
    Filter detections by class and confidence.
    """
    return [d for d in detections if d['class'] in classes and d['confidence'] >= min_conf]