from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
from emotion_detector import EmotionDetector
import json
from werkzeug.utils import secure_filename
import os
from PIL import Image

app = Flask(__name__)
detector = EmotionDetector()

# Global variable to store current emotion
current_emotion = {"emotion": "Neutral", "confidence": 0.0}

# Set the upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def generate_frames():
    """Generate frames from webcam with emotion detection."""
    cap = cv2.VideoCapture(0)
    
    while True:
        success, frame = cap.read()
        if not success:
            break
            
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect emotion
        result = detector.detect_emotion(frame_rgb)
        
        # Update global emotion state
        global current_emotion
        current_emotion = result
        
        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_emotion')
def get_emotion():
    """Get current emotion data."""
    return jsonify(current_emotion)

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Call the emotion detection function
        result = detector.detect_emotion(Image.open(file_path))
        
        return jsonify({"message": "File uploaded successfully", "emotion": result['emotion'], "confidence": result['confidence']}), 200
    else:
        return jsonify({"error": "File type not allowed"}), 400

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True) 