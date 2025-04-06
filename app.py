from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
from emotion_detector import EmotionDetector
import json
from werkzeug.utils import secure_filename
import os
from PIL import Image
import time
import io

app = Flask(__name__)
detector = EmotionDetector()

# Global variable to store current emotion
current_emotion = {"emotion": "Neutral", "confidence": 0.0}

# Set the allowed extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def generate_frames():
    """Generate frames from webcam with emotion detection."""
    cap = cv2.VideoCapture(0)
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Reduced resolution
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)  # Set to 30 FPS
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer size
    
    # Initialize variables for frame skipping
    frame_count = 0
    last_emotion_time = time.time()
    emotion_update_interval = 0.5  # Update emotion every 0.5 seconds
    
    while True:
        success, frame = cap.read()
        if not success:
            break
            
        # Skip frames for emotion detection to improve performance
        current_time = time.time()
        if current_time - last_emotion_time >= emotion_update_interval:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect emotion
            result = detector.detect_emotion(frame_rgb)
            
            # Update global emotion state
            global current_emotion
            current_emotion = result
            
            last_emotion_time = current_time
        
        # Optimize JPEG encoding
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]  # Slightly reduced quality for better performance
        ret, buffer = cv2.imencode('.jpg', frame, encode_param)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        # Small delay to prevent overwhelming the system
        time.sleep(0.01)

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
        # Read the image directly from memory
        image_data = file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Call the emotion detection function
        result = detector.detect_emotion(image)
        
        return jsonify({
            "message": "File processed successfully", 
            "emotion": result['emotion'], 
            "confidence": result['confidence']
        }), 200
    else:
        return jsonify({"error": "File type not allowed"}), 400

if __name__ == '__main__':
    app.run(debug=True) 