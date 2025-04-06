from flask import Blueprint, render_template, Response, jsonify, request
import cv2
import numpy as np
import time
import io
from werkzeug.utils import secure_filename
import os

from app.models.emotion_detector import EmotionDetector
from app.models.face_recognizer import FaceRecognizer

main = Blueprint('main', __name__)

# Initialize detectors
detector = EmotionDetector()
face_recognizer = FaceRecognizer()

# Global variable to store current emotion
current_emotion = {"emotion": "Neutral", "confidence": 0.0}

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
    emotion_update_interval = 0.5
    
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

@main.route('/')
def index():
    """Render the main emotion detection page."""
    return render_template('index.html')

@main.route('/face_recognition')
def face_recognition_page():
    """Render the face recognition page."""
    return render_template('face_recognition.html')

@main.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@main.route('/get_emotion')
def get_emotion():
    """Get current emotion data."""
    return jsonify(current_emotion)

@main.route('/set_reference_image', methods=['POST'])
def set_reference_image():
    """Set the reference image for live comparison."""
    if 'reference_image' not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"}), 400
        
    file = request.files['reference_image']
    success, message = face_recognizer.set_reference_image(file)
    
    if not success:
        return jsonify({"success": False, "error": message}), 400
    
    return jsonify({"success": True, "message": message})

@main.route('/compare_live', methods=['POST'])
def compare_live():
    """Compare the current frame with the reference image."""
    if 'current_image' not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"}), 400
        
    file = request.files['current_image']
    success, message, is_match, confidence = face_recognizer.compare_live(file)
    
    if not success:
        return jsonify({"success": False, "error": message}), 400
    
    return jsonify({
        "success": True,
        "is_match": is_match,
        "confidence": confidence
    })

@main.route('/compare_images', methods=['POST'])
def compare_images():
    """Compare two uploaded images for face recognition."""
    if 'image1' not in request.files or 'image2' not in request.files:
        return jsonify({"success": False, "error": "Two images required"}), 400
        
    success, message, is_match, confidence = face_recognizer.compare_images(
        request.files['image1'],
        request.files['image2']
    )
    
    if not success:
        return jsonify({"success": False, "error": message}), 400
    
    return jsonify({
        "success": True,
        "is_match": is_match,
        "confidence": confidence
    }) 