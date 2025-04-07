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
import face_recognition
import tempfile

app = Flask(__name__)
detector = EmotionDetector()

# Global variables for face recognition
reference_image_encoding = None
face_recognizer_tolerance = 0.5

# Global variable to store current emotion
current_emotion = {"emotion": "Neutral", "confidence": 0.0}

# Set the allowed extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def get_camera():
    """Get camera instance with error handling."""
    try:
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            raise Exception("Could not open camera")
        return camera
    except Exception as e:
        print(f"Error initializing camera: {e}")
        return None

def generate_frames():
    """Generate frames from webcam with emotion detection."""
    camera = get_camera()
    if camera is None:
        return
        
    # Set camera properties for better performance
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Reduced resolution
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_FPS, 30)  # Set to 30 FPS
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer size
    
    # Initialize variables for frame skipping
    frame_count = 0
    last_emotion_time = time.time()
    emotion_update_interval = 0.5
    
    try:
        while True:
            success, frame = camera.read()
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
    finally:
        # Ensure resources are properly released
        camera.release()
        cv2.destroyAllWindows()

@app.route('/')
def index():
    """Render the main emotion detection page."""
    return render_template('index.html')

@app.route('/face_recognition')
def face_recognition_page():
    """Render the face recognition page."""
    return render_template('face_recognition.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_emotion')
def get_emotion():
    """Get current emotion data."""
    return jsonify(current_emotion)

def allowed_file(filename):
    """Check if the file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_uploaded_image(file):
    """Process an uploaded image file for face recognition."""
    if file and allowed_file(file.filename):
        # Read the image data
        image_data = file.read()
        
        # Convert to numpy array for face_recognition
        image = face_recognition.load_image_file(io.BytesIO(image_data))
        
        # Get face encodings
        face_encodings = face_recognition.face_encodings(image)
        
        if not face_encodings:
            return None
            
        return face_encodings[0]
    return None

@app.route('/set_reference_image', methods=['POST'])
def set_reference_image():
    """Set the reference image for live comparison."""
    if 'reference_image' not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"}), 400
        
    file = request.files['reference_image']
    face_encoding = process_uploaded_image(file)
    
    if face_encoding is None:
        return jsonify({"success": False, "error": "No face detected in image"}), 400
        
    global reference_image_encoding
    reference_image_encoding = face_encoding
    
    return jsonify({"success": True})

@app.route('/compare_live', methods=['POST'])
def compare_live():
    """Compare the current frame with the reference image."""
    if reference_image_encoding is None:
        return jsonify({"success": False, "error": "No reference image set"}), 400
        
    if 'current_image' not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"}), 400
        
    file = request.files['current_image']
    current_encoding = process_uploaded_image(file)
    
    if current_encoding is None:
        return jsonify({"success": False, "error": "No face detected in current frame"}), 400
    
    # Compare face encodings
    face_distance = face_recognition.face_distance([reference_image_encoding], current_encoding)[0]
    is_match = bool(face_distance <= face_recognizer_tolerance)  # Convert to Python bool
    
    return jsonify({
        "success": True,
        "is_match": is_match,
        "confidence": float(1 - face_distance)  # Convert to Python float
    })

@app.route('/compare_images', methods=['POST'])
def compare_images():
    """Compare two uploaded images for face recognition."""
    if 'image1' not in request.files or 'image2' not in request.files:
        return jsonify({"success": False, "error": "Two images required"}), 400
        
    # Process both images
    encoding1 = process_uploaded_image(request.files['image1'])
    encoding2 = process_uploaded_image(request.files['image2'])
    
    if encoding1 is None or encoding2 is None:
        return jsonify({"success": False, "error": "No face detected in one or both images"}), 400
    
    # Compare face encodings
    face_distance = face_recognition.face_distance([encoding1], encoding2)[0]
    is_match = bool(face_distance <= face_recognizer_tolerance)  # Convert to Python bool
    
    return jsonify({
        "success": True,
        "is_match": is_match,
        "confidence": float(1 - face_distance)  # Convert to Python float
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)