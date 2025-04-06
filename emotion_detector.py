from transformers import pipeline
from PIL import Image
import numpy as np
import cv2
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os

class EmotionDetector:
    def __init__(self):
        """Initialize the emotion detection pipeline."""
        self.pipe = pipeline("image-classification", model="dima806/facial_emotions_image_detection")
    
    def detect_emotion(self, image):
        """
        Detect emotion in the given image.
        
        Args:
            image: Can be either a PIL Image, numpy array, or path to image file
            
        Returns:
            dict: Dictionary containing the predicted emotion and confidence score
        """
        # Convert numpy array to PIL Image if necessary
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Run inference
        results = self.pipe(image)
        
        # Get the top prediction
        prediction = results[0]
        
        return {
            "emotion": prediction["label"],
            "confidence": prediction["score"]
        }

    def start_webcam(self):
        """
        Start webcam feed and perform real-time emotion detection.
        Press 'q' to quit.
        """
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        print("Webcam started. Press 'q' to quit.")
        
        while True:
            # Read frame from webcam
            ret, frame = cap.read()
            if not ret:
                print("Error: Can't receive frame")
                break
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect emotion
            result = self.detect_emotion(frame_rgb)
            
            # Display emotion and confidence on frame
            text = f"Emotion: {result['emotion']} ({result['confidence']:.2f})"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display the frame
            cv2.imshow('Emotion Detection', frame)
            
            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release resources
        cap.release()
        cv2.destroyAllWindows()

    def upload_image(self, image_path):
        """
        Upload an image from the specified path and detect emotion.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            dict: Dictionary containing the predicted emotion and confidence score
        """
        # Load the image
        image = Image.open(image_path)
        
        # Detect emotion
        result = self.detect_emotion(image)
        
        return result

# Example usage
if __name__ == "__main__":
    # Initialize the detector
    detector = EmotionDetector()
    
    # Start webcam feed
    detector.start_webcam()
    
    # Example with a test image
    # Replace 'path_to_image.jpg' with your image path
    # result = detector.upload_image('path_to_image.jpg')
    # print(f"Detected emotion: {result['emotion']}")
    # print(f"Confidence: {result['confidence']:.2f}")

app = Flask(__name__)

# Set the upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('upload.html')

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

if __name__ == "__main__":
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True) 