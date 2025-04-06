from transformers import pipeline
from PIL import Image
import numpy as np
import cv2

class EmotionDetector:
    def __init__(self):
        """Initialize the emotion detection pipeline."""
        self.pipe = pipeline("image-classification", model="prithivMLmods/Facial-Emotion-Detection-SigLIP2")
    
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