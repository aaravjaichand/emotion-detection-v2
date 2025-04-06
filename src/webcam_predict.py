import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Resize to 48x48
    resized = cv2.resize(gray, (48, 48))
    # Normalize
    normalized = resized / 255.0
    # Reshape for model input
    reshaped = normalized.reshape(1, 48, 48, 1)
    return reshaped

def get_emotion_label(prediction):
    emotions = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
    return emotions[np.argmax(prediction)]

def run_webcam():
    # Load the trained model
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'emotion_model.h5')
    model = load_model(model_path)
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Check if webcam is opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Webcam started. Press 'q' to quit.")
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame")
            break
            
        # Preprocess the frame
        processed_frame = preprocess_image(frame)
        
        # Make prediction
        prediction = model.predict(processed_frame, verbose=0)
        emotion = get_emotion_label(prediction)
        confidence = np.max(prediction) * 100
        
        # Draw rectangle for face
        height, width = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (width, height), (0, 255, 0), 2)
        
        # Display emotion and confidence
        text = f"{emotion}: {confidence:.2f}%"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display the frame
        cv2.imshow('Emotion Detection', frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_webcam() 