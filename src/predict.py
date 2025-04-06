import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

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

def evaluate_emotions():
    # Load the trained model
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'emotion_model.h5')
    model = load_model(model_path)
    
    # Get the test directory path
    test_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'test')
    
    # Initialize lists to store predictions and true labels
    all_predictions = []
    all_true_labels = []
    
    # Process each emotion directory
    for emotion in os.listdir(test_dir):
        emotion_dir = os.path.join(test_dir, emotion)
        if not os.path.isdir(emotion_dir):
            continue
            
        print(f"\nProcessing {emotion} images...")
        
        # Get list of image files
        image_files = [f for f in os.listdir(emotion_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        # Limit to 100 images per emotion
        image_files = image_files[:100]
        
        for img_file in image_files:
            # Read and preprocess the image
            img_path = os.path.join(emotion_dir, img_file)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Could not read image: {img_path}")
                continue
                
            processed_img = preprocess_image(img)
            
            # Make prediction
            prediction = model.predict(processed_img, verbose=0)
            predicted_emotion = get_emotion_label(prediction)
            confidence = np.max(prediction) * 100
            
            # Store prediction and true label
            all_predictions.append(predicted_emotion)
            all_true_labels.append(emotion)
            
            # Print individual prediction
            print(f"Image: {img_file}")
            print(f"True Emotion: {emotion}")
            print(f"Predicted: {predicted_emotion} (Confidence: {confidence:.2f}%)")
            print("-" * 50)
    
    # Calculate overall accuracy
    accuracy = accuracy_score(all_true_labels, all_predictions)
    print(f"\nOverall Accuracy: {accuracy:.4f}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(all_true_labels, all_predictions))
    
    # Create and plot confusion matrix
    cm = confusion_matrix(all_true_labels, all_predictions)
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised'],
                yticklabels=['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

if __name__ == "__main__":
    evaluate_emotions()