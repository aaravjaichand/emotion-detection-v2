import face_recognition
import io
from PIL import Image
import numpy as np

class FaceRecognizer:
    def __init__(self, tolerance=0.6):
        """Initialize the face recognizer with a tolerance value."""
        self.tolerance = tolerance
        self.reference_image_encoding = None

    def process_uploaded_image(self, file):
        """Process an uploaded image file for face recognition."""
        if file:
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

    def set_reference_image(self, file):
        """Set the reference image for comparison."""
        face_encoding = self.process_uploaded_image(file)
        if face_encoding is None:
            return False, "No face detected in image"
        
        self.reference_image_encoding = face_encoding
        return True, "Reference image set successfully"

    def compare_live(self, file):
        """Compare the current frame with the reference image."""
        if self.reference_image_encoding is None:
            return False, "No reference image set", None, None
        
        current_encoding = self.process_uploaded_image(file)
        if current_encoding is None:
            return False, "No face detected in current frame", None, None
        
        # Compare face encodings
        face_distance = face_recognition.face_distance([self.reference_image_encoding], current_encoding)[0]
        is_match = bool(face_distance <= self.tolerance)
        
        return True, "Comparison successful", is_match, float(1 - face_distance)

    def compare_images(self, file1, file2):
        """Compare two uploaded images for face recognition."""
        encoding1 = self.process_uploaded_image(file1)
        encoding2 = self.process_uploaded_image(file2)
        
        if encoding1 is None or encoding2 is None:
            return False, "No face detected in one or both images", None, None
        
        # Compare face encodings
        face_distance = face_recognition.face_distance([encoding1], encoding2)[0]
        is_match = bool(face_distance <= self.tolerance)
        
        return True, "Comparison successful", is_match, float(1 - face_distance) 