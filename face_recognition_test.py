import face_recognition
import numpy as np
from PIL import Image
import os

class FaceRecognitionTester:
    def __init__(self):
        """Initialize the face recognition tester."""
        self.tolerance = 0.6  # Lower is more strict, higher is more lenient
        
    def load_image(self, image_path):
        """
        Load and preprocess an image for face recognition.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            numpy.ndarray: The loaded image array
        """
        return face_recognition.load_image_file(image_path)
    
    def get_face_encoding(self, image):
        """
        Get face encodings from an image.
        
        Args:
            image (numpy.ndarray): The image array
            
        Returns:
            list: List of face encodings found in the image
        """
        return face_recognition.face_encodings(image)
    
    def compare_faces(self, image1_path, image2_path):
        """
        Compare two images to determine if they show the same person.
        
        Args:
            image1_path (str): Path to the first image
            image2_path (str): Path to the second image
            
        Returns:
            dict: Dictionary containing comparison results
        """
        # Load images
        image1 = self.load_image(image1_path)
        image2 = self.load_image(image2_path)
        
        # Get face encodings
        face_encodings1 = self.get_face_encoding(image1)
        face_encodings2 = self.get_face_encoding(image2)
        
        if not face_encodings1 or not face_encodings2:
            return {
                "success": False,
                "error": "No faces found in one or both images",
                "details": {
                    "faces_in_image1": len(face_encodings1),
                    "faces_in_image2": len(face_encodings2)
                }
            }
        
        # Compare all face encodings
        matches = []
        distances = []
        
        for face_encoding1 in face_encodings1:
            for face_encoding2 in face_encodings2:
                # Calculate face distance
                face_distance = face_recognition.face_distance([face_encoding2], face_encoding1)[0]
                is_match = face_distance <= self.tolerance
                
                matches.append(is_match)
                distances.append(face_distance)
        
        # Get the best match
        best_match = any(matches)
        best_distance = min(distances) if distances else None
        
        return {
            "success": True,
            "is_match": best_match,
            "confidence": 1 - best_distance if best_distance is not None else 0,
            "details": {
                "faces_in_image1": len(face_encodings1),
                "faces_in_image2": len(face_encodings2),
                "best_distance": best_distance,
                "tolerance": self.tolerance
            }
        }

def test_face_recognition():
    """Test the face recognition system with sample images."""
    tester = FaceRecognitionTester()
    
    # Create a test directory if it doesn't exist
    test_dir = "face_recognition_test"
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
        print(f"Created test directory: {test_dir}")
        print("Please add test images to this directory and update the test paths below.")
        return
    
    # Test cases with the correct file names
    test_cases = [
        {
            "name": "Same person, different angles",
            "image1": os.path.join(test_dir, "person1_angle1.jpg"),
            "image2": os.path.join(test_dir, "person1_angle2.jpg")
        },
        {
            "name": "Different people",
            "image1": os.path.join(test_dir, "person1_angle1.jpg"),
            "image2": os.path.join(test_dir, "person2.jpg")
        }
    ]
    
    for test_case in test_cases:
        print(f"\nTesting: {test_case['name']}")
        print(f"Image 1: {test_case['image1']}")
        print(f"Image 2: {test_case['image2']}")
        
        if not os.path.exists(test_case['image1']) or not os.path.exists(test_case['image2']):
            print("Error: One or both test images not found. Skipping test case.")
            continue
        
        result = tester.compare_faces(test_case['image1'], test_case['image2'])
        
        if not result['success']:
            print(f"Error: {result['error']}")
            print(f"Details: {result['details']}")
        else:
            print(f"Match: {'Yes' if result['is_match'] else 'No'}")
            print(f"Confidence: {result['confidence']:.2%}")
            print(f"Details: {result['details']}")

if __name__ == "__main__":
    test_face_recognition() 