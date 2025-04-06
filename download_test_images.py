import urllib.request
import os
from PIL import Image
import io

def download_image(url, save_path):
    """Download an image from URL and save it to the specified path."""
    try:
        response = urllib.request.urlopen(url)
        image_data = response.read()
        image = Image.open(io.BytesIO(image_data))
        image.save(save_path)
        print(f"Successfully downloaded: {save_path}")
        return True
    except Exception as e:
        print(f"Error downloading {url}: {str(e)}")
        return False

def main():
    # Create test directory if it doesn't exist
    test_dir = "face_recognition_test"
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    
    # Test images from a more reliable source
    test_images = [
        {
            "url": "https://raw.githubusercontent.com/ageitgey/face_recognition/master/examples/obama.jpg",
            "filename": "person1_angle1.jpg"
        },
        {
            "url": "https://raw.githubusercontent.com/ageitgey/face_recognition/master/examples/obama2.jpg",
            "filename": "person1_angle2.jpg"
        },
        {
            "url": "https://raw.githubusercontent.com/ageitgey/face_recognition/master/examples/biden.jpg",
            "filename": "person2.jpg"
        }
    ]
    
    print("Downloading test images...")
    
    # Download all test images
    for img in test_images:
        save_path = os.path.join(test_dir, img["filename"])
        download_image(img["url"], save_path)
    
    print("\nDownload complete! You can now run the face recognition test script.")

if __name__ == "__main__":
    main() 