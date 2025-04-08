# Emotion Detection and Face Recognition System

This project combines emotion detection and face recognition capabilities using Flask, OpenCV, and face_recognition libraries.

## Features

- Real-time emotion detection using webcam
- Face recognition with reference image comparison
- Live face comparison
- Image-to-image face comparison
- Web interface for easy interaction

## Prerequisites

- Python 3.9 or higher
- pip (Python package installer)
- CMake (required for dlib installation)
- C++ compiler (required for dlib installation)

## Installation

1. Clone the repository:
```bash
git clone <your-repository-url>
cd emotion-detection-v2
```

2. Install CMake (if not already installed):
   - On macOS: `brew install cmake`
   - On Ubuntu: `sudo apt-get install cmake`
   - On Windows: Download and install from https://cmake.org/download/

3. Create a new virtual environment:
```bash
# Create a new virtual environment
python3 -m venv emotion-detection-v3

# Activate the virtual environment
# On macOS/Linux:
source emotion-detection-v3/bin/activate
# On Windows:
.\emotion-detection-v3\Scripts\activate
```

4. Install pip in the virtual environment:
```bash
# Download get-pip.py
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py

# Install pip in the virtual environment
./emotion-detection-v3/bin/python get-pip.py
```

5. Install the required packages:
```bash
# Install using the virtual environment's pip
./emotion-detection-v3/bin/pip install -r requirements.txt
```

## Running the Application

1. Make sure your virtual environment is activated:
```bash
source emotion-detection-v3/bin/activate  # On macOS/Linux
```

2. Start the Flask application:
```bash
python app.py
```

3. Open your web browser and navigate to:
```
http://localhost:8080
```

## Available Routes

- `/` - Main emotion detection page
- `/face_recognition` - Face recognition page
- `/video_feed` - Live video stream with emotion detection
- `/get_emotion` - Get current emotion data
- `/set_reference_image` - Set reference image for face recognition
- `/compare_live` - Compare current frame with reference image
- `/compare_images` - Compare two uploaded images

## Common Issues and Solutions

1. **dlib Installation Issues**:
   - Ensure CMake is installed before installing requirements
   - Make sure you have a C++ compiler installed

2. **OpenCV Camera Access**:
   - Grant camera permissions to your browser
   - Ensure no other application is using the camera

3. **Package Installation Issues**:
   - Always use the virtual environment's pip for installation
   - If using Anaconda, prefer creating a new venv instead

4. **Port Conflicts**:
   - The application runs on port 8080
   - If port 8080 is in use, modify the port number in `app.py`

## Notes

- The application requires good lighting for optimal face detection
- For face recognition, use clear, well-lit face images
- The emotion detection model may take a moment to load on first run

## License

[Your chosen license]
