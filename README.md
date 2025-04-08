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
git clone https://github.com/aaravjaichand/emotion-detection-v2.git
cd emotion-detection-v2
```

2. Install System Dependencies:

### macOS:
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv)"
brew install cmake
```

### Windows:
1. Install Visual Studio Build Tools:
   - Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
   - During installation, select "Desktop development with C++"
2. Install CMake:
   - Download from: https://cmake.org/download/
   - Choose the Windows x64 Installer
   - During installation, select "Add CMake to the system PATH"

3. Create a new virtual environment:
```bash
python3 -m venv emotion-detection-v2
```

4. Activate the virtual environment:

For macOS/Linux:
```bash
source emotion-detection-v2/bin/activate
```

For Windows:
```bash
.\emotion-detection-v2\Scripts\activate
```

5. Install pip in the virtual environment:
```bash
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
./emotion-detection-v2/bin/python get-pip.py
```

6. Install the required packages:
```bash
./emotion-detection-v2/bin/pip install -r requirements.txt
```

## Running the Application

1. Activate the virtual environment:

For macOS/Linux:
```bash
source emotion-detection-v2/bin/activate
```

For Windows:
```bash
.\emotion-detection-v2\Scripts\activate
```

2. Start the Flask application:
```bash
python app.py
```

3. Open your web browser and navigate to:
http://localhost:8080

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
   - On Windows, ensure Visual Studio Build Tools are installed
   - On macOS, ensure Xcode Command Line Tools are installed (`xcode-select --install`)

2. **OpenCV Camera Access**:
   - Grant camera permissions to your browser
   - Ensure no other application is using the camera
   - On macOS, check System Preferences > Security & Privacy > Privacy > Camera

3. **Package Installation Issues**:
   - Always use the virtual environment's pip for installation
   - If using Anaconda, prefer creating a new venv instead
   - If pip fails, try upgrading pip first: `python -m pip install --upgrade pip`

4. **Port Conflicts**:
   - The application runs on port 8080
   - If port 8080 is in use, modify the port number in `app.py`

## Notes

- The application requires good lighting for optimal face detection
- For face recognition, use clear, well-lit face images
- The emotion detection model may take a moment to load on first run
- On macOS, if you get SSL warnings, you may need to install OpenSSL: `brew install openssl`

## License

MIT
