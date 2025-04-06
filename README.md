# Real-time Emotion Detection System

A modern web application that performs real-time emotion detection using facial recognition. The system supports both live webcam feed and image uploads, providing instant emotion classification with confidence scores.

## Features

- **Real-time Webcam Emotion Detection**
  - Live facial emotion analysis
  - Continuous emotion updates (1-second intervals)
  - Smooth, responsive interface

- **Image Upload Support**
  - Drag-and-drop interface
  - Click-to-browse option
  - Support for PNG, JPG, and JPEG formats
  - Instant emotion classification

- **Modern User Interface**
  - Clean, dark theme design
  - Responsive layout for all devices
  - Intuitive drag-and-drop upload area
  - Real-time emotion display with confidence scores
  - Smooth animations and transitions

## Technical Details

### Model Architecture
- Uses a pre-trained deep learning model for emotion detection
- Supports 7 basic emotions:
  - Happy
  - Sad
  - Angry
  - Neutral
  - Fear
  - Surprise
  - Disgust

### Performance Metrics
- Average accuracy: 65-70% on standard test sets
- Processing speed: ~100ms per frame
- Confidence threshold: 0.5 (50%) for classification

### System Requirements
- Python 3.8+
- OpenCV for image processing
- Flask for web server
- Modern web browser with camera access
- Minimum 4GB RAM recommended

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/emotion-detection-v2.git
cd emotion-detection-v2
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

4. Open your browser and navigate to:
```
http://localhost:5000
```

## Usage

### Webcam Mode
1. Allow camera access when prompted
2. Position your face clearly in the frame
3. The detected emotion will update in real-time
4. Confidence score is displayed below the emotion

### Image Upload Mode
1. Drag and drop an image into the upload area
2. Or click the upload area to browse files
3. Click the "Classify Image" button
4. View the emotion detection results

## Limitations

- Requires good lighting conditions for optimal performance
- Works best with front-facing, clear facial images
- May have reduced accuracy with:
  - Side profile faces
  - Heavily occluded faces
  - Low-resolution images
  - Extreme lighting conditions

## Browser Support

- Chrome (recommended)
- Firefox
- Safari
- Edge

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Emotion detection model based on [source]
- UI design inspired by modern web applications
- Flask framework for the web server
- OpenCV for image processing

## Contact

For questions or feedback, please open an issue in the GitHub repository.
