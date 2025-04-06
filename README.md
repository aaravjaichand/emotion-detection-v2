# Emotion Detection System

This project implements a deep learning-based emotion detection system that can classify facial expressions into different emotion categories.

## Project Structure

```
emotion-detector/
├── train/                  # Training images (organized by emotion classes)
├── test/                   # Test images (organized by emotion classes)
├── src/
│   ├── preprocess.py      # Data preprocessing utilities
│   ├── build_model.py     # Model architecture definition
│   ├── train_model.py     # Training loop implementation
│   └── predict.py         # Inference utilities
├── models/                # Directory for saved models
│   └── emotion_model.h5   # Trained model
├── run_training.py        # Main training script
├── requirements.txt       # Project dependencies
└── README.md             # This file
```

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Organize your dataset:
   - Place training images in the `train/` directory
   - Place test images in the `test/` directory
   - Organize images by emotion classes in subdirectories

## Usage

To train the model:
```bash
python run_training.py
```

The trained model will be saved in the `models/` directory.

## Requirements

- Python 3.8+
- TensorFlow 2.8.0+
- OpenCV
- NumPy
- scikit-learn
- Matplotlib
- Pandas

## License

This project is licensed under the MIT License - see the LICENSE file for details.