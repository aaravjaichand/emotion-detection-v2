import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_data_generators(img_size=(48, 48), batch_size=32):
    # Get the project root directory (one level up from src)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Enhanced data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        brightness_range=[0.8, 1.2],
        validation_split=0.2  # Use 20% of training data for validation
    )
    
    # Only rescaling for test data
    test_datagen = ImageDataGenerator(rescale=1./255)

    # Generate training data from the 'train' directory
    train_gen = train_datagen.flow_from_directory(
        os.path.join(project_root, 'train'),
        target_size=img_size,
        color_mode='grayscale',
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=True,
        subset='training'  # Specify this is training data
    )

    # Generate validation data
    validation_gen = train_datagen.flow_from_directory(
        os.path.join(project_root, 'train'),
        target_size=img_size,
        color_mode='grayscale',
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=False,
        subset='validation'  # Specify this is validation data
    )

    # Generate test data from the 'test' directory
    test_gen = test_datagen.flow_from_directory(
        os.path.join(project_root, 'test'),
        target_size=img_size,
        color_mode='grayscale',
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=False
    )

    # Print class counts
    print("\nTraining set class counts:")
    for class_name, class_index in train_gen.class_indices.items():
        class_count = sum(1 for _ in train_gen.filepaths if class_name in _)
        print(f"Class {class_name}: {class_count} images")

    return train_gen, validation_gen, test_gen