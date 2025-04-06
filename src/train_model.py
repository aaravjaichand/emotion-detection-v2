import os
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from .preprocess import get_data_generators
from .build_model import build_model

def train_model():
    # Get data generators
    train_gen, validation_gen, test_gen = get_data_generators()
    
    # Build the model
    model = build_model()
    
    # Define callbacks
    callbacks = [
        # Save the best model based on validation accuracy
        ModelCheckpoint(
            'models/emotion_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        # Stop training if validation accuracy doesn't improve
        EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True
        ),
        # Reduce learning rate when validation accuracy plateaus
        ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.2,
            patience=5,
            min_lr=0.00001
        )
    ]
    
    # Train the model with more epochs and validation data
    history = model.fit(
        train_gen,
        validation_data=validation_gen,
        epochs=100,  # Increased epochs
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate the model on test data
    test_loss, test_accuracy = model.evaluate(test_gen)
    print(f"\nTest accuracy: {test_accuracy:.4f}")

if __name__ == "__main__":
    train_model()