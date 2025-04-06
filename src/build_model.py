from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

def build_model(input_shape=(48, 48, 1), num_classes=7):
    model = Sequential([
        # First Convolutional Block with increased regularization
        Conv2D(32, (3, 3), activation='relu', padding='same', 
               kernel_regularizer=l2(0.05),  # Increased L2 regularization
               input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.4),  # Increased dropout

        # Second Convolutional Block with increased regularization
        Conv2D(64, (3, 3), activation='relu', padding='same',
               kernel_regularizer=l2(0.05)),  # Increased L2 regularization
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.4),  # Increased dropout

        # Flatten and Dense Layers with increased regularization
        Flatten(),
        Dense(64, activation='relu',  # Reduced number of neurons
              kernel_regularizer=l2(0.05)),  # Increased L2 regularization
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    # Compile with a lower learning rate
    optimizer = Adam(learning_rate=0.0001)  # Lower learning rate
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model 