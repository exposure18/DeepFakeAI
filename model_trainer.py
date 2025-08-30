import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import os


def build_model(image_size):
    """
    Builds a Sequential Convolutional Neural Network (CNN) model.

    Args:
        image_size (tuple): The size of the input images (height, width).

    Returns:
        tf.keras.Sequential: The compiled CNN model.
    """
    model = Sequential([
        layers.Rescaling(1. / 255, input_shape=(image_size[0], image_size[1], 3)),
        layers.Conv2D(16, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


if __name__ == "__main__":
    # Define dataset parameters
    IMAGE_SIZE = (150, 150)
    BATCH_SIZE = 32
    DATA_DIR = 'data'  # Assumes 'data' folder is in the project root

    # Load the dataset from your 'data' directory
    train_ds = image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE
    )

    val_ds = image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE
    )

    class_names = train_ds.class_names
    print(f"Detected classes: {class_names}")

    # Build the model
    model = build_model(IMAGE_SIZE)
    model.summary()

    # Train the model
    EPOCHS = 10
    print("\nStarting model training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS
    )
    print("\nModel training complete!")

    # Evaluate the model
    print("\nEvaluating model performance...")
    loss, accuracy = model.evaluate(val_ds)
    print(f"Final Validation Accuracy: {accuracy * 100:.2f}%")

    # Save the trained model to a file
    model_path = os.path.join(os.getcwd(), 'deepfake_detector.h5')
    model.save(model_path)
    print(f"\nModel saved to: {model_path}")