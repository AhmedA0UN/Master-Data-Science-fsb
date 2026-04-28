# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# Assuming you have the following datasets
# train_set_x_orig: images for training (shape: (209, 64, 64, 3))
# train_set_y: corresponding labels for training (shape: (209,))
# test_set_x_orig: images for testing (shape: (N, 64, 64, 3))
# test_set_y: corresponding labels for testing (shape: (N,))

# Step 1: Reshape labels to match the image data
train_set_y = train_set_y.reshape(-1)  # Ensures the shape is (209,)
test_set_y = test_set_y.reshape(-1)  # Similarly reshape test labels

# Step 2: Normalize the image data (important for neural networks)
train_set_x_orig = train_set_x_orig / 255.0
test_set_x_orig = test_set_x_orig / 255.0

# Step 3: One-hot encode the labels
train_set_y = to_categorical(train_set_y, num_classes=2)  # Change num_classes based on your dataset
test_set_y = to_categorical(test_set_y, num_classes=2)

# Step 4: Initialize the ImageDataGenerator for data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,  # Randomly rotate images in the range of 20 degrees
    width_shift_range=0.2,  # Randomly shift images horizontally by 20%
    height_shift_range=0.2,  # Randomly shift images vertically by 20%
    shear_range=0.2,  # Apply shear transformations
    zoom_range=0.2,  # Random zoom
    horizontal_flip=True,  # Randomly flip images horizontally
    fill_mode='nearest'  # Fill any new pixels with the nearest value
)

# Fit the datagen on the training data
datagen.fit(train_set_x_orig)

# Step 5: Build a neural network model (e.g., CNN)
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')  # Assuming 2 classes for classification
])

# Step 6: Compile the model using Adam optimizer
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Step 7: Train the model with data augmentation
history = model.fit(datagen.flow(train_set_x_orig, train_set_y, batch_size=32),
                    epochs=50,  # Adjust number of epochs as needed
                    validation_data=(test_set_x_orig, test_set_y))

# Step 8: Plotting the results

# Plot training & validation accuracy
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plot training & validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Step 9: Evaluate the model
score = model.evaluate(test_set_x_orig, test_set_y, verbose=0)
print(f'Test Loss: {score[0]}')
print(f'Test Accuracy: {score[1]}')
