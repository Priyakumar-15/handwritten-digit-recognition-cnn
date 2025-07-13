# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print(f"TensorFlow version: {tf.__version__}")
print(f"Keras version: {keras.__version__}")

# --- 1. Fetching the dataset ---
print("Fetching MNIST dataset...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
x, y = mnist['data'], mnist['target']

# Convert data to numpy arrays (if not already) and ensure correct types
x = x.astype('float32') # Pixel values should be float
y = y.astype('int')     # Labels should be integers

# --- 2. Data Preprocessing for CNN ---

# Normalize pixel values to 0-1 range
x /= 255.0

# Reshape images to 28x28 with a single channel (grayscale)
# Keras expects input shape (batch_size, height, width, channels)
x = x.reshape(-1, 28, 28, 1)

# One-hot encode the target labels (e.g., 2 becomes [0,0,1,0,0,0,0,0,0,0])
# There are 10 classes (digits 0-9)
num_classes = 10
y = keras.utils.to_categorical(y, num_classes)

# Split data into training and testing sets
# Using sklearn's train_test_split for more robust splitting than simple slicing
print("Splitting data into training and testing sets...")
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42, stratify=y)
# A typical split is 60k train, 10k test.
# Using 15% for test (approx 10.5k samples) and 85% for train (approx 59.5k samples)

print(f"x_train shape: {x_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape}")
print(f"y_test shape: {y_test.shape}")

# --- 3. Visualize a sample digit ---
some_digit_index = 36001
# Ensure the index is within bounds of the original dataset before splitting
if some_digit_index >= len(mnist['data']):
    some_digit_index = np.random.randint(0, len(mnist['data'])) # Pick a random one if out of bounds

original_some_digit = mnist['data'][some_digit_index].reshape(28, 28)
original_some_digit_label = mnist['target'][some_digit_index]

plt.imshow(original_some_digit, cmap=plt.cm.binary, interpolation='nearest')
plt.title(f"Original Digit at Index {some_digit_index}: {original_some_digit_label}")
plt.axis("off")
plt.show()


# --- 4. Build the CNN Model ---
print("\nBuilding the Convolutional Neural Network model...")
model = keras.Sequential([
    # First Convolutional Block
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    # Second Convolutional Block
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    # Flatten the 3D output to 1D to feed into Dense layers
    layers.Flatten(),
    # Fully Connected Layers
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5), # Dropout for regularization to prevent overfitting
    layers.Dense(num_classes, activation='softmax') # Output layer with 10 classes and softmax for probabilities
])

# --- 5. Compile the Model ---
print("Compiling the model...")
model.compile(optimizer='adam',
              loss='categorical_crossentropy', # Appropriate for multi-class classification with one-hot encoded labels
              metrics=['accuracy'])

model.summary()

# --- 6. Train the Model ---
print("\nTraining the model (this will take some time)...")
# You can adjust epochs and batch_size based on your computational resources and desired accuracy
history = model.fit(x_train, y_train,
                    batch_size=128,
                    epochs=10, # Number of passes over the entire dataset
                    validation_split=0.1, # Use a small portion of training data for validation during training
                    verbose=1)

# --- 7. Evaluate the Model ---
print("\nEvaluating the model on the test set...")
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Plot training & validation accuracy values
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.tight_layout()
plt.show()

# --- 8. Make Predictions ---
print("\nMaking predictions on a sample from the test set...")
# Predict probabilities for the first few test samples
predictions = model.predict(x_test[:5])
predicted_classes = np.argmax(predictions, axis=1) # Get the class with the highest probability
true_classes = np.argmax(y_test[:5], axis=1) # Convert one-hot encoded true labels back to single digit

for i in range(5):
    print(f"True label: {true_classes[i]}, Predicted: {predicted_classes[i]}, Probabilities: {predictions[i].round(2)}")
    plt.imshow(x_test[i].reshape(28, 28), cmap=plt.cm.binary)
    plt.title(f"True: {true_classes[i]}, Pred: {predicted_classes[i]}")
    plt.axis('off')
    plt.show()

# --- 9. Detailed Performance Metrics (Confusion Matrix & Classification Report) ---
print("\nGenerating detailed performance metrics...")
y_pred_probs = model.predict(x_test)
y_pred_classes = np.argmax(y_pred_probs, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

print("\nClassification Report:")
print(classification_report(y_true_classes, y_pred_classes, target_names=[str(i) for i in range(num_classes)]))

print("\nConfusion Matrix:")
conf_mat = confusion_matrix(y_true_classes, y_pred_classes)
print(conf_mat)

# Optional: Visualize Confusion Matrix
import seaborn as sns
plt.figure(figsize=(10, 8))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=range(num_classes), yticklabels=range(num_classes))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()