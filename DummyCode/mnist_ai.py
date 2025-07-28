import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Step 1: Load the data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Step 2: Normalize the data (values between 0 and 1)
x_train = x_train / 255.0
x_test = x_test / 255.0

# Step 3: Build the model
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),      # Turn 2D image into 1D array
    layers.Dense(128, activation='relu'),      # Hidden layer
    layers.Dropout(0.2),                       # Prevent overfitting
    layers.Dense(10, activation='softmax')     # Output: one score per digit
])

# Step 4: Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Step 5: Train the model
model.fit(x_train, y_train, epochs=5, validation_split=0.1)

# Step 6: Evaluate on test set
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\nTest accuracy: {test_acc:.3f}")

# Optional: Visualize predictions
predictions = model.predict(x_test)

# Show the first test image and prediction
plt.imshow(x_test[0], cmap='gray')
plt.title(f"Predicted: {predictions[0].argmax()}, Actual: {y_test[0]}")
plt.show()
