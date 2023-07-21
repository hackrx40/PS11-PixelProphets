import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model

# Assuming you have preprocessed input images
# x_train and x_val are numpy arrays of shape (num_samples, height, width, num_channels)

# Normalize images between 0 and 1 (if not done previously)
x_train = x_train.astype('float32') / 255.0
x_val = x_val.astype('float32') / 255.0

# Define the input shape
input_shape = x_train[0].shape

# Build the Convolutional Autoencoder model
input_img = Input(shape=input_shape)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(input_shape[-1], (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)

# Compile the model
autoencoder.compile(optimizer='adam', loss='mse')

# Train the model
autoencoder.fit(x_train, x_train, epochs=100, batch_size=32, validation_data=(x_val, x_val), verbose=1)

# Use the trained autoencoder for anomaly segmentation
# Get the reconstruction of validation data
reconstructed_images = autoencoder.predict(x_val)

# Calculate pixel-wise reconstruction error
pixelwise_errors = np.square(x_val - reconstructed_images)

# Define a threshold for anomaly detection
threshold = 0.01  # You may need to fine-tune this threshold based on your specific use case and dataset.

# Identify anomalies (anomalous pixels) based on the threshold
anomaly_mask = pixelwise_errors > threshold

# Optionally, you can visualize the anomalous regions
import matplotlib.pyplot as plt

# Display some examples of anomalous regions
num_display = 5
for i in range(num_display):
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(x_val[i])
    plt.title("Original Image")

    plt.subplot(1, 2, 2)
    plt.imshow(anomaly_mask[i], cmap='gray')
    plt.title("Anomaly Mask")

    plt.show()
