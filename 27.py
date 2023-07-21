import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model

# Assuming you have preprocessed input images and the latent dimension
# x_train and x_val are numpy arrays of shape (num_samples, height, width, num_channels)

# Normalize images between 0 and 1 (if not done previously)
x_train = x_train.astype('float32') / 255.0
x_val = x_val.astype('float32') / 255.0

# Define the input shape
input_shape = x_train[0].shape

# Define the number of dimensions in the latent space
latent_dim = 64  # Adjust this value based on your requirements

# Build the Convolutional Autoencoder model
input_img = Input(shape=input_shape)

# Encoder layers
x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# Decoder layers
x = Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(input_shape[-1], (3, 3), activation='sigmoid', padding='same')(x)

# Create the autoencoder model
autoencoder = Model(input_img, decoded)

# Compile the model
autoencoder.compile(optimizer='adam', loss='mse')

# Print the model summary (optional)
autoencoder.summary()
