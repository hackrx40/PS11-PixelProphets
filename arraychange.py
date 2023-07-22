import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.ensemble import BaggingRegressor

# Assuming you have preprocessed input images
# x_train and x_val are numpy arrays of shape (num_samples, height, width, num_channels)

# Normalize images between 0 and 1 (if not done previously)
x_train = x_train.astype('float32') / 255.0
x_val = x_val.astype('float32') / 255.0

# Define the input shape
input_shape = x_train[0].shape

# Create the Convolutional Autoencoder model as a function for KerasRegressor
def create_autoencoder():
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
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

# Create the KerasRegressor with Convolutional Autoencoder model
autoencoder_regressor = KerasRegressor(build_fn=create_autoencoder, epochs=10, batch_size=32, verbose=1)

# Create the ensemble of autoencoders using BaggingRegressor
num_estimators = 5  # Number of autoencoders in the ensemble
ensemble = BaggingRegressor(base_estimator=autoencoder_regressor, n_estimators=num_estimators, random_state=42)

# Train the ensemble of autoencoders
ensemble.fit(x_train, x_train)

# Evaluate the ensemble on the validation set
ensemble_score = ensemble.score(x_val, x_val)

# Note: The "score" method of the ensemble returns the R^2 coefficient of determination by default, but you can use any evaluation metric suitable for your task.
