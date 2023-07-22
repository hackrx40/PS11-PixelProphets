import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

# Assuming you have preprocessed input images and the latent dimension
# x_train and x_val are numpy arrays of shape (num_samples, height, width, num_channels)

# Normalize images between 0 and 1 (if not done previously)
x_train = x_train.astype('float32') / 255.0
x_val = x_val.astype('float32') / 255.0

# Define the input shape
input_shape = x_train[0].shape

# Define the number of dimensions in the latent space
latent_dim = 64  # Adjust this value based on your requirements

# Build the Variational Autoencoder model
input_img = Input(shape=input_shape)

# Encoder layers
x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)

# Flatten the last convolutional layer
x = Flatten()(x)

# VAE specific layers
# Variational mean and log variance
z_mean = Dense(latent_dim)(x)
z_log_var = Dense(latent_dim)(x)

# Sampling function
def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

# Reparameterization trick for sampling from the latent distribution
z = Lambda(sampling)([z_mean, z_log_var])

# Decoder layers
decoder_input = Input(shape=(latent_dim,))
x = Dense(np.prod(input_shape))(decoder_input)
x = tf.keras.layers.Reshape(input_shape)(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(input_shape[-1], (3, 3), activation='sigmoid', padding='same')(x)

# Create the VAE model
encoder = Model(input_img, z_mean)  # Encoder model for getting the latent mean
vae = Model(input_img, decoded)     # Full VAE model

# VAE loss function
def vae_loss(x, decoded):
    reconstruction_loss = K.mean(K.square(x - decoded), axis=[1, 2, 3])
    kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return reconstruction_loss + kl_loss

# Compile the model
vae.compile(optimizer='adam', loss=vae_loss)

# Print the model summary (optional)
vae.summary()
