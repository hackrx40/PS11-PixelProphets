import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Assuming you have loaded and preprocessed your normal and anomalous images
# normal_images and anomalous_images should be numpy arrays of shape (num_samples, height, width, num_channels)

# Combine the normal and anomalous images into a single array
x_train = np.concatenate((normal_images, anomalous_images), axis=0)

# Normalize images between 0 and 1
x_train = x_train.astype('float32') / 255.0

# Create an ImageDataGenerator object with data augmentation settings
datagen = ImageDataGenerator(
    rotation_range=20,       # Random rotation (±20 degrees)
    width_shift_range=0.1,   # Random horizontal shift (±10% of the image width)
    height_shift_range=0.1,  # Random vertical shift (±10% of the image height)
    shear_range=0.2,         # Random shear transformation
    zoom_range=0.2,          # Random zoom (±20%)
    horizontal_flip=True,    # Random horizontal flip
    fill_mode='nearest'      # Filling strategy for newly created pixels
)

# Fit the ImageDataGenerator on the training data (optional but recommended)
datagen.fit(x_train)

# Example: Generate augmented images from the original training data
batch_size = 32
num_batches = len(x_train) // batch_size

# Create a generator that will yield augmented batches of images indefinitely
augmented_generator = datagen.flow(x_train, batch_size=batch_size)

# Iterate through the generator to generate augmented batches of images
for batch_index in range(num_batches):
    augmented_images = augmented_generator.next()
    # Optionally, you can use the augmented images for training your model
    # e.g., autoencoder.fit(augmented_images, augmented_images, ...)

# Note: The above loop generates augmented batches of images indefinitely, so you may want to set the appropriate number of steps for training.

# Remember to adjust the code based on your specific dataset and requirements.
