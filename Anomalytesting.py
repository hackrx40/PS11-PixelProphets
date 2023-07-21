# Imports
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
import os
from PIL import Image
import matplotlib.pyplot as plt


# Load and preprocess the images from a directory
def load_images_from_dir(dir_path):
    image_paths = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.png') or f.endswith('.jpg')]
    images = []
    for image_path in image_paths:
        image = Image.open(image_path)
        image = image.convert('RGB')
        image = image.resize((224, 224))
        image = np.array(image)
        image = image.astype('float32') / 255.0
        images.append(image)
    images = np.array(images)
    return images

# Assuming you have two sets of images: normal UI screenshots and anomalous UI screenshots
normal_dir = 'C://Users/DELL/Desktop/Hackerx/UniformFont'
anomalous_dir = 'C://Users/DELL/Desktop/Hackerx/NonUnifromFont'

# Load and preprocess the normal and anomalous UI screenshots
normal_images = load_images_from_dir(normal_dir)
anomalous_images = load_images_from_dir(anomalous_dir)

# Create labels (0 for normal, 1 for anomalous)
y_normal = np.zeros(len(normal_images))
y_anomalous = np.ones(len(anomalous_images))

# Concatenate images and labels
x_all = np.concatenate((normal_images, anomalous_images), axis=0)
y_all = np.concatenate((y_normal, y_anomalous), axis=0)

# Split the data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(x_all, y_all, test_size=0.2, random_state=42)

# Data Augmentation (Apply data augmentation to the training data)
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
datagen.fit(x_train)

# Build the Convolutional Autoencoder model
input_shape = x_train[0].shape

def build_autoencoder():
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
    optimizer = Adam()
    autoencoder.compile(optimizer=optimizer, loss='mse')
    return autoencoder

# Create the autoencoder model
autoencoder = build_autoencoder()
def resize_images(images, target_size):
    resized_images = np.zeros((images.shape[0], target_size[0], target_size[1], images.shape[3]), dtype=np.uint8)
    for i in range(images.shape[0]):
        img = Image.fromarray(images[i])
        img = img.resize((target_size[0], target_size[1]), Image.ANTIALIAS)
        resized_images[i] = np.array(img)
    return resized_images

# Load data and resize images
x_train, x_val = train_test_split(images, test_size=0.2, random_state=42)
x_train_resized = resize_images(x_train, target_size=(224, 224))
x_val_resized = resize_images(x_val, target_size=(224, 224))
# Train the autoencoder with augmented data
autoencoder.fit(datagen.flow(x_train, x_train, batch_size=32), 
                steps_per_epoch=len(x_train) // 32, 
                epochs=100, 
                validation_data=(x_val, x_val))

# Anomaly Segmentation (Thresholding on reconstruction error for individual pixels)
# Get the reconstruction of validation data
reconstructed_images = autoencoder.predict(x_val)

# Calculate pixel-wise reconstruction error
pixelwise_errors = np.square(x_val - reconstructed_images)

# Compute the threshold as the mean plus a factor of the standard deviation of the pixel-wise errors
threshold_factor = 2.0
threshold = np.mean(pixelwise_errors) + threshold_factor * np.std(pixelwise_errors)

# Label anomalies based on the threshold
anomaly_labels = (pixelwise_errors > threshold).astype(int)

# Performance Evaluation
# Assuming y_val contains the true labels (0 for normal, 1 for anomalous)
y_true = y_val

# Flatten the anomaly_labels and y_true for computing metrics
anomaly_labels_flat = anomaly_labels.reshape(anomaly_labels.shape[0], -1)
y_true_flat = y_true.reshape(y_true.shape[0], -1)

# Calculate metrics for each pixel separately and then average them
precision = precision_score(y_true_flat, anomaly_labels_flat, average='micro')
recall = recall_score(y_true_flat, anomaly_labels_flat, average='micro')
f1 = f1_score(y_true_flat, anomaly_labels_flat, average='micro')
roc_auc = roc_auc_score(y_true_flat, anomaly_labels_flat, average='micro')

print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("ROC AUC:", roc_auc)
