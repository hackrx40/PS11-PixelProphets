# Class Imbalance Handling (e.g., oversampling, synthetic data generation)

# Normalize images between 0 and 1 (if not done previously)
x_train = x_train.astype('float32') / 255.0
x_val = x_val.astype('float32') / 255.0

# Reshape the class labels to match the number of samples
y_train = y_train.reshape(-1, 1)
y_val = y_val.reshape(-1, 1)

# Create the SMOTE object and resample the data
# ...

# Train the model on the resampled data (same as the code provided previously)
# ...
