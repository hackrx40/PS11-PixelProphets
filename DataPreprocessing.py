# Normalize images between 0 and 1
normal_images = normal_images.astype('float32') / 255.0
anomalous_images = anomalous_images.astype('float32') / 255.0

# Split the data into training and validation sets
split_ratio = 0.8
split_index = int(len(normal_images) * split_ratio)

x_train = np.concatenate((normal_images[:split_index], anomalous_images[:split_index]))
x_val = np.concatenate((normal_images[split_index:], anomalous_images[split_index:]))
