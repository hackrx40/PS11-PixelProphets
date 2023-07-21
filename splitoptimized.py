split_ratio = 0.8
split_index = int(len(normal_images) * split_ratio)

x_train = np.concatenate((normal_images[:split_index], anomalous_images[:split_index]))
x_val = np.concatenate((normal_images[split_index:], anomalous_images[split_index:]))
