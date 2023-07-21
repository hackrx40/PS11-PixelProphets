def create_data_augmentation(datagen, images, batch_size=32):
    num_batches = len(images) // batch_size
    augmented_data = []
    for batch_index in range(num_batches):
        augmented_images = datagen.flow(images[batch_index*batch_size:(batch_index+1)*batch_size], batch_size=batch_size).next()
        augmented_data.append(augmented_images)
    return np.concatenate(augmented_data, axis=0)

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

x_train_augmented = create_data_augmentation(datagen, x_train)