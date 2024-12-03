import tensorflow as tf
import tensorflow_datasets as tfds
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard

# Bug Fix at download: https://github.com/tensorflow/datasets/issues/3918
setattr(tfds.image_classification.cats_vs_dogs, '_URL',"https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip")
data, metadata = tfds.load('cats_vs_dogs', as_supervised=True, with_info=True)

SIZE=100
images, pet_types = [], []

# Resize images to SIZE x SIZE, convert to grayscale, and store in lists
for image, pet_type in data['train']:
    image = cv2.resize(image.numpy(), (SIZE, SIZE))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    images.append(image.reshape(SIZE, SIZE, 1))
    pet_types.append(pet_type)

# Normalizes all image from 0-255 to 0-1
images, pet_types = np.array(images).astype(float) / 255, np.array(pet_types)

images.shape

# Data augmentation to improve generalization by applying random transformations
datagen = ImageDataGenerator(rotation_range=30, width_shift_range=0.2, height_shift_range=0.2, shear_range=15, zoom_range=[0.7, 1.4], horizontal_flip=True, vertical_flip=True)

datagen.fit(images)

# IA Model
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(100, 100, 1)),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2, 2),

  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(120, activation='relu'),
  tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# Training data and test data

index = int(len(images) * 0.85)

images_train, images_test = images[:index], images[index:]
pet_types_train, pet_types_test = pet_types[:index], pet_types[index:]


data_train = datagen.flow(images_train, pet_types_train, batch_size=32)

tensorboard = TensorBoard(log_dir='logs-new/Artemisa')

model.fit(
    data_train,
    epochs=150, batch_size=32,
    validation_data=(images_test, pet_types_test),
    steps_per_epoch=int(np.ceil(len(pet_types_train) / float(32))),
    validation_steps=int(np.ceil(len(pet_types_test) / float(32))),
    callbacks=[tensorboard]
)

# Save

model.save('artemisa.h5')