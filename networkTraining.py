# Neural network training and testing
# author: Ellinoora Hetemaa

import numpy as np
import os
import tensorflow as tf

import matplotlib.pyplot as plt
from tensorflow import keras
from keras import layers

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def download_data(folder_dir):
    train_ds = keras.preprocessing.image_dataset_from_directory(folder_dir,
                                                                image_size=(64, 64),
                                                                batch_size=32,
                                                                seed=123,
                                                                validation_split=0.2,
                                                                subset="training")
    validation_ds = keras.preprocessing.image_dataset_from_directory(folder_dir,
                                                                     image_size=(64, 64),
                                                                     batch_size=32,
                                                                     seed=123,
                                                                     validation_split=0.2,
                                                                     subset="validation")

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    validation_ds = validation_ds.cache().prefetch(buffer_size=AUTOTUNE)

    normalization_layer = layers.Rescaling(1. / 255)

    normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    image_batch, labels_batch = next(iter(normalized_ds))
    first_image = image_batch[0]
    # Notice the pixel values are now in `[0,1]`.
    print(np.min(first_image), np.max(first_image))

    return normalized_ds, validation_ds


def train(train_ds, validation_ds):

    model = keras.Sequential()
    model.add(layers.Flatten(input_shape=(64, 64, 3)))
    model.add(layers.Dense(100, activation='relu'))
    model.add(layers.Dense(100, activation='relu'))
    model.add(tf.keras.layers.Dense(2, activation='relu'))

    model.compile(optimizer='SGD',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.summary()

    epochs = 10
    history = model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=epochs
    )
    plt.plot(history.history['loss'])
    plt.show()

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()
    print("moi")


# Press the green button in the gutter to run the script.
tr_ds, val_ds = download_data('GTSRB_subset_2')
train(tr_ds, val_ds)

