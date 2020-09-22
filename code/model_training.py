# REFERENCES:
# https://www.tensorflow.org/tutorials/images/classification

import numpy as np
import matplotlib as plt
import pathlib
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

from model import model 

BATCH_SIZE = 32
IMG_HEIGHT = 180
IMG_WIDTH  = 180

EPOCHS = 15

# dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
# data_dir = keras.utils.get_file( 'flower_photos', origin=dataset_url, untar=True )
# data_dir = pathlib.Path( data_dir )
data_dir = pathlib.Path( "data/flower_photos" )

train_ds = keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split = 0.2,
    subset="training",
    seed=123,
    image_size=( IMG_HEIGHT, IMG_WIDTH ),
    batch_size=BATCH_SIZE
)

print()

val_ds = keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=( IMG_HEIGHT, IMG_WIDTH ),
  batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
print( class_names )

train_ds = train_ds.cache().shuffle( 1000 ).prefetch( buffer_size = tf.data.experimental.AUTOTUNE )
val_ds = val_ds.cache().prefetch( buffer_size = tf.data.experimental.AUTOTUNE )
normalization_layer = layers.experimental.preprocessing.Rescaling( 1./255 )

normalized_ds = train_ds.map( lambda x, y: (normalization_layer(x), y ))
image_batch, labels_batch = next( iter(normalized_ds) )
first_image = image_batch[0]
print( np.min(first_image), np.max(first_image) )

##############
# CREATE THE MODEL
##############

model = model( IMG_HEIGHT, IMG_WIDTH, len( class_names ))

model.summary()

history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs= EPOCHS
)

### TRAINING RESULTS
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

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