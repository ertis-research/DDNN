import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

def create_model( img_height, img_width, num_classes ):
    
    # data_augmentation = Sequential(
    #     [
    #         layers.experimental.preprocessing.RandomFlip("horizontal", 
    #                                                     input_shape=(img_height, 
    #                                                                 img_width,
    #                                                                 3)),
    #         layers.experimental.preprocessing.RandomRotation(0.1),
    #         layers.experimental.preprocessing.RandomZoom(0.1),
    #     ]
    # )

    # model = Sequential([
    #     data_augmentation,
    #     layers.experimental.preprocessing.Rescaling(1./255),
    #     layers.Conv2D(16, 3, padding='same', activation='relu'),
    #     layers.MaxPooling2D(),
    #     layers.Conv2D(32, 3, padding='same', activation='relu'),
    #     layers.MaxPooling2D(),
    #     layers.Conv2D(64, 3, padding='same', activation='relu'),
    #     layers.MaxPooling2D(),
    #     layers.Dropout(0.2),
    #     layers.Flatten(),
    #     layers.Dense(128, activation='relu'),
    #     layers.Dense(num_classes)
    # ])

    ## EDGE MODEL
    layer_input = layers.Input(shape=(img_height, img_width, 3,))

    layer_conv1 = layers.Conv2D(96, (3, 3), activation='relu' )(layer_input)
    layer_batchN1 = layers.BatchNormalization()(layer_conv1)
    layer_maxpooling1 = layers.MaxPooling2D((2, 2))(layer_batchN1)
    layer_dropout1 = layers.Dropout(0.5)(layer_maxpooling1)

    layer_flatten1 = layers.Flatten()(layer_dropout1)
    # first_layer_output = layers.Dense(num_classes, activation="softmax", name="output1")(layer_flatten1)
    first_layer_output = layers.Dense(num_classes, name="output1")(layer_flatten1)

    edge_model = keras.Model(inputs=layer_input, outputs=[layer_dropout1, first_layer_output])
    # edge_model.summary()

    ## FOG MODEL
    layer_input2 = layers.Input(shape=(89, 89, 96,))
    layer_conv2 = layers.Conv2D(256, (3, 3), activation='relu')(layer_input2)
    layer_batchN2 = layers.BatchNormalization()(layer_conv2)
    layer_maxpooling2 = layers.MaxPooling2D((2, 2))(layer_batchN2)
    layer_dropout2 = layers.Dropout(0.5)(layer_maxpooling2)

    layer_flatten2 = layers.Flatten()(layer_dropout2)
    # second_layer_output = layers.Dense(num_classes, activation="softmax", name="output2")(layer_flatten2)
    second_layer_output = layers.Dense(num_classes, name="output2")(layer_flatten2)

    fog_model = keras.Model(inputs=layer_input2, outputs=[layer_dropout2, second_layer_output])
    # fog_model.summary()

    ## CLOUD MODEL
    layer_input3 = layers.Input(shape=(43, 43, 256,))

    layer_conv5 = layers.Conv2D(256, (3, 3), activation='relu')(layer_input3)
    layer_batchN5 = layers.BatchNormalization()(layer_conv5)
    layer_maxpooling5 = layers.MaxPooling2D((2, 2))(layer_batchN5)
    layer_dropout5 = layers.Dropout(0.5)(layer_maxpooling5)

    layer_flatten2 = layers.Flatten()(layer_dropout5)

    layer_dense1 = layers.Dense(9216, activation='relu')(layer_flatten2)
    layer_batchN6 = layers.BatchNormalization()(layer_dense1)
    layer_dropout6 = layers.Dropout(0.5)(layer_batchN6)

    layer_dense2 = layers.Dense(4096, activation='relu')(layer_dropout6)
    layer_batchN7 = layers.BatchNormalization()(layer_dense2)
    layer_dropout7 = layers.Dropout(0.5)(layer_batchN7)

    layer_dense3 = layers.Dense(4096, activation='relu')(layer_dropout7)
    layer_batchN8 = layers.BatchNormalization()(layer_dense3)
    layer_dropout8 = layers.Dropout(0.5)(layer_batchN8)

    # layer_output = layers.Dense( num_classes, activation="softmax", name="output3")(layer_dropout8)
    layer_output = layers.Dense( num_classes, name="output3")(layer_dropout8)

    cloud_model = keras.Model(inputs=layer_input3, outputs=layer_output)
    # cloud_model.summary()

    model_input = layers.Input(shape=( img_height, img_width, 3,))
    edge_to_fog, edge = edge_model(model_input)
    fog_to_cloud, fog = fog_model(edge_to_fog)
    cloud = cloud_model(fog_to_cloud)

    model = keras.Model(inputs=model_input, outputs=[edge, fog, cloud])

    model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    # opt = keras.optimizers.SGD(lr=0.001, momentum=0.9)
    # model.compile(optimizer=opt,
    #             loss=keras.losses.CategoricalCrossentropy(),
    #             metrics=["accuracy"])

    return model