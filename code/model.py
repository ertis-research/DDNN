from tensorflow import keras
from tensorflow.keras import layers

def create_model( img_height, img_width, num_classes ):
    
    model_names = [ "model", "submodel_1", "submodel_2", "submodel_3", "submodel_4", 
    "submodel_5", "submodel_6", "submodel_7", "submodel_8" ]

    ## submodel_1
    layer_input = layers.Input(shape=(img_height, img_width, 3,))
    layer_batchN1 = layers.BatchNormalization()(layer_input)
    layer_conv1 = layers.Conv2D(32, (3, 3), activation='relu')(layer_batchN1)
    layer_maxpooling1 = layers.MaxPooling2D((2, 2))(layer_conv1)
    layer_dropout1 = layers.Dropout(0.4)(layer_maxpooling1)
    layer_flatten1 = layers.Flatten()(layer_dropout1)
    layer_d = layers.Dense(32, activation='relu')(layer_flatten1)
    layer_output1 = layers.Dense(num_classes, name="output1")(layer_d)
    submodel_1 = keras.Model(inputs=layer_input, outputs=[ layer_dropout1, layer_output1 ])

    ## submodel_2
    layer_input2 = layers.Input(shape=(89, 89, 32,))
    layer_batchN2 = layers.BatchNormalization()(layer_input2)
    layer_conv2 = layers.Conv2D(32, (3, 3), activation='relu')(layer_batchN2)
    layer_maxpooling2 = layers.MaxPooling2D((2, 2))(layer_conv2)
    layer_dropout2 = layers.Dropout(0.4)(layer_maxpooling2)
    layer_flatten2 = layers.Flatten()(layer_dropout2)
    layer_d2 = layers.Dense(32, activation='relu')(layer_flatten2)
    layer_output2 = layers.Dense(num_classes, name="output1")(layer_d2)
    submodel_2 = keras.Model(inputs=layer_input2, outputs=[layer_dropout2, layer_output2])

    ## submodel_3
    # layer_input3 = layers.Input(shape=( 43, 43, 25,))
    layer_input3 = layers.Input(shape=(43, 43, 32,))
    layer_conv3 = layers.Conv2D(64, (3, 3), activation='relu')(layer_input3)
    layer_batchN3 = layers.BatchNormalization()(layer_conv3)
    layer_maxpooling3 = layers.MaxPooling2D((2, 2))(layer_batchN3)
    layer_dropout3 = layers.Dropout(0.3)(layer_maxpooling3)
    layer_flatten3 = layers.Flatten()(layer_dropout3)
    layer_output3 = layers.Dense(num_classes, name="output3")(layer_flatten3)
    submodel_3 = keras.Model(inputs=layer_input3, outputs=[ layer_dropout3, layer_output3 ])

    ## submodel_4
    layer_input4 = layers.Input(shape=(20, 20, 64,))
    layer_conv4 = layers.Conv2D(64, (2, 2), activation='relu')(layer_input4)
    layer_batchN4 = layers.BatchNormalization()(layer_conv4)
    layer_maxpooling4 = layers.MaxPooling2D((2, 2))(layer_batchN4)
    layer_dropout4 = layers.Dropout(0.3)(layer_maxpooling4)
    layer_flatten4 = layers.Flatten()(layer_dropout4)
    layer_output4 = layers.Dense(num_classes, name="output4")(layer_flatten4)
    submodel_4 = keras.Model(inputs=layer_input4, outputs=[layer_dropout4, layer_output4])

    ## submodel_5
    # layer_input5 = layers.Input(shape=(9, 9, 5,))
    layer_input5 = layers.Input(shape=(9, 9, 64,))
    # layer_conv5 = layers.Conv2D(256, (2, 2), activation='relu')(layer_input5)
    # layer_batchN5 = layers.BatchNormalization()(layer_conv5)
    # layer_maxpooling5 = layers.MaxPooling2D((2, 2))(layer_batchN5)
    # layer_dropout5 = layers.Dropout(0.24)(layer_maxpooling5)
    # layer_flatten5 = layers.Flatten()(layer_dropout5)
    # layer_output5 = layers.Dense(num_classes, name="output5")(layer_flatten5)
    layer_flatten5 = layers.Flatten()(layer_input5)
    layer_dense5 = layers.Dense(1024, activation='relu')(layer_flatten5)
    # layer_batchN5 = layers.BatchNormalization()(layer_dense5)
    layer_dropout5 = layers.Dropout(0.2)(layer_dense5)
    layer_output5 = layers.Dense(num_classes, name="output6")(layer_dropout5)
    submodel_5 = keras.Model(inputs=layer_input5, outputs=[layer_dropout5, layer_output5])
    
    ## submodel_6
    # layer_input6 = layers.Input(shape=( 4, 4, 256,))
    layer_input6 = layers.Input(shape=( 1024, ))
    layer_flatten6 = layers.Flatten()(layer_input6)
    layer_dense1 = layers.Dense(512, activation='relu')(layer_flatten6)
    # layer_batchN6 = layers.BatchNormalization()(layer_dense1)
    layer_dropout6 = layers.Dropout(0.2)(layer_dense1)
    layer_output6 = layers.Dense(num_classes, name="output6")(layer_dropout6)
    submodel_6 = keras.Model(inputs=layer_input6, outputs=[layer_dropout6, layer_output6])

    ## submodel_7
    layer_input7 = layers.Input(shape=(512,))
    layer_dense2 = layers.Dense(256, activation='relu')(layer_input7)
    # layer_batchN7 = layers.BatchNormalization()(layer_dense2)
    layer_dropout7 = layers.Dropout(0.2)(layer_dense2)
    layer_output7 = layers.Dense(num_classes, name="output7")(layer_dropout7)
    submodel_7 = keras.Model(inputs=layer_input7, outputs=[layer_dropout7, layer_output7])

    ## submodel_8
    layer_input8 = layers.Input(shape=(256,))
    layer_dense3 = layers.Dense(64, activation='relu')(layer_input8)
    # layer_batchN8 = layers.BatchNormalization()(layer_dense3)
    layer_dropout8 = layers.Dropout(0.2)(layer_dense3)
    layer_output8 = layers.Dense(num_classes, name="output8")(layer_dropout8)
    submodel_8 = keras.Model(inputs=layer_input8, outputs=layer_output8)


    ## MODEL
    model_input = layers.Input(shape=( img_height, img_width, 3,))
    to_2, output1 = submodel_1(model_input)
    to_3, output2 = submodel_2(to_2)
    to_4, output3 = submodel_3(to_3)
    to_5, output4 = submodel_4(to_4)
    to_6, output5 = submodel_5(to_5)
    to_7, output6 = submodel_6(to_6)
    to_8, output7 = submodel_7(to_7)
    output8 = submodel_8(to_8)

    model = keras.Model(inputs=model_input, outputs=[output1, 
                                                     output2,
                                                     output3,
                                                     output4,
                                                     output5,
                                                     output6,
                                                     output7,
                                                     output8])

    opt = keras.optimizers.SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer = opt,
              loss=[ keras.losses.CategoricalCrossentropy(from_logits=True),
                    keras.losses.CategoricalCrossentropy(from_logits=True),
                    keras.losses.CategoricalCrossentropy(from_logits=True),
                    keras.losses.CategoricalCrossentropy(from_logits=True),
                    keras.losses.CategoricalCrossentropy(from_logits=True),
                    keras.losses.CategoricalCrossentropy(from_logits=True),
                    keras.losses.CategoricalCrossentropy(from_logits=True),
                    keras.losses.CategoricalCrossentropy(from_logits=True) ],
              loss_weights=[ 0.001, 0.001, 0.01, 0.01, 0.01, 0.01, 0.1, 0.1 ],
              metrics=[ 'accuracy' ])

    return model, submodel_1, submodel_2, submodel_3, \
            submodel_4, submodel_5, submodel_6, submodel_7, submodel_8, model_names