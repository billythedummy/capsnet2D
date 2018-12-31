import tensorflow as tf
from capsnet_model.caps_layer import CapsLayer2D

def CapsNet(input_shape=(255, 255, 3),
            n_class=1,
            n_routings=3,
            batch_size=1):

    in_name = "Input_Batch_size_" + str(batch_size)
    input_layer = tf.keras.layers.Input(shape=input_shape, name="input")
    
    conv1 = tf.keras.layers.SeparableConvolution2D(6, (19, 19), padding="same", name="conv1_19x19x6")(input_layer)
    bn1 = tf.keras.layers.BatchNormalization(name="bn1")(conv1)
    relu1 = tf.keras.layers.Activation('relu', name="relu1")(bn1)

    conv2 = tf.keras.layers.SeparableConvolution2D(12, (19, 19), padding="same", name="conv2_19x19x12")(relu1)
    bn2 = tf.keras.layers.BatchNormalization(name="bn2")(conv2)
    relu2 = tf.keras.layers.Activation('relu', name="relu2")(bn2)

    conv3 = tf.keras.layers.SeparableConvolution2D(22, (19, 19), padding="same", name="conv3_19x19x24")(relu2)
    bn3 = tf.keras.layers.BatchNormalization(name="bn3")(conv3)
    relu3 = tf.keras.layers.Activation('relu', name="relu3")(bn3)

    concat = tf.keras.layers.concatenate([relu1, relu2, relu3], axis=-1, name="Channel_Concat")

    reshape = tf.keras.layers.Reshape((input_shape[0], input_shape[1], 8, 5), name="Capsule_Reshape")(concat)

    capsa = CapsLayer2D(4, batch_size=batch_size, n_routings=n_routings, name="Capsule_Layer_7")(reshape)

    capsb = CapsLayer2D(n_class + 1, batch_size=batch_size, n_routings=n_routings, last=True, name="Final_Capsules")(capsa)

    model = tf.keras.Model(input_layer, capsb)
    return model
