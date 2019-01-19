import tensorflow as tf
from capsnet_model.caps_layer import CapsLayer2D
from capsnet_model.final_norm_layer import FinalNormLayer

def CapsNet(input_shape=(255, 255, 3),
            n_class=1,
            n_routings=3):

    input_layer = tf.keras.layers.Input(shape=input_shape, name="capsnet_input")
    
    encoder_layers = encoder(input_layer)

    reshape = tf.keras.layers.Reshape((input_shape[0], input_shape[1], 8, 4),
                                      name="Capsule_Reshape")(encoder_layers)

    capsa = CapsLayer2D(4,
                        n_routings=n_routings,
                        caps_dim=8,
                        name="Capsule_Layer_7")(reshape)

    capsb = CapsLayer2D(n_class,
                        n_routings=n_routings,
                        caps_dim=24,
                        name="Final_Capsules")(capsa)

    finalnorm = FinalNormLayer(name="final_norm")(capsb)
    
    model = tf.keras.Model(input_layer, finalnorm)
    return model

def encoder(input_layer):
    conv1 = tf.keras.layers.Conv2D(32, (5, 5),
                                   strides=(1, 1),
                                   padding="same",
                                   kernel_initializer='he_normal',
                                   #kernel_regularizer=l2(l2_reg),
                                   name='conv1')(input_layer)
    #conv1 = BatchNormalization(axis=3, momentum=0.99, name='bn1')(conv1) # Tensorflow uses filter format [filter_height, filter_width, in_channels, out_channels], hence axis = 3
    conv1 = tf.keras.layers.ELU(name='elu1')(conv1)

    conv2 = tf.keras.layers.Conv2D(48, (3, 3),
                                   strides=(1, 1),
                                   padding="same",
                                   kernel_initializer='he_normal',
                                   #kernel_regularizer=l2(l2_reg),
                                   name='conv2')(conv1)
    #conv2 = BatchNormalization(axis=3, momentum=0.99, name='bn2')(conv2)
    conv2 = tf.keras.layers.ELU(name='elu2')(conv2)

    conv3 = tf.keras.layers.Conv2D(64, (3, 3),
                                   strides=(1, 1),
                                   padding="same",
                                   kernel_initializer='he_normal',
                                   #kernel_regularizer=l2(l2_reg),
                                   name='conv3')(conv2)
    #conv3 = BatchNormalization(axis=3, momentum=0.99, name='bn3')(conv3)
    conv3 = tf.keras.layers.ELU(name='elu3')(conv3)

    conv4 = tf.keras.layers.Conv2D(64, (3,3),
                                   strides = (1,1),
                                   padding = "same",
                                   kernel_initializer='he_normal',
                                   #kernel_regularizer=l2(l2_reg),
                                   name='conv4')(conv3)
    #conv4 = BatchNormalization(axis=3, momentum=0.99, name='bn4')(conv4)
    conv4 = tf.keras.layers.ELU(name='elu4')(conv4)

    conv5 = tf.keras.layers.Conv2D(48, (3, 3),
                                   strides=(1, 1),
                                   padding="same",
                                   kernel_initializer='he_normal',
                                   #kernel_regularizer=l2(l2_reg),
                                   name='conv5')(conv4)
    #conv5 = BatchNormalization(axis=3, momentum=0.99, name='bn5')(conv5)
    conv5 = tf.keras.layers.ELU(name='elu5')(conv5)

    conv6 = tf.keras.layers.Conv2D(48, (3, 3),
                                   strides=(1, 1),
                                   padding="same",
                                   kernel_initializer='he_normal',
                                   #kernel_regularizer=l2(l2_reg),
                                   name='conv6')(conv5)
    #conv6 = BatchNormalization(axis=3, momentum=0.99, name='bn6')(conv6)
    conv6 = tf.keras.layers.ELU(name='elu6')(conv6)

    conv7 = tf.keras.layers.Conv2D(32, (3, 3),
                                   strides=(1, 1),
                                   padding="same",
                                   kernel_initializer='he_normal',
                                   #kernel_regularizer=l2(l2_reg),
                                   name='conv7')(conv6)
    #conv7 = BatchNormalization(axis=3, momentum=0.99, name='bn7')(conv7)
    return tf.keras.layers.ELU(name='elu7')(conv7)

def old_encoder(input_layer):    
    conv1 = tf.keras.layers.SeparableConvolution2D(6,
                                                   (19, 19),
                                                   padding="same",
                                                   name="conv1_19x19x6")(input_layer)
    #bn1 = tf.keras.layers.BatchNormalization(name="bn1")(conv1)
    elu1 = tf.keras.layers.Activation('elu', name="elu1")(conv1)

    conv2 = tf.keras.layers.SeparableConvolution2D(12,
                                                   (19, 19),
                                                   padding="same",
                                                   name="conv2_19x19x12")(elu1)
    #bn2 = tf.keras.layers.BatchNormalization(name="bn2")(conv2)
    elu2 = tf.keras.layers.Activation('elu', name="elu2")(conv2)

    conv3 = tf.keras.layers.SeparableConvolution2D(22,
                                                   (19, 19),
                                                   padding="same",
                                                   name="conv3_19x19x24")(elu2)
    #bn3 = tf.keras.layers.BatchNormalization(name="bn3")(conv3)
    elu3 = tf.keras.layers.Activation('elu', name="elu3")(conv3)

    return tf.keras.layers.concatenate([elu1, elu2, elu3],
                                         axis=-1,
                                         name="Channel_Concat")
