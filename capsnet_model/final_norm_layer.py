import tensorflow as tf

class FinalNormLayer(tf.keras.layers.Layer):
    def __init__(self, caps_axis=-1, **kwargs):
        super(FinalNormLayer, self).__init__(**kwargs)
        self.caps_axis = caps_axis

    def build(self, input_shape):
        self.built = True

    def call(self, inputs, training=None):
        mag_sq = tf.reduce_sum(tf.square(inputs), axis=-1)
        return tf.sqrt(mag_sq + tf.keras.backend.epsilon(), name="capsnet_output")


        
