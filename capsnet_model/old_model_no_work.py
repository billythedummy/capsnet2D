import tensorflow as tf
from capsnet_model.caps_layer import CapsLayer2D

#and other deprecated stuff

def create_background_channel(capsule_channels, caps_dim):
    shape = capsule_channels[0].shape
    res = np.ones(shape)
    #res /= np.sqrt(caps_dim)
    for caps_channel in capsule_channels:
        res[np.nonzero(caps_channel)] = 0
    return res

class CapsNet(tf.keras.Model):
    # Input 4D tensor (samples/ batch size, rows, cols, channels)
    def __init__(self, routings=3, n_conv_layers=3, rows=255, cols=255):
        super(CapsNet, self).__init__(name = '')
        self.rows = rows
        self.cols = cols
        
        #Seems like keras layers can only be top level class attributes?
        #self.conv_layers = []
        #for i in range(n_conv_layers):
            #self.conv_layers.append(tf.keras.layers.SeparableConvolution2D((2**i)*6, (19, 19), padding="same"))
            #self.conv_layers.append(tf.keras.layers.BatchNormalization())
        
        
        self.conv2a = tf.keras.layers.SeparableConvolution2D(6, (19, 19), padding="same")
        self.bna = tf.keras.layers.BatchNormalization()

        self.conv2b = tf.keras.layers.SeparableConvolution2D(12, (19, 19), padding="same")
        self.bnb = tf.keras.layers.BatchNormalization()

        self.conv2c = tf.keras.layers.SeparableConvolution2D(24, (19, 19), padding="same")
        self.bnc = tf.keras.layers.BatchNormalization()
        
        self.concat = tf.keras.layers.Concatenate(axis=-1)

        self.capsule_reshape = tf.keras.layers.Reshape((self.rows, self.cols, 7, 6))

        self.capsa = CapsLayer2D(7, n_routings=routings)

        self.capsb = CapsLayer2D(2, n_routings=routings)

    def call(self, input_tensor, training=False):
        assert input_tensor.shape[1] == self.rows and input_tensor.shape[2] == self.cols, "Image dimensions (" + str(input_tensor.shape[1]) + ", " + str(input_tensor.shape[2]) + ") do not match network input image dimesions ("+ str(self.rows) + ", " + str(self.cols) + ")"
        #Seems like keras layers can only be top level class attributes?
        #conv_tensors = []
        #for i in range(0, len(self.conv_layers), 2):
            #if i == 0:
                #conv_output = self.conv_layers[i](input_tensor)
            #else:
                #conv_output = self.conv_layers[i](conv_tensors[i / 2 - 1])
            #conv_output = self.conv_layers[i+1](conv_output, training = training)
            #conv_tensors.append(tf.nn.relu(conv_output))
    
        a = self.conv2a(input_tensor)
        a = self.bna(a)
        a = tf.nn.relu(a)
        #print a.shape

        b = self.conv2b(a)
        b = self.bnb(b)
        b = tf.nn.relu(b)
        #print b.shape

        c = self.conv2c(b)
        c = self.bnc(c)
        c = tf.nn.relu(c)
        #print c.shape
        
        x = self.concat([a, b, c])
        #print x.shape

        x = self.capsule_reshape(x)
        #print x.shape
        
        x = self.capsa(x)
        #print x.shape

        x = self.capsb(x)
        #print x.shape
        return x
