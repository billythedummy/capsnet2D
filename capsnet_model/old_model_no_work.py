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

#old loss function
def weighted_vec_loss(y_true, y_pred):
    weight = 1626351.9411764706
    
    #n_classes = tf.keras.backend.int_shape(y_pred)[-2]
    #y_classes, y_bg = tf.split(y_true, [n_classes, 1], -2)
    #y_pred_classes, y_pred_bg = tf.split(y_pred, [n_classes, 1], -2)

    #probability for classifiation error
    #and vector difference for regression error

    #weighted log/ BCE loss for classification error
    y_classes_prob = y_true[:,:,:,:,0] #1 or 0
    y_pred_classes_prob = y_pred[:,:,:,:,0]
    class_prob_ones = tf.ones(tf.shape(y_classes_prob))
    class_loss = -(weight * tf.multiply(y_classes_prob, tf.log(y_pred_classes_prob))
                + tf.multiply(class_prob_ones - y_classes_prob, tf.log(class_prob_ones - y_pred_classes_prob)))
    class_loss = tf.reduce_mean(class_loss)
    
    #norm of vector difference/ MSE for regression error
    y_classes_vec = y_true[:,:,:,:,1:]
    y_pred_classes_vec = y_pred[:,:,:,:,1:]
    vec_diff = y_classes_vec - y_pred_classes_vec 
    vec_diff_squared = tf.reduce_sum(tf.square(vec_diff), -1)
    ##vec_diff_norm = tf.sqrt(vec_diff_squared)
    regr_loss = weight * y_classes_prob * vec_diff_squared #so 0 for background pixels
    regr_loss = tf.reduce_mean(regr_loss)
    
    loss = class_loss + regr_loss
    #print(loss)
    return loss

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
