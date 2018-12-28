import tensorflow as tf

class CapsLayer2D(tf.keras.layers.Layer):
    def __init__(self,
                 n_caps,
                 batch_size=1,
                 n_routings=3,
                 rows=None, #same as input by default
                 cols=None, #same as input by default
                 caps_dim=None, #same dimension as input capsules by default
                 kernel_initializer='glorot_uniform',
                 b_initializer='zeros',
                 **kwargs):
        super(CapsLayer2D, self).__init__(**kwargs)
        self.n_caps = n_caps
        self.batch_size = batch_size
        self.n_routings = n_routings
        self.rows = rows
        self.cols = cols
        self.caps_dim = caps_dim
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.b_initializer = tf.keras.initializers.get(b_initializer)

    def build(self, input_shape):
        #input tensor shape: [None, rows, cols, n_capsules_per_pixel, capsule_dimension] 
        self.input_rows = int(input_shape[1])
        self.input_cols = int(input_shape[2])
        self.input_caps_dim = int(input_shape[-1])
        self.n_input_caps = int(input_shape[3])
        if self.rows is None:
            self.rows = self.input_rows
        if self.cols is None:
            self.cols = self.input_cols
        if self.caps_dim is None:
            self.caps_dim = int(self.input_caps_dim)
        self.W = self.add_weight(shape=[int(self.n_caps),
                                        int(self.n_input_caps),
                                        int(self.input_caps_dim),
                                        int(self.caps_dim)],
                                 initializer=self.kernel_initializer,
                                 name="W")
        self.built = True

    def call(self, inputs, training=None):
        weights = tf.expand_dims(self.W, 0)
        weights = tf.expand_dims(weights, 0)
        weights = tf.expand_dims(weights, 0)
        #[1, 1, 1, n_caps, n_input_caps, input_caps_dim, caps_dim]
        w_multiples = tf.constant(
                    [self.batch_size, self.input_rows, self.input_cols,
                     1, 1, 1, 1])
        weights = tf.tile(weights, w_multiples)
        #[None, rows, cols, n_caps, n_input_caps, input_caps_dim, caps_dim]
        #last 4 dimensions is the actual weights
        #(last 2 is the transformation matrix) that is copied
        #across batch, rows and cols
        
        input_proc= tf.expand_dims(inputs, -3)
        #[None, rows, cols, 1, n_input_caps, caps_dim]
        input_proc = tf.expand_dims(input_proc, -2)
        #[None, rows, cols, 1, n_input_caps, 1, caps_dim]
        in_multiples = tf.constant([1, 1, 1, self.n_caps, 1, 1, 1])
        input_proc = tf.tile(input_proc, in_multiples)
        #[None, rows, cols, n_caps, n_input_caps, 1, caps_dim]
        #last 2 dimensions is the 1 x caps_dim row vectors/ capsules

        # this is against convention with row vectors and v * M instead of col vectors and M * v
        res = tf.matmul(input_proc, weights)
        res = tf.squeeze(res, [5])
        #[None, rows, cols, n_caps, n_input_caps, caps_dim]

        #The n_input_caps dimension is removed by having a weighted sum of the
        #vectors to determine the new capsule, determined by routing by agreement

        b = tf.nn.softmax(tf.zeros([self.batch_size,
                                    self.input_rows,
                                    self.input_cols,
                                    self.n_caps,
                                    self.n_input_caps]))
        #[None, rows, cols, n_caps, n_input_caps]
        for i in range(self.n_routings - 1):
            b_tiled = tf.expand_dims(b, -1)
            b_tiled = tf.tile(b_tiled, tf.constant([1, 1, 1, 1, 1, self.caps_dim]))
            res_pred = tf.multiply(b_tiled, res)
            res_pred = tf.reduce_sum(res_pred, -2)
            #[None, rows, cols, n_caps, caps_dim]
            res_pred = squash(res_pred)
            res_pred = tf.expand_dims(res_pred, -2)
            res_pred = tf.tile(res_pred, tf.constant([1, 1, 1, 1, self.n_input_caps, 1]))
            #copied along input_caps axis [None, row, cols, n_caps, n_input_caps, caps_dim]
            #update weights step:
            b += tf.reduce_sum(tf.multiply(res, res_pred), -1)             

        b_tiled = tf.expand_dims(b, -1)
        b_tiled = tf.tile(b_tiled, tf.constant([1, 1, 1, 1, 1, self.caps_dim]))
        res = tf.multiply(b_tiled, res)
        res = tf.reduce_sum(res, -2)
        return squash(res)
    
def squash(vectors, axis=-1):
    squared_norm = tf.reduce_sum(tf.square(vectors), axis, keepdims=True)
    scale = tf.sqrt(squared_norm) / (1 + squared_norm)
    return scale * vectors
