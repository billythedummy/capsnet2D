import tensorflow as tf

class CapsLayer2D(tf.keras.layers.Layer):
    def __init__(self,
                 n_caps,
                 n_routings=3,
                 rows=None, #same as input by default
                 cols=None, #same as input by default
                 caps_dim=None, #same dimension as input capsules by default
                 kernel_initializer='glorot_uniform',
                 **kwargs):
        super(CapsLayer2D, self).__init__(**kwargs)
        self.n_caps = n_caps
        self.n_routings = n_routings
        self.rows = rows
        self.cols = cols
        self.caps_dim = caps_dim
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)

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
        self.batch_size = tf.Variable(1, name="batch_size")
        self.built = True

    def call(self, inputs, training=None):
        batch_size = tf.assign(self.batch_size, tf.shape(inputs)[0])
        weights = tf.expand_dims(self.W, 0)
        weights = tf.expand_dims(weights, 0)
        weights = tf.expand_dims(weights, 0)
        #[1, 1, 1, n_caps, n_input_caps, input_caps_dim, caps_dim]
        weights = tf.tile(weights,
                          [batch_size, self.input_rows, self.input_cols, 1, 1, 1, 1])
        #[None, rows, cols, n_caps, n_input_caps, input_caps_dim, caps_dim]
        #last 4 dimensions is the actual weights
        #(last 2 is the transformation matrix) that is copied
        #across batch, rows and cols
        
        input_proc= tf.expand_dims(inputs, -3)
        #[None, rows, cols, 1, n_input_caps, input_caps_dim]
        input_proc = tf.expand_dims(input_proc, -2)
        #[None, rows, cols, 1, n_input_caps, 1, input_caps_dim]
        in_multiples = tf.constant([1, 1, 1, self.n_caps, 1, 1, 1])
        input_proc = tf.tile(input_proc, in_multiples)
        #[None, rows, cols, n_caps, n_input_caps, 1, input_caps_dim]
        #last 2 dimensions is the 1 x input_caps_dim row vectors/ capsules

        # this is against convention with row vectors and v * M instead of col vectors and M * v
        predicted = tf.matmul(input_proc, weights)
        predicted = tf.squeeze(predicted, [5])
        #[None, rows, cols, n_caps, n_input_caps, caps_dim]
        
        #The n_input_caps dimension is removed by having a weighted sum of the
        #vectors in that dimension to determine the new capsule, determined by routing by agreement
        b_raw = tf.zeros([batch_size,
                          self.input_rows,
                          self.input_cols,
                          self.n_caps,
                          self.n_input_caps])
        #[None, rows, cols, n_caps, n_input_caps]

        cond = lambda o, i, p, b: i < self.n_routings # see route_by_agreement param names

        def route_by_agreement(prev_output, counter, predicted, b_raw):
            b = tf.nn.softmax(b_raw)
            b_exp = tf.expand_dims(b, -2)

            out = tf.matmul(b_exp, predicted)
            #b_exp: [None, rows, cols, n_caps, 1, n_input_caps]
            #predicted: [None, rows, cols, n_caps, n_input_caps, caps_dim]

            out = tf.squeeze(out, [4])
            out = squash(out)
            #out: [None, rows, cols, n_caps, caps_dim]
            out_update_b = tf.expand_dims(out, -1)
            #out_update_b: [None, rows, cols, n_caps, caps_dim, 1]
            
            #update weights step:
            #b_raw: [None, row, cols, n_caps, n_input_caps]            
            b_raw += tf.squeeze(tf.matmul(predicted, out_update_b), [-1])
            return (out, counter + 1, predicted, b_raw)

        after_routing = tf.while_loop(cond,
                                      route_by_agreement,
                                      loop_vars=(tf.zeros([batch_size, self.input_rows, self.input_cols, self.n_caps, self.caps_dim]),
                                                 tf.constant(0),
                                                 predicted,
                                                 b_raw))
        return after_routing[0]
    
def squash(s, axis=-1, epsilon=tf.keras.backend.epsilon(), name=None):
    squared_norm = tf.reduce_sum(tf.square(s), axis=axis,
                                 keepdims=True)
    safe_norm = tf.sqrt(squared_norm + epsilon)
    squash_factor = squared_norm / (1. + squared_norm)
    unit_vector = s / safe_norm
    return squash_factor * unit_vector
    
