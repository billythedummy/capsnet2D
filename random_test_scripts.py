if __name__ == "__main__":
    # Testing model compilation and run through
    from capsnet_model.model import CapsNet
    import numpy as np
    import tensorflow as tf
    sess = tf.Session()
    with sess.as_default():
        model = CapsNet()
        model.load_weights("../capsnet_data/trained_model.h5")
        model.summary()
        x = tf.convert_to_tensor(np.ones([4, 255, 255, 3]).astype(np.float32))
        pred = model(x)
        print pred.shape
        print pred.eval().shape
    
    # Testing visutils
    '''
    from visutils import draw_on
    import matplotlib as mpl
    mpl.use('TkAgg')
    import matplotlib.pyplot as plt
    import glob
    from utils.to_tfrecord import to_tensors
    import numpy as np
    direc = "../capsnet_data/data/raw"
    csv_files = glob.glob(direc+"/*.csv")
    for csv_file in csv_files:
        splitted = csv_file.split(".")[-2]
        name = splitted.split("/")[-1]
        x, y = to_tensors(direc, name, out_width=255, out_height=255)
        x_exp = np.expand_dims(x, 0)
        y = np.expand_dims(y, 0)
        fig, ax = plt.subplots()
        draw_on(x_exp, y, ax)
        x = x.astype(np.uint8)
        plt.imshow(x)
        plt.show()
    '''

    '''
    #Loss function sanity check
    import tensorflow as tf

    from utils.to_tfrecord import to_tensors
    import numpy as np
    x, y = to_tensors("../capsnet_data/data/raw", "red_drive_30", out_width=255, out_height=255)
    y = np.expand_dims(y, 0)
    sess = tf.Session()
    with sess.as_default():
        l = weighted_vec_loss(y, y).eval()
    '''
