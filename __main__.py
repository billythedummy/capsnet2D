import tensorflow as tf
import numpy as np

from capsnet_model.model import CapsNet
from capsnet_model.caps_layer import squash

from utils.to_tfrecord import parse_fn_caps_tfrecord

def weighted_vec_loss(y_true, y_pred):
    scale = tf.keras.backend.int_shape(y_pred)[1] * tf.keras.backend.int_shape(y_pred)[2] #For scaling purposes

    n_classes = tf.keras.backend.int_shape(y_pred)[-2] - 1
    y_classes, y_bg = tf.split(y_true, [n_classes, 1], -2)
    y_pred_classes, y_pred_bg = tf.split(y_pred, [n_classes, 1], -2)


    #background only uses dimension[0] probability for comparison
    #log loss for background classification error
    #y_bg_prob = y_bg[:,:,:,:,0]  #1 or 0
    #y_pred_bg_prob = y_pred_bg[:,:,:,:,0]
    #bg_ones = tf.ones(tf.shape(y_bg_prob)) #[None, h, w, 1]
    #binary cross entropy for background
    #bg_loss = -(tf.multiply(y_bg_prob, tf.log(y_pred_bg_prob))
                #+ tf.multiply(bg_ones - y_bg_prob, tf.log(bg_ones - y_pred_bg_prob)))
    #bg_loss = tf.reduce_sum(bg_loss)
    #bg_loss /= scale

    #classes uses both probability for classifiation error
    #and vector difference for regression error

    #log loss for classification error
    y_classes_prob = y_classes[:,:,:,:,0] #1 or 0
    y_pred_classes_prob = y_pred_classes[:,:,:,:,0]
    class_prob_ones = tf.ones(tf.shape(y_classes_prob))
    class_loss = -(tf.multiply(y_classes_prob, tf.log(y_pred_classes_prob))
                + tf.multiply(class_prob_ones - y_classes_prob, tf.log(class_prob_ones - y_pred_classes_prob)))
    class_loss = tf.reduce_sum(class_loss)
    
    #log of magnitude of squashed vector difference for regression error
    y_classes_vec = y_classes[:,:,:,:,1:]
    y_pred_classes_vec = y_pred_classes[:,:,:,:,1:]
    vec_diff = y_classes_vec - y_pred_classes_vec #squash
    vec_diff_norm = tf.sqrt(tf.reduce_sum(tf.square(vec_diff), -1))
    #vec_diff_ones = tf.ones(tf.shape(vec_diff_norm)) #[None, h, w, n_classes, caps_dim]
    #regr_loss = -tf.log(vec_diff_ones - vec_diff_norm) #we want the vec diff to be 0
    vec_diff_norm = y_classes_prob * vec_diff_norm
    regr_loss = tf.reduce_sum(vec_diff_norm)#tf.reduce_sum(regr_loss)
    
    loss = regr_loss + class_loss #equal weighting
    #print(loss)
    return loss

def tfrecord_generator(filenames, batch_size):
    dataset = tf.data.TFRecordDataset(filenames, compression_type='GZIP')
    dataset = dataset.map(parse_fn_caps_tfrecord, num_parallel_calls=8)
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
    dataset = dataset.repeat()
    return dataset

def train_generator(x, y, batch_size):
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
    generator = train_datagen.flow(x, y, batch_size=batch_size)
    while 1:
        x_batch, y_batch = generator.next()
        #print("Yielding new batch...")
        yield (x_batch, y_batch)

def train(model, args, data=None):
    tb = tf.keras.callbacks.TensorBoard(log_dir=args.working_dir + "/tb-logs", batch_size=args.batch_size) 

    checkpt = tf.keras.callbacks.ModelCheckpoint(args.working_dir + "/chkpts/chkpt-{epoch:02d}.h5",
                                                 save_best_only=True, #True can only work if there is a validation/ test set
                                                 save_weights_only=True,
                                                 verbose=1)
    
    lr_decay = tf.keras.callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (args.lr_decay ** epoch))

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=args.lr),
                  loss=weighted_vec_loss,
                  metrics={'capsnet': 'accuracy'})

    if data is not None: #for local batches
        (x_train, y_train), (x_test, y_test) = data
        model.fit_generator(generator=train_generator(x_train, y_train, args.batch_size),
                            steps_per_epoch=max(1, int(y_train.shape[0] / args.batch_size)),
                            epochs=args.epochs,
                            validation_data=[x_test, y_test],
                            callbacks=[tb,checkpt,lr_decay],
                            verbose=1)
    else: #for TFRecord
        training_set = tfrecord_generator(args.working_dir + "/" + args.train_path, args.batch_size)
        eval_set = tfrecord_generator(args.working_dir + "/" + args.eval_path, args.batch_size)
        #bec I know i have 5 images in my eval set rn. Really gotta fix these hardcoded sample size bs
        #but eval_set batch size must be equal to training set batch size bec capsule layer dimensions is hardcoded at init rn
        model.fit(training_set.make_one_shot_iterator(),
                  steps_per_epoch=max(1, int(20 / args.batch_size)), #lol shit how to determine dataset size from tfrecord file
                  callbacks=[tb, checkpt, lr_decay],
                  epochs=args.epochs,
                  validation_data=eval_set.make_one_shot_iterator(),
                  validation_steps=1,
                  verbose=1)

    model.save_weights(args.working_dir + "/trained_model.h5")
    print("Trained model saved to working directory!")

    return model
                  
if __name__ == "__main__":
    
    import argparse
    import glob
    parser = argparse.ArgumentParser(description="2D Capsule Network. Run from inside the main repo.")
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--batch_size", default=5, type=int)
    parser.add_argument("--lr", default=0.001, type=float, help="Initial learning rate")
    parser.add_argument("--lr_decay", default=0.95, type=float, help="Exponent base for decay. Between 0-1")
    parser.add_argument("--routings", default=3, type=int, help="Number of routings for capsules")
    parser.add_argument("--from_chkpt", default=0, type=int)
    parser.add_argument("--from_saved", default=0, type=int)
    parser.add_argument("--working_dir", default="../capsnet_data", type=str)
    parser.add_argument("--train_path", default="data/tfrecord/train.tfrecords", type=str)
    parser.add_argument("--eval_path", default="data/tfrecord/test.tfrecords", type=str)

    args = parser.parse_args()

    if args.from_chkpt and args.from_saved:
        raise Exception("Error: You can only choose either from_chkpt or from_saved")

    model = CapsNet(batch_size=args.batch_size)

    if args.from_chkpt or args.from_saved:
        if args.from_chkpt:
            chkpts = glob.glob(args.working_dir + "/chkpts/chkpt-*.h5")
        else:
            chkpts = glob.glob(args.working_dir + "/trained_model.h5")
        if len(chkpts) > 0:
            path = sorted(chkpts)[-1]
            model.load_weights(path)
            print("Weights loaded from " + path)
        else:
            print("No existing saved model/ checkpoint found, reinitializing random weights..")
    
    model.summary()

    train(model=model, args=args, data=None)

    '''
    x_train = np.ones([2, 255, 255, 3])
    y_train = np.zeros([2, 255, 255, 2, 6])
    x_test = np.ones([1, 255, 255, 3])
    y_test = np.zeros([1, 255, 255, 2, 6])
    data = (x_train, y_train), (x_test, y_test)
    
    sess = tf.Session()
    with sess.as_default():
        t1 = np.ones([56, 6], dtype=np.float32)
        t2 = np.ones([28, 6], dtype=np.float32)
        t2 = np.concatenate((np.zeros([28, 6], dtype=np.float32), t2))
        t1 = np.expand_dims(t1, 1)
        t2 = np.expand_dims(t2, 1)
        tx = np.concatenate((t1, t2), 1)
        print tx.shape
        t3 = tf.map_fn(lambda x: tf.cond(tf.constant(x[1][0] == 0, dtype=tf.bool), lambda: x[1], lambda:x[0]+x[1]), #generally bad practice to use a var outside of lambda scope here but its constant so wtv
                               tx,
                               dtype=tf.float32)
        print t3.eval()
    '''
