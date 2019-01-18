import tensorflow as tf
import numpy as np

from capsnet_model.model import CapsNet
from capsnet_model.caps_layer import squash

from utils.to_tfrecord import parse_fn_caps_tfrecord

def weighted_bce(y_true, y_pred):
    weight = 33.971468469433276
    ones = tf.ones(tf.shape(y_pred))
    #all tensors in operation below have dims [None, h, w, n_classes]
    loss_sum = -(weight * tf.multiply(y_true, tf.log(y_pred))
                 + tf.multiply(ones - y_true, tf.log(ones - y_pred)))
    return tf.reduce_mean(loss_sum)

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

    checkpt = tf.keras.callbacks.ModelCheckpoint(args.working_dir + "/chkpts/chkpt-saved.h5", #after - {epoch:02d}
                                                 save_best_only=True, #True can only work if there is a validation/ test set
                                                 save_weights_only=True,
                                                 verbose=1)

    halfway_pt = args.lr_cycle / 2
    def cyclic_lr(epoch):
        modulo = epoch % args.lr_cycle
        if modulo < halfway_pt:
            multiple = args.lr_decay ** modulo
        else:
            multiple = args.lr_cycle - modulo
            multiple = args.lr_decay ** multiple
        return args.lr * multiple
    
    lr_decay = tf.keras.callbacks.LearningRateScheduler(schedule=cyclic_lr)

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=args.lr),
                  loss=weighted_bce, #tf.keras.losses.mean_squared_error,
                  metrics={'capsnet': 'accuracy'})
    
    model.summary()
    
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
        eval_set = tfrecord_generator(args.working_dir + "/" + args.eval_path, 2)
        #bec I know i have 20 images in my eval set rn. Really gotta fix these hardcoded sample size bs
        #but eval_set batch size must be equal to training set batch size bec capsule layer dimensions is hardcoded at init rn
        model.fit(training_set.make_one_shot_iterator(),
                  steps_per_epoch=max(1, int(40 / args.batch_size)), #lol shit how to determine dataset size from tfrecord file
                  callbacks=[tb, checkpt, lr_decay],
                  epochs=args.epochs,
                  validation_data=eval_set.make_one_shot_iterator(),
                  validation_steps=5,
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
    parser.add_argument("--lr_cycle", default=25, type=int)

    args = parser.parse_args()

    if args.from_chkpt and args.from_saved:
        raise Exception("Error: You can only choose either from_chkpt or from_saved")

    model = CapsNet()
    
    sess = tf.Session()
    with sess.as_default():
        sess.run(tf.global_variables_initializer())
        train(model=model, args=args, data=None)
