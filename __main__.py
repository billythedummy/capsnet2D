import tensorflow as tf
import numpy as np

from capsnet_model.model import CapsNet
from capsnet_model.caps_layer import CapsLayer2D

from utils.to_tfrecord import parse_fn_caps_tfrecord

def load_data(tfrecordpath):
    x_train = np.ones([1, 255, 255, 3])
    y_train = np.zeros([1, 255, 255, 2, 6])
    x_test = np.ones([1, 255, 255, 3])
    y_test = np.zeros([1, 255, 255, 2, 6])
    return (x_train, y_train), (x_test, y_test)

def weighted_vec_loss(y_true, y_pred):
    n_classes = tf.keras.backend.int_shape(y_pred)[-2] - 1
    y_classes, y_bg = tf.split(y_true, [n_classes, 1], -2)
    y_pred_classes, y_pred_bg = tf.split(y_pred, [n_classes, 1], -2)

    #background only uses magnitude of vectors for comparison
    y_bg_mag = tf.sqrt(tf.reduce_sum(tf.square(y_bg), -1)) 
    y_pred_bg_mag = tf.sqrt(tf.reduce_sum(tf.square(y_pred_bg), -1))
    bg_loss = tf.abs(y_bg_mag - y_pred_bg_mag)
    bg_loss = tf.reduce_sum(bg_loss)

    #classes uses both magnitude and direction so magnitude of vector diff
    elems = (tf.reshape(y_classes, [-1]), tf.reshape(y_pred_classes, [-1]))
    length = tf.keras.backend.int_shape(elems[1])[0]
    class_loss = tf.map_fn(lambda x: tf.sqrt(tf.square(x[0] - x[1])) if x[0] == 0
                           else length * tf.sqrt(tf.square(x[0] - x[1])), #generally bad practice to use a var outside of lambda scope here but its constant so wtv
                           elems,
                           dtype=tf.float32)
    
    class_loss = tf.reduce_sum(class_loss)
    loss = 0.6 * class_loss + 0.4 * bg_loss #weigh background less
    #print(loss)
    return loss

def tfrecord_generator(filenames, batch_size):
    dataset = tf.data.TFRecordDataset(filenames, compression_type='GZIP')
    dataset = dataset.map(parse_fn_caps_tfrecord, num_parallel_calls=8)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
    return dataset

def train_generator(x, y, batch_size):
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
    generator = train_datagen.flow(x, y, batch_size=batch_size)
    while 1:
        x_batch, y_batch = generator.next()
        #print("Yielding new batch...")
        yield (x_batch, y_batch)

def train(model, data, args):
    #(x_train, y_train), (x_test, y_test) = data

    tb = tf.keras.callbacks.TensorBoard(log_dir=args.working_dir + "/tb-logs", batch_size=args.batch_size) 

    if args.on_colab:
        from utils.colab_utils import GDriveCheckpointer
        def compare(best, new):
            return best.losses['val_acc'] < new.losses['val_acc']

        def path(new):
            if new.losses['val_acc'] > 0.8:
                return 'chkpt.h5'
            
        checkpt = GDriveCheckpointer(compare, path)

    else:
        checkpt = tf.keras.callbacks.ModelCheckpoint(args.working_dir + "/chkpts/chkpt-{epoch:02d}.h5",
                                                     save_best_only=False, #True can only work with validation loss
                                                     save_weights_only=True,
                                                     verbose=1)
    lr_decay = tf.keras.callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (args.lr_decay ** epoch))

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=args.lr),
                  loss=weighted_vec_loss,
                  metrics={'capsnet': 'accuracy'})


    #print("Beginning to train...")
    
    #for TFRecord
    training_set = tfrecord_generator(args.working_dir + "/" + args.train_path, args.batch_size)
    model.fit(training_set.make_one_shot_iterator(),
              steps_per_epoch=max(1, int(20 / args.batch_size)), #lol shit how to determine dataset size from tfrecord file
              callbacks=[tb, checkpt, lr_decay],
              epochs=args.epochs,
              verbose=1)
    
    #for fit_generator
    #model.fit_generator(generator=train_generator(x_train, y_train, args.batch_size),
                        #steps_per_epoch=max(1, int(y_train.shape[0] / args.batch_size)),
                        #epochs=args.epochs,
                        #validation_data=[x_test, y_test],
                        #callbacks=[tb,checkpt,lr_decay],
                        #verbose=1)

    model.save_weights("../capsnet_data/trained_model.h5")
    print("Trained model saved to same directory!")

    return model
                  
if __name__ == "__main__":
    import argparse
    import glob
    parser = argparse.ArgumentParser(description="2D Capsule Network. Run from inside the main repo.")
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--batch_size", default=20, type=int)
    parser.add_argument("--lr", default=0.001, type=float, help="Initial learning rate")
    parser.add_argument("--lr_decay", default=0.95, type=float, help="Exponent base for decay. Between 0-1")
    parser.add_argument("--routings", default=3, type=int, help="Number of routings for capsules")
    parser.add_argument("--from_chkpt", default=0, type=int)
    parser.add_argument("--from_saved", default=0, type=int)
    parser.add_argument("--working_dir", default="../capsnet_data", type=str)
    parser.add_argument("--train_path", default="data/tfrecord/train.tfrecords", type=str)
    parser.add_argument("--on_colab", default=0, type=int)
    
    args = parser.parse_args()

    if args.from_chkpt and args.from_saved:
        raise Exception("Error: You can only choose either from_chkpt or from_saved")

    model = CapsNet(batch_size=args.batch_size)

    if args.from_chkpt or args.from_saved:
        if args.from_chkpt:
            chkpts = glob.glob(args.working_dir + "/chkpts/chkpt-*.h5")
            #for colab
            #chkpts = glob.glob("./chkpt-*.h5")
        else:
            chkpts = glob.glob(args.working_dir + "/trained_model.h5")
        if len(chkpts) > 0:
            path = sorted(chkpts)[-1]
            model.load_weights(path)
            print("Weights loaded from " + path)
        else:
            print("No existing saved model/ checkpoint found, reinitializing random weights..")
    
    model.summary()

    train(model=model, data=load_data("lol"), args=args)
