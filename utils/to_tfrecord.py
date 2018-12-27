import numpy as np
import tensorflow as tf
import csv
from PIL import Image
from utils.to_capsule import to_capsule, linear_map
import glob
import os

def to_tensors(directory, name,
              img_format="jpg",
              out_width=None,
              out_height=None,
              caps_dim=6):
    img = Image.open(directory + "/" + name + "." + img_format)
    #rgba to rgb conversion
    #img.load()
    #background = Image.new("RGB", img.size, (255, 255, 255))
    #background.paste(img, mask=img.split()[3])
    #img = background
    img_np = np.array(img)
    if out_width is None:
        out_width = img_np.shape[1]
    if out_height is None:
        out_height = img_np.shape[0]        
    #print out_width, out_height
    csv_file = directory + "/" + name + ".csv"
    with open(csv_file, "rb") as csv_f:
        reader = csv.reader(csv_f, delimiter=",")
        capsules = []
        for row in reader:
            capsules.append(to_capsule(row, img_np))
    if out_width != img_np.shape[1] or out_height != img_np.shape[0]:
        img = img.resize((out_width, out_height))
        img_np = np.array(img)
    #print capsules
    caps_channel = create_capsule_channel(img_np, capsules, caps_dim)
    bg_channel = create_background_channel([caps_channel], caps_dim)
    caps_channel = np.expand_dims(caps_channel, -2)
    bg_channel = np.expand_dims(bg_channel, -2)
    target_tensor = np.concatenate((caps_channel, bg_channel), axis=-2)
    return img_np.astype(np.float32), target_tensor.astype(np.float32)
    
def create_capsule_channel(img_tensor, capsules, caps_dim):
    shape = np.array(img_tensor.shape[:2])
    shape = np.append(shape, caps_dim)
    res = np.zeros(shape)
    for capsule in capsules:
        x = int(np.round(linear_map(capsule[0], -1.0, 1.0, 0, 1.0) * shape[1]))
        y = int(np.round(linear_map(capsule[1], -1.0, 1.0, 0, 1.0) * shape[0]))
        res[y, x] = capsule
    return res

def create_background_channel(capsule_channels, caps_dim):
    shape = capsule_channels[0].shape
    res = np.ones(shape)
    res /= caps_dim
    for caps_channel in capsule_channels:
        res[np.nonzero(caps_channel)] = 0
    return res

def parse_fn_caps_tfrecord(example_proto):
    features = {'height': tf.FixedLenFeature((), tf.int64),
                'width': tf.FixedLenFeature((), tf.int64),
                'depth': tf.FixedLenFeature((), tf.int64),
                'caps_dim': tf.FixedLenFeature((), tf.int64),
                'image_raw': tf.FixedLenFeature((), tf.string),
                'output_raw': tf.FixedLenFeature((), tf.string)}
    parsed_features = tf.parse_single_example(example_proto, features)
    h = tf.cast(parsed_features['height'], tf.int32)
    w = tf.cast(parsed_features['width'], tf.int32)
    d = tf.cast(parsed_features['depth'], tf.int32)
    img = tf.decode_raw(parsed_features["image_raw"], tf.float32)
    img = tf.reshape(img, [h, w, d])
    caps_dim = tf.cast(parsed_features['caps_dim'], tf.int32)
    target_output = tf.decode_raw(parsed_features["output_raw"], tf.float32)
    target_output = tf.reshape(target_output, [h, w, -1, caps_dim])
    return img, target_output

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def write_dir(dirname, tf_record_name,
              caps_dim=6, n_classes=1, out_width=None, out_height=None):
    names = [os.path.basename(os.path.normpath(fname)).rsplit(".", 1)[0] for fname in glob.glob(dirname+"/*.csv")]
    options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
    writer = tf.python_io.TFRecordWriter(tf_record_name, options=options)
    for name in names:
        x, y = to_tensors(dirname,
                          name,
                          caps_dim=caps_dim,
                          out_width=out_width,
                          out_height=out_height)
        height = x.shape[0]
        width = x.shape[1]
        depth = x.shape[2]
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(height),
            'width': _int64_feature(width),
            'depth': _int64_feature(depth),
            'caps_dim': _int64_feature(caps_dim),
            'image_raw': _bytes_feature(x.tostring()),
            'output_raw': _bytes_feature(y.tostring())}))
        writer.write(example.SerializeToString())
    writer.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Data labelling for Capsule network")
    parser.add_argument("--data_dir", default="../../capsnet_data/data/raw", type=str)
    parser.add_argument("--image_format", default="jpg", type=str)
    parser.add_argument("--out_file", default="../../capsnet_data/data/tfrecord/train.tfrecords", type=str)
    parser.add_argument("--out_width", default=255, type=int)
    parser.add_argument("--out_height", default=255, type=int)
    args = parser.parse_args()
    
    write_dir(args.data_dir,
              args.out_file,
              out_width=args.out_width,
              out_height=args.out_height)
    
    print("Write " + args.data_dir + " to " + args.out_file + "complete!")
