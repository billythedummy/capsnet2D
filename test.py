import tensorflow as tf
import numpy as np
from utils.visutils import draw_on

from capsnet_model.model import CapsNet

import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from PIL import Image

import argparse
import glob

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test trained network")
    parser.add_argument("--image_dir", default="../capsnet_data/data/raw", type=str)
    parser.add_argument("--weights_path", default="../capsnet_data/trained_model.h5", type=str)
    args = parser.parse_args()
    
    shape = (255, 255)
    input_shape = (255, 255, 3)

    sess = tf.Session()
    with sess.as_default():
        model = CapsNet(input_shape=input_shape)
        model.load_weights(args.weights_path)
        print("Weights loaded from " + args.weights_path)
        #model.summary()

        img_files = glob.glob(args.image_dir+"/*.jpg")
        img_files.extend(glob.glob(args.image_dir+"/*.png"))
        
        for img_file in img_files:
            fig, ax = plt.subplots()
            ax.lines = []
            img = Image.open(img_file)
            img = img.resize(shape)
            img = np.array(img)
            img_exp = np.expand_dims(img, axis=0)
            img_in = tf.convert_to_tensor(img_exp, dtype=tf.float32)
            capsules = model(img_in)
            draw_on(img_exp, capsules.eval(), ax, limit=5)
            plt.imshow(img)
            plt.show()

    

    

