import tensorflow as tf
import numpy as np
from utils.visutils import draw_on, draw_seg

from capsnet_model.model import CapsNet

import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from PIL import Image

import argparse
import glob
import time

THRESH = 0.832

def test_img_dir(img_dir, sess):
    img_files = glob.glob(img_dir+"/*.jpg")
    img_files.extend(glob.glob(img_dir+"/*.png"))
    
    for img_file in img_files:
        fig, ax = plt.subplots()
        ax.lines = []
        img = Image.open(img_file)
        img = img.resize(shape)
        img = np.array(img)
        img_exp = np.expand_dims(img, axis=0)

        input_tensor = sess.graph.get_tensor_by_name('capsnet_input:0')
        output_tensor = sess.graph.get_tensor_by_name('final_norm/capsnet_output:0')
        start = time.time()
        capsules = sess.run(output_tensor, feed_dict={input_tensor: img_exp})
        
        print(capsules)
        #print(capsules.shape)
        print("Took: " + str(time.time() - start) + "s")

        #print capsules
        mask = draw_seg(img_exp, capsules, THRESH)
        img_and_mask = img.astype(int) + np.squeeze(mask, 0).astype(int)
        img_and_mask = np.clip(img_and_mask, 0, 255).astype(np.uint8)
        plt.imshow(img_and_mask)
        plt.show()

def run_live(sess):
    import cv2
    cam_index = 0
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print("Camera not opened, retrying")
        cap.open(cam_index)
    ret = True
    while ret:
        ret, img = cap.read()
        img = cv2.resize(img, (255, 255))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_exp = np.expand_dims(img, axis=0)

        input_tensor = sess.graph.get_tensor_by_name('capsnet_input:0')
        output_tensor = sess.graph.get_tensor_by_name('final_norm/capsnet_output:0')
        capsules = sess.run(output_tensor, feed_dict={input_tensor: img_exp})

        bgr_mask = draw_seg(img_exp, capsules, THRESH)
        img_and_mask = img_exp.astype(int) + bgr_mask.astype(int)
        img_and_mask = np.clip(img_and_mask, 0, 255).astype(np.uint8)
        img_and_mask = np.squeeze(img_and_mask, 0)
        img_and_mask = cv2.cvtColor(img_and_mask, cv2.COLOR_RGB2BGR)
        cv2.imshow("Live Feed", img_and_mask)

        if cv2.waitKey(1) == 27:
            cv2.destroyAllWindows()
            cap.release()
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test trained network")
    parser.add_argument("--img_dir", default="../capsnet_data/data/raw", type=str)
    parser.add_argument("--weights_path", default="../capsnet_data/trained_model.h5", type=str)
    parser.add_argument("--live", default=0, type=int)
    args = parser.parse_args()
    
    shape = (255, 255)
    input_shape = (255, 255, 3)

    sess = tf.Session()
    with sess.as_default():
        model = CapsNet(input_shape=input_shape)
        model.load_weights(args.weights_path)
        #sess.run(tf.initializers.global_variables())
        print("Weights loaded from " + args.weights_path)
        model.summary()
	#print [n.name for n in tf.get_default_graph().as_graph_def().node]
        if args.live:
            run_live(sess)
        else:
            test_img_dir(args.img_dir, sess)

    

    

