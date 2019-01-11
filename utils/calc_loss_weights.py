from to_tfrecord import to_tensors
import numpy as np
import glob
import os

CLASS_LABELS = ["red"]

def calculate_pixel_weights(directory,
                            img_format="jpg",
                            out_width=None,
                            out_height=None,
                            caps_dim=5,
                            class_labels=CLASS_LABELS):
    names = [os.path.basename(os.path.normpath(fname)).rsplit(".", 1)[0] for fname in glob.glob(directory+"/*.csv")]
    counts = {"background": 0}
    for name in class_labels:
        counts[name] = 0
    print counts
    for name in names:
        img, target = to_tensors(directory,
                                 name,
                                 img_format,
                                 out_width,
                                 out_height,
                                 caps_dim)
        
        probs = target[:,:,:,:1]
        non_zero_indices = np.array(np.nonzero(probs)).T
        non_zero_counts = non_zero_indices[:, -2]
        non_zero_counts_int = 0
        for i in range(len(class_labels)):
            count = np.array(np.where(non_zero_counts == i)).shape[-1]
            counts[class_labels[i]] += count
            non_zero_counts_int += count
        counts["background"] += (probs.size - non_zero_counts_int)
    print counts
    base = counts["background"]
    res = [1.0]
    for label in class_labels:
        res.append(base / float(counts[label]))
    return res

if __name__ == "__main__":
    print(calculate_pixel_weights("../../capsnet_data/data/raw"))
