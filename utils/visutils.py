import numpy as np
import tensorflow as tf #for .eval() call
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from to_capsule import to_drawable

colors = ["red", "blue"]

def draw_on(imgs, capsules, ax, limit=None): #both numpy arrays
    for i in range(3): #batch, height, width
        assert imgs.shape[i] == capsules.shape[i], "Shapes [" + str(imgs.shape[i]) +"], [" + str(capsules.shape[i]) + "] do not match"
    #print capsules
    capsules = capsules[:,:,:,:-1,:] #last channel is background channel
    bg = capsules[:,:,:,-1:,:]
    caps_dim = capsules.shape[-1]
    max_squared_norm = np.float32(caps_dim)
    bg_squared_norm = np.sum(np.square(bg), axis=-1)
    #capsules_squared_norm = np.sum(np.square(capsules), axis=-1)
    #confident = np.where((capsules_squared_norm / max_squared_norm) > 0.00575, #0.5^2
                         #1, 0)
    confident = np.where(bg_squared_norm < 0.00000005, 1, 0)
    indices = np.nonzero(confident)
    indices = np.array(indices).T
    print indices.shape[0]
    # dont need to change -1 index for now bec theres only 1 class at index 0
    x = np.empty(shape=(0, 5))
    y = np.empty(shape=(0, 5))
    classes = np.array([], dtype=int)
    thetas = np.array([])
    i = 0
    for index in indices:
        if limit is None or i < limit:
            caps = capsules[tuple(index)]
            img = imgs[index[0]]
            vertices, theta = to_drawable(caps, img)
            this_x = vertices[:,0]
            this_x = np.append(this_x, vertices[0, 0])
            this_x = np.reshape(this_x, (1, 5))
            this_y = vertices[:,1]
            this_y = np.append(this_y, vertices[0, 1])
            this_y = np.reshape(this_y, (1, 5))
            x = np.vstack((x, this_x))
            y = np.vstack((y, this_y))
            thetas = np.append(thetas, theta)
            classes = np.append(classes, index[-1])
        else:
            break
        i += 1
    for j in range(x.shape[0]):
        x_list = x[j]
        y_list = y[j]
        this_class = classes[j]
        #print x_list
        #print y_list
        ax.plot(x_list, y_list, linewidth=3, color=colors[this_class % len(colors)])
    
if __name__ == "__main__":
    img_path = "../../capsnet_data/data/raw/red_top_3_194.jpg"
    caps = np.concatenate((np.ones(shape=[1, 1080, 1920, 1, 6], dtype=np.float32),
                           np.zeros(shape=[1, 1080, 1920, 1, 6], dtype=np.float32)), axis=-2)
    img = plt.imread(img_path)
    img = np.expand_dims(img, 0)
    draw_on(img, caps, None, limit=10)