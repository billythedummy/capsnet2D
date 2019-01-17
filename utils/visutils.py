import numpy as np
import tensorflow as tf 
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from to_capsule import to_drawable

colors = ["red", "blue"]
color_dict = {"red": (255, 0, 0), "blue": (0, 0, 255)}

def draw_on(imgs, capsules, ax, limit=5): #both numpy arrays
    for i in range(3): #batch, height, width
        assert imgs.shape[i] == capsules.shape[i], "Shapes [" + str(imgs.shape[i]) +"], [" + str(capsules.shape[i]) + "] do not match"
    caps_prob = np.copy(capsules[:,:,:,:,0]) #first dim is probability
    
    #Loop here finds top `limit` number of probabilities' indices
    indices = []
    while len(indices) < limit:
        confident = np.argmax(caps_prob)
        confident = np.unravel_index(confident, caps_prob.shape)
        indices.append(confident)
        caps_prob[confident] = 0
    
    # dont need to change -1 index for now bec theres only 1 class at index 0
    x = np.empty(shape=(0, 5))
    y = np.empty(shape=(0, 5))
    classes = np.array([], dtype=int)
    thetas = np.array([])
    for index in indices:
        caps = capsules[tuple(index)]
        #print(caps)
        img = imgs[index[0]]
        vertices, theta = to_drawable(caps, img, index[2], index[1])
        this_x = vertices[:,0]
        this_x = np.append(this_x, vertices[0, 0]) #add x of first vertex as last so full rect gets drawn
        this_x = np.reshape(this_x, (1, 5))
        this_y = vertices[:,1]
        this_y = np.append(this_y, vertices[0, 1]) #add y of first vertex as last so full rect gets drawn
        this_y = np.reshape(this_y, (1, 5))
        x = np.vstack((x, this_x))
        y = np.vstack((y, this_y))
        thetas = np.append(thetas, theta)
        classes = np.append(classes, index[-1])

    for j in range(x.shape[0]):
        x_list = x[j]
        y_list = y[j]
        this_class = classes[j]
        #print(x_list)
        #print(y_list)
        ax.plot(x_list, y_list, linewidth=3, color=colors[this_class % len(colors)])

def draw_seg(imgs, capsules):
    #Returns the RGB (NOT BGR) mask of the images in the batch imgs.
    #imgs is input to network, capsules is output
    for i in range(3): #batch, height, width
        assert imgs.shape[i] == capsules.shape[i], "Shapes [" + str(imgs.shape[i]) +"], [" + str(capsules.shape[i]) + "] do not match"
    print(capsules)
    #[batch, height, width, n_classes]
    rgb_mask = np.zeros(imgs.shape)
    #print bgr_mask.shape
    #[batch, height, width, channels]
    cutoff = 0.6491
    for i in range(len(colors) - 1): #-1 for now bec blue hasnt been implemented
        class_channel = capsules[:,:,:,i]
        this_class = colors[i]
        r = np.array(np.where(class_channel > cutoff,
                              color_dict[this_class][0], 0))
        r = np.expand_dims(r, -1)
        g = np.array(np.where(class_channel > cutoff,
                              color_dict[this_class][1], 0))
        g = np.expand_dims(g, -1)
        b = np.array(np.where(class_channel > cutoff,
                              color_dict[this_class][2], 0))
        b = np.expand_dims(b, -1)
        this_mask = np.concatenate((r, g, b), axis=-1)
        rgb_mask += this_mask
    return rgb_mask

if __name__ == "__main__":
    img_path = "../../capsnet_data/data/raw/red_top_3_194.jpg"
    caps = np.concatenate((np.ones(shape=[1, 1080, 1920, 1, 5], dtype=np.float32),
                           np.zeros(shape=[1, 1080, 1920, 1, 5], dtype=np.float32)), axis=-2)
    img = plt.imread(img_path)
    img = np.expand_dims(img, 0)
    draw_on(img, caps, None, limit=10)
