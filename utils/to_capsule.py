import numpy as np
from skimage.draw import line
from scipy import ndimage as ndi

SCALE_MIN = 0.0
SCALE_MAX = 1.0

def to_capsule(csv_row, img):
    csv_row = np.array(csv_row, dtype=np.float32)
    x1, y1, x2, y2, x3, y3, x4, y4, theta = csv_row
    xi, yi, xiplus1, yiplus1 = get_vertex_arrays([(x1, y1), (x2, y2), (x3, y3), (x4, y4)])
    sa = signed_area(xi, yi, xiplus1, yiplus1)
    c = centroid(xi, yi, xiplus1, yiplus1, sa)
    e_vs = get_edge_vectors(xi, yi, xiplus1, yiplus1)
    a_vs = get_ave_edge_vectors(e_vs)
    ave_h = a_vs[0]
    ave_v = a_vs[1]
    w = np.linalg.norm(ave_h)
    h = np.linalg.norm(ave_v)
    phi = get_phi(ave_h, ave_v, w, h)
    w = linear_map(w, 0, img.shape[1])
    h = linear_map(h, 0, img.shape[0])
    phi = angle_map(phi)
    theta = angle_map(theta)
    x = linear_map(c[0], 0, img.shape[1])
    y = linear_map(c[1], 0, img.shape[0])
    return np.array([1.0, w, h, phi, theta]), x, y #x y encoded by linear map

def add_mask(csv_row, channel_zeros):
    # Adds the pixel mask to an img of zeros (binary encoding)
    # Modifies channel_zeros, a np array (height, width) of zeros
    # Returns np array (rows, cols)
    csv_row = np.array(csv_row, dtype=np.float32)
    x1, y1, x2, y2, x3, y3, x4, y4, theta = csv_row
    channel_zeros[line(y1.astype(int), x1.astype(int), y2.astype(int), x2.astype(int))] = 1
    channel_zeros[line(y2.astype(int), x2.astype(int), y3.astype(int), x3.astype(int))] = 1
    channel_zeros[line(y3.astype(int), x3.astype(int), y4.astype(int), x4.astype(int))] = 1
    channel_zeros[line(y4.astype(int), x4.astype(int), y1.astype(int), x1.astype(int))] = 1
    return ndi.binary_fill_holes(channel_zeros)
 
def to_drawable(capsule, img, x, y):
    w = capsule_unmap(capsule[1], 0, img.shape[1])
    h = capsule_unmap(capsule[2], 0, img.shape[0])
    phi = angle_unmap(capsule[3])
    theta = angle_unmap(capsule[4])
    print(capsule[0], x, y, w, h, phi, theta)
    d1 = np.array([np.cos(np.radians(phi)), np.sin(np.radians(phi))])
    scale = np.linalg.norm(d1)
    d1 = d1 / scale
    #i fukt up, phi isnt enough to capture the pose
    d2 = np.array([-d1[1], d1[0]])
    d1 = w * d1 * 0.5
    d2 = h * d2 * 0.5
    center = np.array([x, y])
    v1 = np.round(center - d1 + d2).astype(int)
    v2 = np.round(center + d1 + d2).astype(int)
    v3 = np.round(center + d1 - d2).astype(int)
    v4 = np.round(center - d1 - d2).astype(int)
    vertices = np.vstack((v1, v2, v3, v4))
    mins = np.zeros(vertices.shape, dtype=int)
    maxs = np.array([img.shape[1], img.shape[0]], dtype=int)
    maxs = np.tile(maxs, (4, 1))
    vertices = np.minimum(maxs, np.maximum(mins, vertices))
    return vertices, theta

def signed_area(xi, yi, xiplus1, yiplus1):
    elems = xi * yiplus1 - xiplus1 * yi
    return 0.5 * np.sum(elems)

def centroid(xi, yi, xiplus1, yiplus1, signed_area):
    factor = 1 / (6 * signed_area)
    elems_x = (xi + xiplus1) * (xi * yiplus1 - xiplus1 * yi)
    elems_y = (yi + yiplus1) * (xi * yiplus1 - xiplus1 * yi)
    return factor * np.sum(elems_x), factor * np.sum(elems_y)

def get_vertex_arrays(vertices):
    np_vertices = np.array(vertices)
    xi = np_vertices[..., 0]
    yi = np_vertices[..., 1]
    xiplus1 = np.roll(xi, 1)
    yiplus1 = np.roll(yi, 1)
    return xi, yi, xiplus1, yiplus1

def get_edge_vectors(xi, yi, xiplus1, yiplus1):
    # returns [[e1_x, e1_y], [e2_x, e2_y], [e3_x, e3_y], [e4_x, e4_y]]
    # Where ei is the edge vector and the sequence is in clockwise
    # or counter clockwise directions around the polygon
    x_diff = xiplus1 - xi
    y_diff = yiplus1 - yi
    x_diff = np.expand_dims(x_diff, 0)
    y_diff = np.expand_dims(y_diff, 0)
    unsigned = np.squeeze(np.dstack((x_diff, y_diff)))
    return unsigned

def get_ave_edge_vectors(unsigned_edge_vectors):
    # returns [[h_x, h_y] [v_x v_y]] where h_x and v_y is positive
    # so 2 vectors pointing downwards and rightwards from origin
    e1 = unsigned_edge_vectors[0]
    e2 = unsigned_edge_vectors[1]
    e3 = unsigned_edge_vectors[2]
    e4 = unsigned_edge_vectors[3]
    #print unsigned_edge_vectors
    ave_1 = (e1 - e3) / 2.0
    ave_2 = (e2 - e4) / 2.0
    if np.abs(ave_1[0]) > np.abs(ave_1[1]):
        ave_h = ave_1
        ave_v = ave_2
    else:
        ave_h = ave_2
        ave_v = ave_1
    if ave_h[0] < 0:
        ave_h = -ave_h
    if ave_v[1] < 0:
        ave_v = -ave_v
    return np.array([ave_h, ave_v])

def get_phi(ave_h, ave_v, ave_h_norm, ave_v_norm):
    dot_h = np.dot(ave_h, np.array([1, 0]))
    dot_v = np.dot(ave_v, np.array([0, 1]))
    phi_1 = np.arccos(dot_h / ave_h_norm)
    phi_2 = np.arccos(dot_v / ave_v_norm)
    deg = np.degrees((phi_1 + phi_2) / 2)
    #if ave_h[1] >= 0 and ave_v[0] <= 0:
        #return deg
    #el
    if ave_h[1] <= 0 and ave_v[0] >= 0:
        return -deg
    else:
        return deg

def angle_map(deg):
    return linear_map(deg, -90.0, 90.0)

def capsule_unmap(prop, conv_min, conv_max):
    return linear_map(prop, SCALE_MIN, SCALE_MAX, conv_min, conv_max)

def angle_unmap(prop):
    return capsule_unmap(prop, -90.0, 90.0)

def linear_map(val, val_min, val_max, conv_min=SCALE_MIN, conv_max=SCALE_MAX):
    val_range = np.float32(val_max - val_min)
    conv_range = np.float32(conv_max - conv_min)
    return ((val - val_min) / val_range) * conv_range + conv_min
    
if __name__ == "__main__":
    import csv
    import matplotlib as mpl
    mpl.use('TkAgg')
    import matplotlib.pyplot as plt
    from PIL import Image
    import glob
    import os
    
    direc = "../../capsnet_data/data/raw/"
    names = [os.path.basename(os.path.normpath(fname)).rsplit(".", 1)[0] for fname in glob.glob(direc+"/*.csv")]
    for im_name in names:
        im_file = direc + im_name + ".jpg"
        img = plt.imread(im_file)
        plt.imshow(img)
        plt.show()
        csv_file = direc + im_name + '.csv'
        with open(csv_file, 'rb') as csv_f:
            reader = csv.reader(csv_f, delimiter=',')
            img_zeros = np.zeros(img.shape[0:2])
            for row in reader:
                row = np.array(row, dtype=np.float32)
                img_zeros = add_mask(row, img_zeros)
                #print to_capsule(row, img)
                #x = row[0:7:2]
                #x = np.append(x, row[0])
                #y = row[1:8:2]
                #y = np.append(y, row[1])
                #ax.plot(x, y, linewidth=3, color="red")
            img_zeros = Image.fromarray(np.uint8(img_zeros))
            img_zeros = img_zeros.resize((255, 255))
            img_zeros = np.array(img_zeros)
            caps_dim = 6
            target_tensor = np.expand_dims(img_zeros, -1) #just 1 class for now
            target_tensor = np.expand_dims(target_tensor, -1)
            target_tensor_shape = target_tensor.shape
            for i in range(caps_dim - 1):
                target_tensor = np.concatenate((target_tensor, np.zeros(target_tensor_shape)), axis=-1)

            plt.imshow(img_zeros)
            plt.show()
            
