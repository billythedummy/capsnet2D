import numpy as np

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
    return np.array([x, y, w, h, phi, theta])
    

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
    if ave_h[1] >= 0 and ave_v[0] <= 0:
        return deg
    else:
        return -deg

def angle_map(deg):
    return linear_map(deg, -90.0, 90.0)

def linear_map(val, val_min, val_max, conv_min=-1.0, conv_max=1.0):
    val_range = float(val_max - val_min)
    conv_range = float(conv_max - conv_min)
    return ((val - val_min) / val_range) * conv_range + conv_min
    

if __name__ == "__main__":
    import csv
    import matplotlib as mpl
    mpl.use('TkAgg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    direc = "../../capsnet_data/data/raw/"
    im_name = "test"
    im_file = direc + im_name + ".jpg"
    img = plt.imread(im_file)
    plt.imshow(img)
    csv_file = direc + im_name + '.csv'
    with open(csv_file, 'rb') as csv_f:
        reader = csv.reader(csv_f, delimiter=',')
        for row in reader:
            row = np.array(row, dtype=np.float32)
            #print to_capsule(row, img)
            x = row[0:7:2]
            x = np.append(x, row[0])
            y = row[1:8:2]
            y = np.append(y, row[1])
            ax.plot(x, y, linewidth=3, color="red")
        plt.show()

            
