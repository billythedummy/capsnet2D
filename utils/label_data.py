# CSV Schema:
# File name same as image
# 1 Row per plate
# 9 columns: x1 y1 x2 y2 x3 y3 x4 y4 theta

import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import glob
import csv
import argparse

this_plates = [] #[[(x1, y1), (x2, y2)...], ...]
this_thetas = []
this_fname = None

colors = ["red", "green", "blue"]

def get_theta_input(plate_index):
    prompt = "Enter theta for the " + colors[plate_index % len(colors)] + " plate"
    return float(input(prompt))

def on_click(event):
    #right click to delete
    if event.button == 3:
        if len(this_plates) > 0:
            if len(this_plates[-1]) == 0:
                del this_plates[-1]
            if len(this_plates) == len(this_thetas):
                del this_thetas[-1]
            if len(this_plates) > 0:
                del this_plates[-1][-1]
    else:
        if len(this_plates) == 0 or len(this_plates[-1]) == 4:
            this_plates.append([])
        if event.xdata is not None and event.ydata is not None:
            this_plates[-1].append((event.xdata, event.ydata))
    #print("this_plates=" + str(this_plates) + ", this_thetas=" + str(this_thetas))
    render_lines()
    plt.draw()

def render_lines():
    global ax
    ax.lines = []
    x = []
    y = []
    for i in range(len(this_plates)):
        box = this_plates[i]
        for pt in box:
            x.append(pt[0])
            y.append(pt[1])
        if len(box) == 4:
            first_pt = box[0]
            x.append(first_pt[0])
            y.append(first_pt[1])
        ax.plot(x, y, linewidth=3, color=colors[i%len(colors)])
        x = []
        y = []
        
def reset_state_vars(fname):
    global this_fname, this_plates, this_thetas, fig, ax
    fig, ax = plt.subplots()
    this_plates = []
    this_thetas = []
    this_fname = fname
    ax.lines = []
    print("Reset: this_plates=" + str(this_plates) + ", this_thetas=" + str(this_thetas) + ", this_fname=" + this_fname)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data labelling for Capsule network")
    parser.add_argument("--image_dir", default="../../capsnet_data/data/raw", type=str)
    parser.add_argument("--image_format", default="png", type=str)
    args = parser.parse_args()
    
    img_fnames = glob.glob(args.image_dir + "/*." + args.image_format)
    
    def key_handler(event):
        #print(event.key)
        global this_fname
        if event.key == 'enter':
            if len(this_plates) > 0 and len(this_plates[-1]) != 4:
                print("Incomplete last plate!")
                return
            while len(this_thetas) < len(this_plates):
                this_thetas.append(get_theta_input(len(this_thetas)))
            fname_no_suf = this_fname.rsplit(".", 1)[0]
            with open(fname_no_suf + ".csv", mode="w") as csv_file:
                writer = csv.writer(csv_file, delimiter=",")
                for i in range(len(this_plates)):
                    row = []
                    #sorted_coords = sorted(this_plates[i], key=lambda pt: (pt[0], pt[1]))
                    for coord in this_plates[i]:
                        row.append(coord[0])
                        row.append(coord[1])
                    row.append(this_thetas[i])
                    writer.writerow(row)
            print("Data saved for " + this_fname + " : this_plates=" + str(this_plates) + ", this_thetas="+str(this_thetas))
            return
        if event.key == ' ' and len(this_plates) > 0 and len(this_plates[-1]) == 4:
            this_thetas.append(get_theta_input(len(this_plates) - 1))
            return
        
    for fname in img_fnames:
        reset_state_vars(fname)
        img = plt.imread(fname)
        plt.imshow(img)
        fig.canvas.mpl_connect('key_press_event', key_handler)
        fig.canvas.mpl_connect('button_press_event', on_click)
        plt.show()
