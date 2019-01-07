    # Testing visutils
    x, y = to_tensors("../../capsnet_data/data/raw", "red_drive_30", out_width=255, out_height=255)
    from visutils import draw_on
    import matplotlib as mpl
    mpl.use('TkAgg')
    import matplotlib.pyplot as plt
    x_exp = np.expand_dims(x, 0)
    y = np.expand_dims(y, 0)
    fig, ax = plt.subplots()
    draw_on(x_exp, y, ax)
    x = x.astype(np.uint8)
    plt.imshow(x)
    plt.show()
