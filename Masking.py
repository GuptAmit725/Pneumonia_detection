def apply_mask(img_x, name=None):
    """
    This function applies mask to a single image on 6 different parts of the image.
    The image size is 128*128 and mask size is 32*32 with stride = 32.
    For every image we will have 6 variants of images with different masks.
    """

    import numpy as np
    masked = []

    for k in range(4):
        if k >= 1:
            for l in range(4):
                if l > 0 and l < 3:
                    f = np.array(img_x).copy()
                    f = f.reshape((1, 128, 128, 3))
                    f[0, 0 + k * 32:32 + k * 32, 0 + l * 32:32 + l * 32, :] = np.zeros((32, 32, 3))
                    masked.append(f)

    #pickle.dump(masked, open('/content/sample_data/img_files/img_' + name + '.pkl', 'wb'))
    # This dumping for every image has to be done to avoid notebook from crashing.
    # After every iterration I am saving the 6 images of a single image in list.

    return masked