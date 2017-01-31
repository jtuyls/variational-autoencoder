
import os

import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def compare_images(input_images, reconstructed_images, stamp):
    print("Compare images in visualization")
    if not len(input_images) == len(reconstructed_images):
        raise ValueError("Inputs should have the same size")
    path = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(path + "/figures"):
        os.makedirs(path + "/figures")

    plt.figure(figsize=(6, 30))
    for i in range(0, len(input_images)):
        if input_images.shape[1] == 1:
            size = input_images.shape[2]
            input_image = input_images[i].reshape(size, size)
            reconstructed_image = reconstructed_images[i].reshape(size, size)
        else:
            input_image = deprocess_image(input_images[i])
            reconstructed_image = deprocess_image(reconstructed_images[i])
        plt.subplot(len(input_images), 2, 2 * i + 1)
        plt.imshow(input_image, cmap='gray')
        plt.title("Input image")
        plt.subplot(len(input_images), 2, 2 * i + 2)
        plt.imshow(reconstructed_image, cmap='gray')
        plt.title("Reconstructed image")
        # plt.show()
    plt.tight_layout()
    plt.savefig('figures/fig_comparison_' + stamp + '.png')
    plt.show()
    plt.close()
    #plt.savefig('figures/fig_' + str(i) + '.png')

def visualize_image_canvas(inputs, stamp):
    path = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(path + "/figures"):
        os.makedirs(path + "/figures")

    shape_x = inputs.shape[2]
    shape_c = inputs.shape[1]

    # inputs should be a squared number
    nx = int(np.sqrt(inputs.shape[0]))
    inputs = inputs.reshape((nx, nx, shape_c, shape_x, shape_x))

    if shape_c == 1:
        canvas = np.empty((shape_x * nx, shape_x * nx))
    else:
        canvas = np.empty((shape_x * nx, shape_x * nx, shape_c))

    for i in range(0, nx):
        for j in range(0, nx):
            if shape_c == 1:
                image = inputs[i][j].reshape(shape_x, shape_x)
                canvas[i*shape_x:(i+1)*shape_x, j*shape_x:(j+1)*shape_x] = image
            else:
                image = deprocess_image(inputs[i][j])
                canvas[i * shape_x:(i + 1) * shape_x, j * shape_x:(j + 1) * shape_x] = image
    plt.figure(figsize=(8, 10))
    plt.imshow(canvas, origin="upper", cmap="gray")
    plt.tight_layout()
    plt.savefig('figures/fig_canvas_' + stamp + '.png')
    plt.show()
    plt.close()




def visualize_images(inputs, stamp):
    print("visualize image")
    path = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(path + "/figures"):
        os.makedirs(path + "/figures")
    for i in range(0, len(inputs)):
        if inputs.shape[1] == 1:
            image = inputs[i].reshape(inputs.shape[2], inputs.shape[3])
        else:
            image = deprocess_image(inputs[i])
        plt.imshow(image, cmap='gray')
        plt.savefig('figures/fig_' + stamp + "_" + str(i) + '.png')
        plt.show()
        plt.close()


# Code of this method is fully copied from
    #   https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x
