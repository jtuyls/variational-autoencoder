import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def visualize_images(inputs):
    for i in range(0, len(inputs)):
        if inputs.shape[1] == 1:
            image = inputs[i].reshape(inputs.shape[2], inputs.shape[3])
        else:
            image = deprocess_image(inputs[i])
        plt.imshow(image, cmap='gray')
        plt.savefig('fig_' + str(i) + '.png')
        plt.show()

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
