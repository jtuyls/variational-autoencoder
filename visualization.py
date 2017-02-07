
import os

import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

class Visualization(object):

    def __init__(self, output_dir= None):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.output_dir = output_dir

    def compare_images(self, input_images, reconstructed_images, stamp):
        # this method is inspired by https://jmetzen.github.io/2015-11-27/vae.html
        print("Compare images in visualization")
        if not len(input_images) == len(reconstructed_images):
            raise ValueError("Inputs should have the same size")

        plt.figure(figsize=(35, 20))
        for i in range(0, len(input_images)):
            if input_images.shape[1] == 1:
                size = input_images.shape[2]
                input_image = input_images[i].reshape(size, size)
                reconstructed_image = reconstructed_images[i].reshape(size, size)
            else:
                input_image = self.deprocess_image(input_images[i])
                reconstructed_image = self.deprocess_image(reconstructed_images[i])
            plt.subplot(len(input_images), 10, 1 + i)
            plt.imshow(input_image, cmap='gray')
            plt.title("Input image")
            plt.subplot(len(input_images), 10, 11 + i)
            plt.imshow(reconstructed_image, cmap='gray')
            plt.title("Reconstructed image")
            # plt.show()
        plt.tight_layout()
        plt.savefig(self.output_dir + '/fig_comparison_' + stamp + '.png')
        #plt.show()
        plt.close()
        #plt.savefig('figures/fig_' + str(i) + '.png')

    def visualize_image_canvas(self, inputs, stamp, nx=None):
        # this method is inspired by https://jmetzen.github.io/2015-11-27/vae.html

        shape_x = inputs.shape[2]
        shape_c = inputs.shape[1]

        if nx == None:
            # inputs should be a squared number
            nx = ny = int(np.sqrt(inputs.shape[0]))
        else:
            ny = inputs.shape[0] / nx

        inputs = inputs.reshape((ny, nx, shape_c, shape_x, shape_x))

        if shape_c == 1:
            canvas = np.empty((shape_x * ny, shape_x * nx))
        else:
            canvas = np.empty((shape_x * ny, shape_x * nx, shape_c))

        for i in range(0, ny):
            for j in range(0, nx):
                if shape_c == 1:
                    image = inputs[i][j].reshape(shape_x, shape_x)
                    canvas[i*shape_x:(i+1)*shape_x, j*shape_x:(j+1)*shape_x] = image
                else:
                    image = self.deprocess_image(inputs[i][j]) / 255.0
                    canvas[i * shape_x:(i + 1) * shape_x, j * shape_x:(j + 1) * shape_x] = image
        plt.figure(figsize=(8, 10))
        plt.imshow(canvas, origin="upper", cmap='gray')
        plt.tight_layout()
        plt.savefig(self.output_dir + '/fig_canvas_' + stamp + '.png')
        #plt.show()
        plt.close()

    def visualize_latent_space(self, output_function, shape, stamp):
        # Passing the output function from the variational autoencoder here is not very nice, but yeah... it works
        nx = ny = 20
        x_values = np.linspace(-3, 3, nx)
        y_values = np.linspace(-3, 3, ny)

        shape_x = shape[2]
        shape_c = shape[1]

        if shape_c == 1:
            canvas = np.empty((shape_x * ny, shape_x * nx))
        else:
            canvas = np.empty((shape_x * ny, shape_x * nx, shape_c))

        for i, yi in enumerate(x_values):
            for j, xi in enumerate(y_values):
                z_mu = np.array([[xi, yi]])
                # Get output for z
                constructed_image = output_function(z_mu)
                if shape_c == 1:
                    constructed_image = constructed_image[0].reshape(shape_x, shape_x)
                else:
                    constructed_image = self.deprocess_image(constructed_image[0]) / 255.0
                canvas[(nx - i - 1) * shape_x:(nx - i) * shape_x, j * shape_x:(j + 1) * shape_x] = constructed_image

        self.visualize_canvas(canvas=canvas, stamp=stamp)


    def visualize_images(self, inputs, stamp):
        print("visualize image")
        path = os.path.dirname(os.path.abspath(__file__))
        if not os.path.exists(path + "/figures"):
            os.makedirs(path + "/figures")
        for i in range(0, len(inputs)):
            if inputs.shape[1] == 1:
                image = inputs[i].reshape(inputs.shape[2], inputs.shape[3])
            else:
                image = self.deprocess_image(inputs[i])
            plt.imshow(image, cmap='gray')
            plt.savefig(self.output_dir + '/fig_' + stamp + "_" + str(i) + '.png')
            #plt.show()
            plt.close()

    def visualize_latent_layer_scatter(self, mu, y_values, stamp):
        plt.figure(figsize=(8, 6))
        plt.scatter(mu[:, 0], mu[:, 1], c=y_values, cmap='jet')
        plt.colorbar()
        plt.grid()
        plt.savefig(self.output_dir + '/fig_latent_' + stamp + '.png')
        #plt.show()
        plt.close()

    def visualize_canvas(self, canvas, stamp):
        plt.figure(figsize=(8, 10))
        #Xi, Yi = np.meshgrid(x_values, y_values)
        plt.imshow(canvas, origin="upper", cmap="gray")
        plt.savefig(self.output_dir + '/fig_canvas_latent_' + stamp + '.png')
        plt.close()

    def save_loss(self, lst, stamp):
        filename = self.output_dir + "/loss_" + stamp
        with open(filename, 'w') as fn:
            for i, elem in enumerate(lst):
                fn.write("{}\t{}\n".format(i+1, elem))


    # Code of this method is fully copied from
        #   https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html
    def deprocess_image(self,x):
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
