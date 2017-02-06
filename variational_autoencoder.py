
# This code follows the paper "Tutorial on Variational Autoencoders" by Carl Doersch
#
import time
import theano
import theano.tensor as T
import lasagne
import numpy as np

from visualization import Visualization
from data import mnist, celeb_data, cell_data
from latent_layer import GaussianLayer

class VariationalAutoEncoder(object):

    def __init__(self, visualization=None):
        self.encoder = None
        self.mu = None
        self.log_sigma = None
        self.gml = None
        self.vae = None

        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_test = None

        self.visualization = visualization if visualization != None else Visualization()

        self.load_data_functions = {
            'celeb_data': celeb_data,
            'cell_data': cell_data,
            'mnist': mnist
        }

    def load_data(self, data_set, downsampling):
        if data_set in self.load_data_functions.keys():
            X_train, X_val, X_test, y_test = self.load_data_functions[data_set]()
        else:
            # load mnist data set
            X_train, X_val, X_test, y_test = mnist()

        X_train = X_train[:downsampling] if downsampling else X_train
        X_val = X_val[:downsampling] if downsampling else X_val
        return X_train, X_val, X_test, y_test

    def build_encoder(self, n_latent=20, shape=(None,1,28,28), input_var=None):
        encoder = lasagne.layers.InputLayer(shape, input_var=input_var) #(*, 1, 28, 28)

        encoder = lasagne.layers.DenseLayer(
           encoder,
           num_units=400,
           nonlinearity=lasagne.nonlinearities.rectify,
           W=lasagne.init.Normal())

        # Mean
        mu = lasagne.layers.DenseLayer(
            encoder,
            num_units=n_latent,
            nonlinearity=None,
            W=lasagne.init.Normal()) #(*, n_latent)

        # Standard deviation
        # predict log sigma instead of sigma so KLD does not become infinity later on
        log_sigma = lasagne.layers.DenseLayer(
            encoder,
            num_units=n_latent,
            nonlinearity=None,
            W=lasagne.init.Normal()) #(*, n_latent)
        return mu, log_sigma

    def build_decoder(self, gaussian_merge_layer, shape=(-1,1,28,28)):
        num_units = shape[1] * shape[2] * shape[3]
        d1 = lasagne.layers.DenseLayer(gaussian_merge_layer,
                                                num_units=num_units,
                                                nonlinearity=lasagne.nonlinearities.rectify,
                                                W=lasagne.init.Normal())

        d2 = lasagne.layers.ReshapeLayer(d1, shape=shape) #(*, 1, 28, 28)
        #
        d3 = lasagne.layers.Conv2DLayer(d2,
                                        num_filters=shape[1],
                                        filter_size=(5, 5),
                                        pad='same',
                                        nonlinearity=lasagne.nonlinearities.sigmoid,
                                        W=lasagne.init.Normal()) #(*, 1, 28, 28)

        return d3

    def build_decoder_from_weights(self, weights, input_shape, output_shape=(-1,1,28,28), input_var=None):
        input_layer = lasagne.layers.InputLayer(input_shape, input_var=input_var)  # (*, n_latent)

        num_units = output_shape[1] * output_shape[2] * output_shape[3]
        d1 = lasagne.layers.DenseLayer(input_layer,
                                      num_units=num_units,
                                      nonlinearity=lasagne.nonlinearities.rectify,
                                      W=weights[0],
                                      b=weights[1])

        d2 = lasagne.layers.ReshapeLayer(d1, shape=output_shape)  # (*, 1, 28, 28)

        d3 = lasagne.layers.Conv2DLayer(d2,
                                        num_filters=output_shape[1],
                                        filter_size=(5, 5),
                                        pad='same',
                                        nonlinearity=lasagne.nonlinearities.sigmoid,
                                        W=weights[2],
                                        b=weights[3])  # (*, 1, 28, 28)
        return d3

    '''
    Method that return the Kullcback-leible divergence
    '''
    def get_kl_div(self, mu_output, log_sigma_output):
        # Kullback-Leibler divergence:
        # For expectation of log(Q(z|X) - log(P(z|X))) see equation (7) in tutorial paper
        #kld = 0.5 * (T.sum(T.exp(log_sigma_output)) + T.dot(mu_output, mu_output.transpose()) \
        #            - distr_dimensionality - T.prod(log_sigma_output))
        kld = 0.5 * T.sum(1 + log_sigma_output - mu_output**2 - T.exp(log_sigma_output), axis=1)
        return kld

    '''
    Method to iterate inputs and targets in batches of size batchsize
    Code of this function is fully copied from
      https://github.com/Lasagne/Lasagne/tree/master/examples
    '''

    def iterate_minibatches(self, inputs, targets, batchsize, shuffle):
        assert len(inputs) == len(targets)
        if shuffle:
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)
        for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            yield inputs[excerpt], targets[excerpt]

    def train_vae(self, input_var, vae, encoder, X_train, Y_train, X_val, Y_val, num_epochs=20, learning_rate=0.001, batch_size=64):
        # Create Theano variable for output tensor
        true_output = T.tensor4('targets')

        # TRAINING FUNCTION

        # Loss of the total variational autoencoder, so between the generated images
        # and the true images
        output = lasagne.layers.get_output(vae, deterministic=False)
        loss_vae = T.sum(lasagne.objectives.binary_crossentropy(output, true_output), axis=[1,2,3])

        # Kullback-Leibler divergence
        mu, log_sigma = encoder
        mu_output = lasagne.layers.get_output(mu, deterministic=False)
        log_sigma_output = lasagne.layers.get_output(log_sigma, deterministic=False)
        loss_kl = self.get_kl_div(mu_output, log_sigma_output)

        # Total loss is loss from the reconstructed image after the decoder - Kullblack Leibler divergence
        loss = T.mean(loss_vae - loss_kl) # single value

        # update the vae according to loss
        all_params = lasagne.layers.get_all_params(vae)
        updates = lasagne.updates.adam(loss, all_params, learning_rate=learning_rate)

        # train function
        train = theano.function([input_var, true_output], loss, updates=updates)

        # VALIDATION FUNCTION

        # Loss of the total variational autoencoder, so between the generated images
        # and the true images
        val_output = lasagne.layers.get_output(vae, deterministic=True)
        val_loss_vae = T.sum(lasagne.objectives.binary_crossentropy(val_output, true_output), axis=[1, 2, 3])

        # Kullback-Leibler divergence
        val_mu_output = lasagne.layers.get_output(mu, deterministic=True)
        val_log_sigma_output = lasagne.layers.get_output(log_sigma, deterministic=True)
        val_loss_kl = self.get_kl_div(val_mu_output, val_log_sigma_output)

        # Total loss is loss from the reconstructed image after the decoder - Kullblack Leibler divergence
        val_loss = T.mean(val_loss_vae - val_loss_kl)  # single value

        # validation function
        val = theano.function([input_var, true_output], val_loss)

        # output function
        get_output = theano.function([input_var], output)


        print("Start training")
        lst_loss_train = []
        lst_loss_val = []
        for epoch in range(num_epochs):

            train_err = 0
            train_batches = 0
            start_time = time.time()
            for batch in self.iterate_minibatches(X_train, Y_train, batch_size, shuffle=True):
                inputs, targets = batch
                # Calculate batch error
                train_err += train(inputs, targets)
                train_batches += 1

            lst_loss_train.append(train_err / train_batches)

            val_err = 0
            val_batches = 0
            for batch in self.iterate_minibatches(X_val, Y_val, batch_size, shuffle=False):
                inputs, targets = batch
                # Calculate batch error
                val_err += val(inputs, targets)
                val_batches += 1

            lst_loss_val.append(val_err / val_batches)

            if epoch % 1 == 0:
                output = get_output(X_train[:100])
                self.visualization.visualize_image_canvas(output, stamp="train_" + str(epoch + 1))

            print("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, num_epochs, time.time() - start_time))
            print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
            print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))

        print("Variational autoencoder trained")
        return lst_loss_train, lst_loss_val

    def test_vae(self, downsampling=None):
        X_test = self.X_test if downsampling == None else self.X_test[:downsampling]

        # After training test on test images and visualize 10 test images
        output = lasagne.layers.get_output(self.vae)
        get_output = theano.function([self.input_var], output)
        test_reconstructed = get_output(X_test)
        self.visualization.compare_images(X_test, test_reconstructed, stamp="test_compare")

    def visualize_train_images_original(self, nb_images=100):
        self.visualization.visualize_image_canvas(self.X_train[:nb_images], stamp="test_images_original")

    def visualize_latent_space(self):
        output = lasagne.layers.get_output(self.test_decoder)
        get_output = theano.function([self.test_input_var], output)

        nx = ny = 20
        x_values = np.linspace(-3, 3, nx)
        y_values = np.linspace(-3, 3, ny)

        canvas = np.empty((28 * ny, 28 * nx))
        for i, yi in enumerate(x_values):
            for j, xi in enumerate(y_values):
                z_mu = np.array([[xi, yi]])
                # Get output for z
                constructed_image = get_output(z_mu)
                canvas[(nx - i - 1) * 28:(nx - i) * 28, j * 28:(j + 1) * 28] = constructed_image[0].reshape(28, 28)

        self.visualization.visualize_canvas(canvas=canvas)
        
    def construct_images_from_scratch(self, nb_images):
        constructed_images = self._construct_images_from_scratch(nb_images, self.test_decoder, self.test_input_var,
                                                                 self.n_latent)
        self.visualization.visualize_image_canvas(constructed_images, stamp="test_construction")

    def visualize_latent_layer_unsupervised(self):
        if self.n_latent != 2:
            raise NotImplementedError("n_latent should be equal to 2 to be visualized in a 2D diagram")
        if self.y_test == np.array([]):
            raise NotImplementedError("There should be labels for the latent layer to be visualized")

        self._visualize_latent_layer_unsupervised(self.mu, self.input_var, self.X_test, self.y_test)


    def main(self, data_set, n_latent, num_epochs=20, learning_rate=0.001, batch_size=64, downsampling=None):
        # Load data
        self.X_train, self.X_val, self.X_test, self.y_test = self.load_data(data_set=data_set, downsampling=downsampling)
        print(self.X_train.shape, self.X_val.shape, self.X_test.shape, self.y_test.shape)
        #visualize_images(X_train[:1])

        input_shape = self.X_train.shape

        # encoder
        self.input_var = T.tensor4()
        self.n_latent = n_latent
        shape = (None, input_shape[1], input_shape[2], input_shape[3])
        self.encoder = self.build_encoder(input_var=self.input_var, n_latent=self.n_latent, shape=shape)

        # Gaussian layer in between encoder and decoder
        self.mu, self.log_sigma = self.encoder
        self.gml = GaussianLayer(self.mu, self.log_sigma)

        # decoder
        shape = (-1, input_shape[1], input_shape[2], input_shape[3])
        self.vae = self.build_decoder(self.gml, shape=shape)

        # train
        self.lst_loss_train, self.lst_loss_val = self.train_vae(input_var=self.input_var,
                                                                vae=self.vae,
                                                                encoder=self.encoder,
                                                                X_train=self.X_train,
                                                                Y_train=self.X_train,
                                                                X_val=self.X_val,
                                                                Y_val=self.X_val,
                                                                num_epochs=num_epochs,
                                                                learning_rate=learning_rate,
                                                                batch_size=batch_size)

        # Test decoder to construct images from scratch
        self.test_input_var = T.matrix()
        self.test_decoder = self.build_decoder_from_weights(weights=lasagne.layers.get_all_params(self.vae)[-4:],
                                                       input_shape=(None, self.n_latent),
                                                       output_shape=shape,
                                                       input_var=self.test_input_var)

    ##########################
    #### INTERNAL METHODS ####
    ##########################

    def _visualize_latent_layer_unsupervised(self, mu, input_var, X_values, y_values):
        # Output of mu
        output = lasagne.layers.get_output(mu)
        get_output = theano.function([input_var], output)

        #
        z_mu = get_output(X_values)
        self.visualization.visualize_latent_layer_scatter(z_mu, y_values=y_values, stamp="")

    def _construct_images_from_scratch(self, test_nb, test_decoder, input_var, n_latent):
        # Output of test decoder
        output = lasagne.layers.get_output(test_decoder)
        get_output = theano.function([input_var], output)

        # Sample from N(0,I)
        rng = np.random.RandomState()
        shape = (test_nb, n_latent)
        z = np.float32(rng.normal(size=shape))

        # Get output for z
        constructed_images = get_output(z)
        return constructed_images


