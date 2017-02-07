
import numpy as np

import lasagne
import theano.tensor as T
import theano

from latent_layer import GaussianLayer
from variational_autoencoder import VariationalAutoEncoder


class VaeFfnn(VariationalAutoEncoder):

    def __init__(self, visualization=None):
        super(VaeFfnn, self).__init__(visualization)

    def build_encoder(self, n_latent=20, shape=(None, 1, 28, 28), input_var=None):
        encoder = lasagne.layers.InputLayer(shape, input_var=input_var)  # (*, 1, 28, 28)

        encoder = lasagne.layers.DenseLayer(
            encoder,
            num_units=1000,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.Normal())

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
            W=lasagne.init.Normal())  # (*, n_latent)

        # Standard deviation
        # predict log sigma instead of sigma so KLD does not become infinity later on
        log_sigma = lasagne.layers.DenseLayer(
            encoder,
            num_units=n_latent,
            nonlinearity=None,
            W=lasagne.init.Normal())  # (*, n_latent)
        return mu, log_sigma

    def build_decoder(self, gaussian_merge_layer, shape=(-1, 1, 28, 28)):
        num_units = shape[1] * shape[2] * shape[3]

        d1 = lasagne.layers.DenseLayer(gaussian_merge_layer,
                                       num_units=400,
                                       nonlinearity=lasagne.nonlinearities.rectify,
                                       W=lasagne.init.Normal())

        d2 = lasagne.layers.DenseLayer(d1,
                                       num_units=1000,
                                       nonlinearity=lasagne.nonlinearities.rectify,
                                       W=lasagne.init.Normal())

        d3 = lasagne.layers.DenseLayer(d2,
                                       num_units=num_units,
                                       nonlinearity=None,
                                       W=lasagne.init.Normal())

        d4 = lasagne.layers.ReshapeLayer(d3, shape=shape)

        d5 = lasagne.layers.NonlinearityLayer(d4, nonlinearity=lasagne.nonlinearities.sigmoid)

        return d5

    def build_decoder_from_weights(self, weights, input_shape, output_shape=(-1, 1, 28, 28), input_var=None):
        input_layer = lasagne.layers.InputLayer(input_shape, input_var=input_var)  # (*, n_latent)

        num_units = output_shape[1] * output_shape[2] * output_shape[3]

        d1 = lasagne.layers.DenseLayer(input_layer,
                                       num_units=400,
                                       nonlinearity=lasagne.nonlinearities.rectify,
                                       W=weights[0],
                                       b=weights[1])

        d2 = lasagne.layers.DenseLayer(d1,
                                       num_units=1000,
                                       nonlinearity=lasagne.nonlinearities.rectify,
                                       W=weights[2],
                                       b=weights[3])

        d3 = lasagne.layers.DenseLayer(d2,
                                       num_units=num_units,
                                       nonlinearity=None,
                                       W=weights[4],
                                       b=weights[5])

        d4 = lasagne.layers.ReshapeLayer(d3, shape=output_shape)

        d5 = lasagne.layers.NonlinearityLayer(d4, nonlinearity=lasagne.nonlinearities.sigmoid)

        return d5

    def main(self, data_set, n_latent, num_epochs=20, optimizer_name="adam", learning_rate=0.001, batch_size=64,
             downsampling=None):
        # Load data
        self.X_train, self.X_val, self.X_test, self.y_test = self.load_data(data_set=data_set,
                                                                            downsampling=downsampling)
        print(self.X_train.shape, self.X_val.shape, self.X_test.shape, self.y_test.shape)
        # visualize_images(X_train[:1])

        input_shape = self.X_train.shape

        # encoder
        self.input_var = T.tensor4()
        self.n_latent = n_latent
        self._shape = (None, input_shape[1], input_shape[2], input_shape[3])
        self.encoder = self.build_encoder(input_var=self.input_var, n_latent=self.n_latent, shape=self._shape)

        # Gaussian layer in between encoder and decoder
        self.mu, self.log_sigma = self.encoder
        self.gml = GaussianLayer(self.mu, self.log_sigma)

        # decoder
        self.shape = (-1, input_shape[1], input_shape[2], input_shape[3])
        self.vae = self.build_decoder(self.gml, shape=self.shape)

        # train
        self.lst_loss_train, self.lst_loss_val = self.train_vae(input_var=self.input_var,
                                                                vae=self.vae,
                                                                encoder=self.encoder,
                                                                X_train=self.X_train,
                                                                Y_train=self.X_train,
                                                                X_val=self.X_val,
                                                                Y_val=self.X_val,
                                                                num_epochs=num_epochs,
                                                                optimizer_name=optimizer_name,
                                                                learning_rate=learning_rate,
                                                                batch_size=batch_size)

        # Save training and validation loss for later plotting
        self.visualization.save_loss(self.lst_loss_train, "training")
        self.visualization.save_loss(self.lst_loss_val, "validation")

        # Test decoder to construct images from scratch
        self.test_input_var = T.matrix()
        self.test_decoder = self.build_decoder_from_weights(weights=lasagne.layers.get_all_params(self.vae)[-6:],
                                                            input_shape=(None, self.n_latent),
                                                            output_shape=self.shape,
                                                            input_var=self.test_input_var)