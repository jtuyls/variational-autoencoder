
import lasagne

from variational_autoencoder import VariationalAutoEncoder

class VaeConvNet(VariationalAutoEncoder):

    def build_encoder(self, n_latent=20, shape=(None, 1, 28, 28), input_var=None):
        encoder = lasagne.layers.InputLayer(shape, input_var=input_var)  # (*, 1, 28, 28)

        encoder = lasagne.layers.Conv2DLayer(encoder, num_filters=16, filter_size=(5, 5), pad='same',
                                             nonlinearity=lasagne.nonlinearities.rectify,
                                             W=lasagne.init.Normal())  # (*, 16, 28, 28)

        encoder = lasagne.layers.Conv2DLayer(encoder, num_filters=16, filter_size=(5, 5), pad='same',
                                             nonlinearity=lasagne.nonlinearities.rectify,
                                             W=lasagne.init.Normal())  # (*, 16, 28, 28)

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
        self.dec_l1 = lasagne.layers.DenseLayer(gaussian_merge_layer,
                                                num_units=num_units,
                                                nonlinearity=lasagne.nonlinearities.rectify,
                                                W=lasagne.init.Normal())

        self.dec_l2 = lasagne.layers.ReshapeLayer(self.dec_l1, shape=shape)  # (*, 1, 28, 28)

        self.dec_l3 = lasagne.layers.Conv2DLayer(self.dec_l2, num_filters=3, filter_size=(5, 5), pad='same',
                                                 nonlinearity=lasagne.nonlinearities.rectify,
                                                 W=lasagne.init.HeNormal(gain='relu'))  # (*, 16, 28, 28)

        self.dec_l4 = lasagne.layers.Conv2DLayer(self.dec_l3, num_filters=shape[1], filter_size=(5, 5), pad='same',
                                                 nonlinearity=lasagne.nonlinearities.sigmoid,
                                                 W=lasagne.init.Normal())  # (*, 1, 28, 28)

        return self.dec_l4