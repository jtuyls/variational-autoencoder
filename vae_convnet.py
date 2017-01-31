
import theano
import theano.tensor as T
import lasagne

from variational_autoencoder import VariationalAutoEncoder
from latent_layer import GaussianLayer
from visualization import visualize_images, compare_images, visualize_image_canvas

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
        d1 = lasagne.layers.DenseLayer(gaussian_merge_layer,
                                                num_units=num_units,
                                                nonlinearity=lasagne.nonlinearities.rectify,
                                                W=lasagne.init.Normal())

        d2 = lasagne.layers.ReshapeLayer(d1, shape=shape)  # (*, 1, 28, 28)

        d3= lasagne.layers.Conv2DLayer(d2, num_filters=3, filter_size=(5, 5), pad='same',
                                                 nonlinearity=lasagne.nonlinearities.rectify,
                                                 W=lasagne.init.HeNormal(gain='relu'))  # (*, 16, 28, 28)

        d4 = lasagne.layers.Conv2DLayer(d3, num_filters=shape[1], filter_size=(5, 5), pad='same',
                                                 nonlinearity=lasagne.nonlinearities.sigmoid,
                                                 W=lasagne.init.Normal())  # (*, 1, 28, 28)

        return d4

    def build_decoder_from_weights(self, weights, input_shape, output_shape=(-1, 1, 28, 28), input_var=None):
        input_layer = lasagne.layers.InputLayer(input_shape, input_var=input_var)  # (*, n_latent)

        num_units = output_shape[1] * output_shape[2] * output_shape[3]
        d1 = lasagne.layers.DenseLayer(input_layer,
                                       num_units=num_units,
                                       nonlinearity=lasagne.nonlinearities.rectify,
                                       W=weights[0],
                                       b=weights[1])

        d2 = lasagne.layers.ReshapeLayer(d1, shape=output_shape)  # (*, 1, 28, 28)

        d3 = lasagne.layers.Conv2DLayer(d2,
                                        num_filters=3,
                                        filter_size=(5, 5),
                                        pad='same',
                                        nonlinearity=lasagne.nonlinearities.rectify,
                                        W=weights[2],
                                        b=weights[3])

        d4 = lasagne.layers.Conv2DLayer(d3,
                                        num_filters=output_shape[1],
                                        filter_size=(5, 5),
                                        pad='same',
                                        nonlinearity=lasagne.nonlinearities.sigmoid,
                                        W=weights[4],
                                        b=weights[5])  # (*, 1, 28, 28)
        return d4

    def main(self, data_set, num_epochs=20, learning_rate=0.001, batch_size=64, downsampling=None):
        # Load data
        X_train, X_val, X_test = self.load_data(data_set=data_set, downsampling=downsampling)
        print(X_train.shape, X_val.shape, X_test.shape)
        # visualize_images(X_train[:1])

        input_shape = X_train.shape

        # encoder
        input_var = T.tensor4()
        n_latent = 20
        shape = (None, input_shape[1], input_shape[2], input_shape[3])
        encoder = self.build_encoder(input_var=input_var, n_latent=n_latent, shape=shape)

        # Gaussian layer in between encoder and decoder
        mu, log_sigma = encoder
        gml = GaussianLayer(mu, log_sigma)

        # decoder
        shape = (-1, input_shape[1], input_shape[2], input_shape[3])
        vae = self.build_decoder(gml, shape=shape)

        # train
        self.train_vae(input_var, vae, encoder, X_train, X_val,
                       num_epochs=num_epochs,
                       learning_rate=learning_rate,
                       batch_size=batch_size)

        # Construct images from scratch
        test_input_var = T.matrix()
        test_decoder = self.build_decoder_from_weights(weights=lasagne.layers.get_all_params(vae)[-6:],
                                                       input_shape=(None, n_latent),
                                                       output_shape=shape,
                                                       input_var=test_input_var)
        self.test_vae(100, test_decoder, test_input_var, n_latent)

        # After training test on test images and visualize 10 test images
        output = lasagne.layers.get_output(vae)
        get_output = theano.function([input_var], output)
        test_input = X_test[:10]
        test_reconstructed = get_output(test_input)
        compare_images(test_input, test_reconstructed, stamp="test_compare")