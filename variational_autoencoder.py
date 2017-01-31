
# This code follows the paper "Tutorial on Variational Autoencoders" by Carl Doersch
#
import time
import theano
import theano.tensor as T
import lasagne
import numpy as np

from data import mnist, celeb_data
from latent_layer import GaussianLayer
from visualization import visualize_images, compare_images, visualize_image_canvas

class VariationalAutoEncoder(object):

    def __init__(self):
        pass

    def load_data(self, data_set, downsampling):
        if data_set == "celeb_data":
            X_train, X_val, X_test = celeb_data()
        else:
            # load mnist data set
            X_train, X_val, X_test = mnist()

        X_train = X_train[:downsampling] if downsampling else X_train
        X_val = X_val[:downsampling] if downsampling else X_val
        return X_train, X_val, X_test

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
    def iterate_minibatches(self, inputs, batchsize, shuffle):
        if shuffle:
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)
        for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            yield inputs[excerpt]

    def train_vae(self, input_var, vae, encoder, X_train, X_val, num_epochs=20, learning_rate=0.001, batch_size=64):
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
            for batch in self.iterate_minibatches(X_train, batch_size, shuffle=True):
                # Calculate batch error
                train_err += train(batch, batch)
                train_batches += 1

            lst_loss_train.append(train_err / train_batches)

            val_err = 0
            val_batches = 0
            for batch in self.iterate_minibatches(X_val, batch_size, shuffle=False):
                # Calculate batch error
                val_err += val(batch, batch)
                val_batches += 1

            lst_loss_val.append(val_err / val_batches)

            if epoch % 1 == 0:
                output = get_output(X_train[:100])
                visualize_image_canvas(output, stamp="train_" + str(epoch + 1))

            print("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, num_epochs, time.time() - start_time))
            print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
            print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))

        print("Variational autoencoder trained")
        return lst_loss_train, lst_loss_val

    def test_vae(self, test_nb, test_decoder, input_var, n_latent):
        # Output of test decoder
        output = lasagne.layers.get_output(test_decoder)
        get_output = theano.function([input_var], output)

        # Sample from N(0,I)
        rng = np.random.RandomState()
        shape = (test_nb, n_latent)
        z = rng.normal(size=shape)

        # Get output for z
        constructed_images = get_output(z)
        visualize_image_canvas(constructed_images, stamp="test_construction")


    def main(self, data_set, num_epochs=20, learning_rate=0.001, batch_size=64, downsampling=None):
        # Load data
        X_train, X_val, X_test = self.load_data(data_set=data_set, downsampling=downsampling)
        print(X_train.shape, X_val.shape, X_test.shape)
        #visualize_images(X_train[:1])

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
        test_decoder = self.build_decoder_from_weights(weights=lasagne.layers.get_all_params(vae)[-4:],
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



if __name__ == '__main__':
    vae = VariationalAutoEncoder()
    vae.main(num_epochs=20)
