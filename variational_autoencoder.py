
# This code follows the paper "Tutorial on Variational Autoencoders" by Carl Doersch
#
import time
import theano
import theano.tensor as T
import lasagne
import numpy as np

from mnist import mnist
from latent_layer import GaussianLayer
from visualize_images import visualize_mnist

class VariationalAutoEncoder(object):

    def __init__(self):
        pass

    def build_encoder(self, n_latent=20, shape=(None,1,28,28), input_var=None):
        encoder = lasagne.layers.InputLayer(shape, input_var=input_var) #(*, 1, 28, 28)

        # encoder = lasagne.layers.Conv2DLayer(encoder, num_filters=16, filter_size=(5, 5), pad='same',
        # nonlinearity=lasagne.nonlinearities.rectify,
        # W=lasagne.init.Normal()) #(*, 16, 28, 28)
        #
        # encoder = lasagne.layers.Conv2DLayer(encoder, num_filters=16, filter_size=(5, 5), pad='same',
        #                                      nonlinearity=lasagne.nonlinearities.rectify,
        #                                      W=lasagne.init.Normal())  # (*, 16, 28, 28)

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
        decoder = lasagne.layers.DenseLayer(gaussian_merge_layer,
                        num_units=num_units,
                        nonlinearity=lasagne.nonlinearities.rectify,
                        W=lasagne.init.HeNormal(gain='relu'))

        #decoder = lasagne.layers.DenseLayer(gaussian_merge_layer,
        #                num_units=num_units,
        #                nonlinearity=lasagne.nonlinearities.sigmoid,
        #                W=lasagne.init.HeNormal(gain='relu'))

        decoder = lasagne.layers.ReshapeLayer(decoder, shape=shape) #(*, 1, 28, 28)

        #decoder = lasagne.layers.Conv2DLayer(decoder, num_filters=16, filter_size=(5, 5), pad='same',
        #nonlinearity=lasagne.nonlinearities.rectify,
        #W=lasagne.init.HeNormal(gain='relu')) #(*, 16, 28, 28)

        x_reconstr_mean = lasagne.layers.Conv2DLayer(decoder, num_filters=1, filter_size=(5, 5), pad='same',
        nonlinearity=lasagne.nonlinearities.sigmoid,
        W=lasagne.init.Normal()) #(*, 1, 28, 28)

        return x_reconstr_mean

    def build_vae(self, n_latent=20, shape=(None,1,28,28), input_var=None):
        mu, sigma = self.build_encoder(n_latent, shape, input_var)
        gml = GaussianMergeLayer(mu, sigma)
        return gml

    def get_kl_div(self, mu_output, log_sigma_output, distr_dimensionality):
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

    def train_vae(self, input_var, vae, encoder, inputs, num_epochs=20, learning_rate=0.001, batch_size=64):
        # Create Theano variable for output tensor
        true_output = T.tensor4('targets')

        # Loss of the total variational autoencoder, so between the generated images
        # and the true images
        output = lasagne.layers.get_output(vae)
        loss_vae = T.sum(lasagne.objectives.binary_crossentropy(output, true_output), axis=[1,2,3])
        #compute_loss_vae = theano.function([input_var, true_output], loss_vae)
        #print(compute_loss_vae(inputs, inputs))

        # Kullback-Leibler divergence
        mu, log_sigma = encoder
        mu_output = lasagne.layers.get_output(mu)
        log_sigma_output = lasagne.layers.get_output(log_sigma)
        loss_kl = self.get_kl_div(mu_output, log_sigma_output, mu.output_shape[1])
        #compute_loss_kl = theano.function([input_var], loss_kl)
        #rint(compute_loss_kl(inputs))

        # TODO IS this correct?
        # Total loss is addition of loss_kl and loss_vae
        loss = T.mean(loss_vae - loss_kl) # single value

        # update the vae according to loss
        all_params = lasagne.layers.get_all_params(vae)
        updates = lasagne.updates.adam(loss, all_params, learning_rate=learning_rate)

        # train function
        train = theano.function([input_var, true_output], loss, updates=updates)

        # output function
        get_output = theano.function([input_var], output)

        lst_loss_train = []
        print("Start training")
        for epoch in range(num_epochs):

            train_err = 0
            train_batches = 0
            start_time = time.time()
            for batch in self.iterate_minibatches(inputs, batch_size, shuffle=True):
                # Calculate batch error
                train_err += train(batch, batch)
                train_batches += 1

            lst_loss_train.append(train_err / train_batches)

            if epoch % 1 == 0:
                output = get_output(inputs[:10])
                visualize_mnist(output)

            print("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, num_epochs, time.time() - start_time))
            print("  training loss:\t\t{:.6f}".format(train_err / train_batches))


    def main(self, num_epochs=20, learning_rate=0.001, batch_size=64, downsampling=None):

        X = mnist()
        inputs = X[:downsampling] if downsampling else X

        #visualize_mnist(inputs)
        input_var = T.tensor4()

        # encoder
        encoder = self.build_encoder(input_var=input_var)
        #get_output_encoder = theano.function([input_var], lasagne.layers.get_output(encoder))

        # Gaussian layer in between encoder and decoder
        mu, log_sigma = encoder
        gml = GaussianLayer(mu, log_sigma)
        #get_output_gml = theano.function([input_var], lasagne.layers.get_output(gml))

        # decoder
        vae = self.build_decoder(gml)
        #get_output_decoder = theano.function([input_var], lasagne.layers.get_output(decoder))

        # test

        self.train_vae(input_var, vae, encoder, inputs,
                       num_epochs=num_epochs,
                       learning_rate=learning_rate,
                       batch_size=batch_size)

if __name__ == '__main__':
    vae = VariationalAutoEncoder()
    vae.main(num_epochs=20)
