
import lasagne
import theano
import theano.tensor as T
import numpy as np

# See https://github.com/Lasagne/Recipes/blob/master/examples/variational_autoencoder/variational_autoencoder.py
class GaussianLayer(lasagne.layers.MergeLayer):

    def __init__(self, mu, sigma, **kwargs):
        self.srng = T.shared_randomstreams.RandomStreams()
        super(GaussianLayer, self).__init__([mu, sigma], **kwargs)

    def get_output_shape_for(self, input_shapes):
        # input_shapes is a list of tuples, each tuple gives the shape of one input
        # For this merge layer, there should be two tuples in the list and both should
        # have the same shape. The output should also have the same shape
        return input_shapes[0]

    def get_output_for(self, inputs, **kwargs):
        mu, log_sigma = inputs # mu: (*, n_latent), sigma: (*, n_latent)
        shape=(inputs[0].shape[0], inputs[0].shape[1])
        eps = self.srng.normal(shape) #(*, n_latent)
        #epsilon = self.srng.normal(avg=np.zeros((epsilon_shape,epsilon_shape)),std=np.identity(epsilon_shape))
        return mu + T.exp(log_sigma) * eps #(*, n_latent)
