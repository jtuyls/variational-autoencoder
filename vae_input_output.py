

import theano.tensor as T
import lasagne

from vae_convnet import VaeConvNet
from latent_layer import GaussianLayer

from data import cell_data_input_output

class VaeInputOutput(VaeConvNet):

    def __init__(self, visualization=None):
        super(VaeInputOutput, self).__init__(visualization)

    def load_data(self, data_set, downsampling):
        # data_set doesn't matter for now because this only works for cell data
        X_train, Y_train, X_val, Y_val, X_test, Y_test = cell_data_input_output()

        X_train = X_train[:downsampling] if downsampling else X_train
        Y_train = Y_train[:downsampling] if downsampling else Y_train
        X_val = X_val[:downsampling] if downsampling else X_val
        Y_val = Y_val[:downsampling] if downsampling else Y_val
        X_test = X_test[:downsampling] if downsampling else X_test
        Y_test = Y_test[:downsampling] if downsampling else Y_test
        return  X_train, Y_train, X_val, Y_val, X_test, Y_test


    def main(self, data_set, n_latent, num_epochs=20, learning_rate=0.001, batch_size=64, downsampling=None):
        # Load data
        self.X_train, self.Y_train, self.X_val, self.Y_val, self.X_test, self.Y_test = self.load_data(data_set=data_set,
                                                                                                    downsampling=downsampling)
        print(self.X_train.shape, self.Y_train.shape, self.X_val.shape,
              self.Y_val.shape, self.X_test.shape, self.Y_test.shape)
        # visualize_images(X_train[:1])

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
        self.train_vae(input_var=self.input_var,
                       vae=self.vae,
                       encoder=self.encoder,
                       X_train=self.X_train,
                       Y_train=self.Y_train,
                       X_val=self.X_val,
                       Y_val=self.Y_val,
                       num_epochs=num_epochs,
                       learning_rate=learning_rate,
                       batch_size=batch_size)

        # Construct images from scratch
        self.test_input_var = T.matrix()
        self.test_decoder = self.build_decoder_from_weights(weights=lasagne.layers.get_all_params(self.vae)[-6:],
                                                            input_shape=(None, self.n_latent),
                                                            output_shape=shape,
                                                            input_var=self.test_input_var)