

from variational_autoencoder import VariationalAutoEncoder
from vae_convnet import VaeConvNet

scenario = 1

if __name__ == "__main__":
    if scenario == 1:
        # Train standard variational auoencoder
        vae = VariationalAutoEncoder()
        vae.main(data_set="mnist", num_epochs=10, batch_size=100, downsampling=None)

    if scenario == 2:
        # Train variational auoencoder with convolutional layers
        vae = VaeConvNet()
        vae.main(data_set="mnist", num_epochs=10000, batch_size=100, downsampling=100)


