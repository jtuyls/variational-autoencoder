

from variational_autoencoder import VariationalAutoEncoder
from vae_convnet import VaeConvNet

scenario = 2

if __name__ == "__main__":
    if scenario == 1:
        # Train standard variational autoencoder
        vae = VariationalAutoEncoder()
        vae.main(data_set="celeb_data", n_latent=20, num_epochs=100, batch_size=16, downsampling=100)
        vae.test_vae(downsampling=10)
        vae.construct_images_from_scratch(100)

    if scenario == 2:
        # Train variational autoencoder with convolutional layers
        vae = VaeConvNet()
        vae.main(data_set="celeb_data", n_latent=20, num_epochs=50, batch_size=100, downsampling=None)
        vae.test_vae(downsampling=10)
        vae.construct_images_from_scratch(100)

    if scenario == 3:
        # Train standard variational auoencoder and visualize latent layer
        vae = VariationalAutoEncoder()
        vae.main(data_set="mnist", n_latent=2, num_epochs=2, batch_size=100, downsampling=100)
        vae.test_vae(downsampling=10)
        vae.construct_images_from_scratch(100)
        vae.visualize_latent_layer_unsupervised()



