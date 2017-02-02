

from variational_autoencoder import VariationalAutoEncoder
from vae_convnet import VaeConvNet
from data import cell_data

scenario = 1

if __name__ == "__main__":
    if scenario == 1:
        # Train standard variational auoencoder
        vae = VariationalAutoEncoder()
        vae.main(data_set="mnist", n_latent=20, num_epochs=2, batch_size=100, downsampling=100)
        vae.test_vae(downsampling=10)
        vae.construct_images_from_scratch(100)

    if scenario == 2:
        # Train variational auoencoder with convolutional layers
        vae = VaeConvNet()
        vae.main(data_set="celeb_data", n_latent=20, num_epochs=100, batch_size=100, downsampling=None)
        vae.test_vae(downsampling=10)
        vae.construct_images_from_scratch(100)


    if scenario == 3:
        # Train standard variational auoencoder and visualize n_latent layer
        vae = VariationalAutoEncoder()
        vae.main(data_set="mnist", n_latent=2, num_epochs=2, batch_size=100, downsampling=100)
        vae.test_vae(downsampling=10)
        vae.construct_images_from_scratch(100)
        vae.visualize_latent_layer_unsupervised()



    if scenario == 4:
        # Run on cell data
        cell_data()



