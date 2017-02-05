import os

from visualization import Visualization
from variational_autoencoder import VariationalAutoEncoder
from vae_convnet import VaeConvNet

scenario = 2

if __name__ == "__main__":
    if scenario == 1:
        # Train standard variational autoencoder
        output_dir = os.path.dirname(os.path.abspath(__file__)) + '/figures_scenario_1'
        viz = Visualization(output_dir=output_dir)
        vae = VariationalAutoEncoder(visualization=viz)
        vae.main(data_set="celeb_data", n_latent=20, num_epochs=100, batch_size=16, downsampling=100)
        vae.test_vae(downsampling=10)
        vae.visualize_train_images_orginal(100)
        vae.construct_images_from_scratch(100)

    if scenario == 2:
        # Train variational autoencoder with convolutional layers
        output_dir = os.path.dirname(os.path.abspath(__file__)) + '/figures_scenario_2'
        viz = Visualization(output_dir=output_dir)
        vae = VaeConvNet(visualization=viz)
        vae.main(data_set="mnist", n_latent=20, num_epochs=2, batch_size=100, downsampling=100)
        vae.test_vae(downsampling=10)
        vae.visualize_train_images_orginal(100)
        vae.construct_images_from_scratch(100)

    if scenario == 3:
        # Train standard variational autoencoder and visualize latent layer
        output_dir = os.path.dirname(os.path.abspath(__file__)) + '/figures_scenario_3'
        viz = Visualization(output_dir=output_dir)
        vae = VariationalAutoEncoder(visualization=viz)
        vae.main(data_set="celeb_data", n_latent=2, num_epochs=2, batch_size=100, downsampling=100)
        vae.test_vae(downsampling=10)
        vae.visualize_train_images_orginal(100)
        vae.construct_images_from_scratch(100)
        vae.visualize_latent_layer_unsupervised()

    if scenario == 4:
        # Train variational autoencoder with convolutional layers and visualize latent layer
        output_dir = os.path.dirname(os.path.abspath(__file__)) + '/figures_scenario_3'
        viz = Visualization(output_dir=output_dir)
        vae = VaeConvNet(visualization=viz)
        vae.main(data_set="celeb_data", n_latent=2, num_epochs=2, batch_size=100, downsampling=100)
        vae.test_vae(downsampling=10)
        vae.visualize_train_images_orginal(100)
        vae.construct_images_from_scratch(100)
        vae.visualize_latent_layer_unsupervised()


