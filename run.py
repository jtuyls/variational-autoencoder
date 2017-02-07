import os

from visualization import Visualization
from variational_autoencoder import VariationalAutoEncoder
from vae_convnet import VaeConvNet
from vae_ffnn import VaeFfnn
from vae_convnet2 import VaeConvNet2
from vae_input_output import VaeInputOutput

scenario = 1

if __name__ == "__main__":

    if scenario == 0: # Test scenario
        output_dir = os.path.dirname(os.path.abspath(__file__)) + '/figures_scenario_0'
        viz = Visualization(output_dir=output_dir)
        vae = VaeConvNet2(visualization=viz)
        vae.main(data_set="mnist", n_latent=20, num_epochs=2, batch_size=100, downsampling=100)
        vae.test_vae(downsampling=10)
        vae.visualize_train_images_original(100)
        vae.construct_images_from_scratch(100)
        vae.visualize_latent_space()

    ###############
    #### MNIST ####
    ###############
    if scenario == 1.0: # Test scenario
        # Train standard variational autoencoder
        output_dir = os.path.dirname(os.path.abspath(__file__)) + '/figures_scenario_1'
        viz = Visualization(output_dir=output_dir)
        vae = VariationalAutoEncoder(visualization=viz)
        vae.main(data_set="mnist", n_latent=20, num_epochs=10, batch_size=100, downsampling=None)
        vae.test_vae(downsampling=10)
        vae.visualize_train_images_original(100)
        vae.construct_images_from_scratch(100)
        vae.visualize_latent_space()

    #### n_latent == 2 ####
    if scenario == 1.1:
        # Train standard variational autoencoder
        output_dir = os.path.dirname(os.path.abspath(__file__)) + '/figures_scenario_1_1'
        viz = Visualization(output_dir=output_dir)
        vae = VariationalAutoEncoder(visualization=viz)
        vae.main(data_set="mnist", n_latent=2, num_epochs=100, batch_size=100, downsampling=None)
        vae.test_vae(downsampling=10)
        vae.visualize_train_images_original(100)
        vae.construct_images_from_scratch(100)
        vae.visualize_latent_space_2nlatent()
        vae.visualize_latent_layer_unsupervised()

    if scenario == 1.2:
        # Train variational autoencoder with convolutional layers
        output_dir = os.path.dirname(os.path.abspath(__file__)) + '/figures_scenario_1_2'
        viz = Visualization(output_dir=output_dir)
        vae = VaeConvNet(visualization=viz)
        vae.main(data_set="mnist", n_latent=2, num_epochs=100, batch_size=100, downsampling=None)
        vae.test_vae(downsampling=10)
        vae.visualize_train_images_original(100)
        vae.construct_images_from_scratch(100)
        vae.visualize_latent_space_2nlatent()
        vae.visualize_latent_layer_unsupervised()

    #### n_latent == 20 ####
    if scenario == 1.3:
        # Train standard variational autoencoder
        output_dir = os.path.dirname(os.path.abspath(__file__)) + '/figures_scenario_1_3'
        viz = Visualization(output_dir=output_dir)
        vae = VariationalAutoEncoder(visualization=viz)
        vae.main(data_set="mnist", n_latent=20, num_epochs=100, batch_size=100, downsampling=None)
        vae.test_vae(downsampling=10)
        vae.visualize_train_images_original(100)
        vae.construct_images_from_scratch(100)
        vae.visualize_latent_space()

    if scenario == 1.4:
        # Train variational autoencoder with convolutional layers
        output_dir = os.path.dirname(os.path.abspath(__file__)) + '/figures_scenario_1_4'
        viz = Visualization(output_dir=output_dir)
        vae = VaeConvNet(visualization=viz)
        vae.main(data_set="mnist", n_latent=20, num_epochs=100, batch_size=100, downsampling=None)
        vae.test_vae(downsampling=10)
        vae.visualize_train_images_original(100)
        vae.construct_images_from_scratch(100)
        vae.visualize_latent_space()

    #### n_latent == 200 ####
    if scenario == 1.5:
        # Train standard variational autoencoder
        output_dir = os.path.dirname(os.path.abspath(__file__)) + '/figures_scenario_1_5'
        viz = Visualization(output_dir=output_dir)
        vae = VariationalAutoEncoder(visualization=viz)
        vae.main(data_set="mnist", n_latent=200, num_epochs=100, batch_size=100, downsampling=None)
        vae.test_vae(downsampling=10)
        vae.visualize_train_images_original(100)
        vae.construct_images_from_scratch(100)
        vae.visualize_latent_space(nx=20)

    if scenario == 1.6:
        # Train variational autoencoder with convolutional layers
        output_dir = os.path.dirname(os.path.abspath(__file__)) + '/figures_scenario_1_6'
        viz = Visualization(output_dir=output_dir)
        vae = VaeConvNet(visualization=viz)
        vae.main(data_set="mnist", n_latent=200, num_epochs=100, batch_size=100, downsampling=None)
        vae.test_vae(downsampling=10)
        vae.visualize_train_images_original(100)
        vae.construct_images_from_scratch(100)
        vae.visualize_latent_space(nx=20)

    ###########################
    #### Celebrity dataset ####
    ###########################

    if scenario == 2.0: # Test scenario
        # Train standard variational autoencoder and visualize latent layer
        output_dir = os.path.dirname(os.path.abspath(__file__)) + '/figures_scenario_2'
        viz = Visualization(output_dir=output_dir)
        vae = VariationalAutoEncoder(visualization=viz)
        vae.main(data_set="celeb_data", n_latent=2, num_epochs=2, batch_size=100, downsampling=100)
        vae.test_vae(downsampling=10)
        vae.visualize_train_images_original(100)
        vae.construct_images_from_scratch(100)
        vae.visualize_latent_space_2nlatent()
        vae.visualize_latent_layer_unsupervised()

    #### n_latent == 2 ####
    if scenario == 2.1:
        # Train standard variational autoencoder
        output_dir = os.path.dirname(os.path.abspath(__file__)) + '/figures_scenario_2_1'
        viz = Visualization(output_dir=output_dir)
        vae = VariationalAutoEncoder(visualization=viz)
        vae.main(data_set="celeb_data", n_latent=2, num_epochs=100, batch_size=100, downsampling=None)
        vae.test_vae(downsampling=10)
        vae.visualize_train_images_original(100)
        vae.construct_images_from_scratch(100)
        vae.visualize_latent_space_2nlatent()
        vae.visualize_latent_layer_unsupervised()

    if scenario == 2.2:
        # Train variational autoencoder with convolutional layers
        output_dir = os.path.dirname(os.path.abspath(__file__)) + '/figures_scenario_2_2'
        viz = Visualization(output_dir=output_dir)
        vae = VaeConvNet(visualization=viz)
        vae.main(data_set="celeb_data", n_latent=2, num_epochs=100, batch_size=100, downsampling=None)
        vae.test_vae(downsampling=10)
        vae.visualize_train_images_original(100)
        vae.construct_images_from_scratch(100)
        vae.visualize_latent_space_2nlatent()
        vae.visualize_latent_layer_unsupervised()

    #### n_latent == 20 ####
    if scenario == 2.3:
        # Train standard variational autoencoder
        output_dir = os.path.dirname(os.path.abspath(__file__)) + '/figures_scenario_2_3'
        viz = Visualization(output_dir=output_dir)
        vae = VariationalAutoEncoder(visualization=viz)
        vae.main(data_set="celeb_data", n_latent=20, num_epochs=100, batch_size=100, downsampling=None)
        vae.test_vae(downsampling=10)
        vae.visualize_train_images_original(100)
        vae.construct_images_from_scratch(100)
        vae.visualize_latent_space()

    if scenario == 2.4:
        # Train variational autoencoder with convolutional layers
        output_dir = os.path.dirname(os.path.abspath(__file__)) + '/figures_scenario_2_4'
        viz = Visualization(output_dir=output_dir)
        vae = VaeConvNet(visualization=viz)
        vae.main(data_set="celeb_data", n_latent=20, num_epochs=100, batch_size=100, downsampling=None)
        vae.test_vae(downsampling=10)
        vae.visualize_train_images_original(100)
        vae.construct_images_from_scratch(100)
        vae.visualize_latent_space()

    #### n_latent == 200 ####
    if scenario == 2.5:
        # Train standard variational autoencoder
        output_dir = os.path.dirname(os.path.abspath(__file__)) + '/figures_scenario_2_5'
        viz = Visualization(output_dir=output_dir)
        vae = VariationalAutoEncoder(visualization=viz)
        vae.main(data_set="celeb_data", n_latent=200, num_epochs=100, batch_size=100, downsampling=None)
        vae.test_vae(downsampling=10)
        vae.visualize_train_images_original(100)
        vae.construct_images_from_scratch(100)
        vae.visualize_latent_space(nx=20)

    if scenario == 2.6:
        # Train variational autoencoder with convolutional layers
        output_dir = os.path.dirname(os.path.abspath(__file__)) + '/figures_scenario_2_6'
        viz = Visualization(output_dir=output_dir)
        vae = VaeConvNet(visualization=viz)
        vae.main(data_set="celeb_data", n_latent=200, num_epochs=100, batch_size=100, downsampling=None)
        vae.test_vae(downsampling=10)
        vae.visualize_train_images_original(100)
        vae.construct_images_from_scratch(100)
        vae.visualize_latent_space(nx=20)

    ###################
    #### Cell data ####
    ###################

    if scenario == 3.0:
        # Train variational autoencoder with convolutional layers on cell data set with different inputs and outputs
        output_dir = os.path.dirname(os.path.abspath(__file__)) + '/figures_scenario_3'
        viz = Visualization(output_dir=output_dir)
        vae = VaeInputOutput(visualization=viz)
        vae.main(data_set="cell_data", n_latent=20, num_epochs=2, batch_size=100, downsampling=100)
        vae.test_vae(downsampling=10)
        vae.visualize_train_images_original(100)
        vae.construct_images_from_scratch(100)

    #### n_latent == 2 ####
    if scenario == 3.1:
        # Train standard variational autoencoder
        output_dir = os.path.dirname(os.path.abspath(__file__)) + '/figures_scenario_3_1'
        viz = Visualization(output_dir=output_dir)
        vae = VariationalAutoEncoder(visualization=viz)
        vae.main(data_set="cell_data", n_latent=2, num_epochs=100, batch_size=100, downsampling=None)
        vae.test_vae(downsampling=10)
        vae.visualize_train_images_original(100)
        vae.construct_images_from_scratch(100)
        vae.visualize_latent_space_2nlatent()
        vae.visualize_latent_layer_unsupervised()

    if scenario == 3.2:
        # Train variational autoencoder with convolutional layers
        output_dir = os.path.dirname(os.path.abspath(__file__)) + '/figures_scenario_3_2'
        viz = Visualization(output_dir=output_dir)
        vae = VaeConvNet(visualization=viz)
        vae.main(data_set="cell_data", n_latent=2, num_epochs=100, batch_size=100, downsampling=None)
        vae.test_vae(downsampling=10)
        vae.visualize_train_images_original(100)
        vae.construct_images_from_scratch(100)
        vae.visualize_latent_space_2nlatent()
        vae.visualize_latent_layer_unsupervised()

    #### n_latent == 20 ####
    if scenario == 3.3:
        # Train standard variational autoencoder
        output_dir = os.path.dirname(os.path.abspath(__file__)) + '/figures_scenario_3_3'
        viz = Visualization(output_dir=output_dir)
        vae = VariationalAutoEncoder(visualization=viz)
        vae.main(data_set="cell_data", n_latent=20, num_epochs=100, batch_size=100, downsampling=None)
        vae.test_vae(downsampling=10)
        vae.visualize_train_images_original(100)
        vae.construct_images_from_scratch(100)
        vae.visualize_latent_space()

    if scenario == 3.4:
        # Train variational autoencoder with convolutional layers
        output_dir = os.path.dirname(os.path.abspath(__file__)) + '/figures_scenario_3_4'
        viz = Visualization(output_dir=output_dir)
        vae = VaeConvNet(visualization=viz)
        vae.main(data_set="cell_data", n_latent=20, num_epochs=100, batch_size=100, downsampling=None)
        vae.test_vae(downsampling=10)
        vae.visualize_train_images_original(100)
        vae.construct_images_from_scratch(100)
        vae.visualize_latent_space()

    #### n_latent == 200 ####
    if scenario == 3.5:
        # Train standard variational autoencoder
        output_dir = os.path.dirname(os.path.abspath(__file__)) + '/figures_scenario_3_5'
        viz = Visualization(output_dir=output_dir)
        vae = VariationalAutoEncoder(visualization=viz)
        vae.main(data_set="cell_data", n_latent=200, num_epochs=100, batch_size=100, downsampling=None)
        vae.test_vae(downsampling=10)
        vae.visualize_train_images_original(100)
        vae.construct_images_from_scratch(100)
        vae.visualize_latent_space(nx=20)

    if scenario == 3.6:
        # Train variational autoencoder with convolutional layers
        output_dir = os.path.dirname(os.path.abspath(__file__)) + '/figures_scenario_3_6'
        viz = Visualization(output_dir=output_dir)
        vae = VaeConvNet(visualization=viz)
        vae.main(data_set="cell_data", n_latent=200, num_epochs=100, batch_size=100, downsampling=None)
        vae.test_vae(downsampling=10)
        vae.visualize_train_images_original(100)
        vae.construct_images_from_scratch(100)
        vae.visualize_latent_space(nx=20)



