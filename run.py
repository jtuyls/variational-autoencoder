
from variational_autoencoder import VariationalAutoEncoder
vae = VariationalAutoEncoder()
vae.main(data_set="celeb_data", num_epochs=301, batch_size=100, downsampling=None)


