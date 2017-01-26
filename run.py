
from variational_autoencoder import VariationalAutoEncoder
vae = VariationalAutoEncoder()
vae.main(data_set="mnist", num_epochs=1000, batch_size=100, downsampling=2000)


