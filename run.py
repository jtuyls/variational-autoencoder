
from variational_autoencoder import VariationalAutoEncoder
vae = VariationalAutoEncoder()
vae.main(num_epochs=20, downsampling=10, batch_size=10)
