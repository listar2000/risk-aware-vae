import torch
from dataset import read_mnist
from backbone import VAE, vanilla_config

mnist_train, mnist_test = read_mnist('./data')

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

vae_model = VAE(28 * 28, 20, vanilla_config, device=device, risk_aware="neutral")
vae_model.fit(mnist_train, mnist_test, epochs=10)