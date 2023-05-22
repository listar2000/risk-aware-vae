import torch
from dataset import read_mnist
from backbone import VAE, two_layer_config
from utils import generate_img, reconstruct_img
import matplotlib.pyplot as plt

if __name__ == '__main__':
    mnist_train, mnist_val, mnist_test = read_mnist()
    train_features, _ = next(iter(mnist_train))

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    z_dim = 20
    vae_model = VAE(28 * 28, z_dim, two_layer_config, device=device, batch_size=64, recon_loss_f="bce",
                    risk_aware="seeking", subsample=100)
    vae_model.fit(mnist_train, mnist_test, epochs=5)
    plt.subplot(1, 2, 1)
    reconstruct_img(vae_model, train_features[0], device=device)
    plt.subplot(1, 2, 2)
    generate_img(vae_model, z_dim, device=device)
