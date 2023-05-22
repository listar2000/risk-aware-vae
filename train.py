import torch
from torch.utils.data import DataLoader
from dataset import read_mnist
from backbone import VAE, two_layer_config
from utils import show_gen_img, show_recon_img
import matplotlib.pyplot as plt


def train_mnist(z_dim, config, device, risk_aware, epochs=10, risk_q=0.5, show_config=True, plot=True):
    mnist_train, mnist_val, mnist_test = read_mnist()
    train_dataloader = DataLoader(mnist_train, batch_size=64, shuffle=True)
    train_features, train_labels = next(iter(train_dataloader))

    model = VAE(28 * 28, z_dim, config, device=device, risk_aware=risk_aware, risk_q=risk_q)
    if show_config:
        print(model.model)
    model.fit(mnist_train, mnist_val, epochs=epochs)
    if plot:
        plt.subplot(1, 2, 1)
        show_gen_img(model, z_dim)
        plt.subplot(1, 2, 2)
        show_recon_img(model, train_features[0])
        plt.show()
    return model


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
