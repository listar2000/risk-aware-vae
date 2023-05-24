import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import read_mnist
from backbone import VAE
from utils import show_gen_img, show_recon_img
import matplotlib.pyplot as plt


vanilla_config = {
    "enc": [400],
    "mu_enc": [],
    "var_enc": [],
    "dec": [400],
    "enc_ac": nn.ReLU,  # enc_ac only uses the same activation
    "dec_ac": nn.ReLU,  # we allow more activation here
    "final_ac": nn.Sigmoid,  # activation on the final level
}

# credit: https://github.com/lyeoni/pytorch-mnist-VAE
two_layer_config = {
    "enc": [512, 256],
    "mu_enc": [],
    "var_enc": [],
    "dec": [256, 512],
    "enc_ac": torch.nn.ReLU,  # enc_ac only uses the same activation
    "dec_ac": torch.nn.ReLU,  # we allow more activation here
    "final_ac": torch.nn.Sigmoid,  # activation on the final level
}


def train_VAE(model_config, train, val, verbose=False):
    img_size = model_config["img_size"]
    latent_dim = model_config["latent_dim"]
    layer_config = model_config["layer_config"]
    subsample = model_config["subsample"]
    device = model_config["device"]
    risk_aware = model_config["risk_aware"]
    epochs = model_config["epochs"]
    model = VAE(img_size, latent_dim, layer_config, subsample=subsample, device=device, risk_aware=risk_aware)
    if verbose:
        print(model.model)
    model.fit(train, val, epochs=epochs)
    return model
    
    
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
    vae_ra = VAE(28 * 28, z_dim, two_layer_config, device=device, batch_size=64, recon_loss_f="mse",
                    risk_aware="abiding", subsample=100, risk_q=0.9, batch_aware=False, ema_alpha=0.99)
    vae_ra.fit(mnist_train, mnist_test, epochs=7)
    # plt.subplot(1, 2, 1)
    # reconstruct_img(vae_model, train_features[0], device=device)
    # plt.subplot(1, 2, 2)
    # generate_img(vae_model, z_dim, device=device)
    vae_vanilla = VAE(28 * 28, z_dim, two_layer_config, device=device, batch_size=64, recon_loss_f="mse",
                    risk_aware="neutral", subsample=100, risk_q=0.5)
    vae_vanilla.fit(mnist_train, mnist_test, epochs=7)

    from utils import compute_recon_loss
    import numpy as np

    val_dataloader = DataLoader(mnist_val, batch_size=64, shuffle=True)

    # num of batch x batch size
    recon_samples_a, list_recon_loss_a = compute_recon_loss(vae_vanilla, val_dataloader, device)
    recon_samples_b, list_recon_loss_b = compute_recon_loss(vae_ra, val_dataloader, device)

    worst_a, worst_b = [], []
    for i in range(len(list_recon_loss_a)):
        batch_a = list_recon_loss_a[i].flatten()
        batch_b = list_recon_loss_b[i].flatten()
        q_a = np.quantile(batch_a, 0.9)
        q_b = np.quantile(batch_b, 0.9)
        worst_a.append(torch.mean(batch_a[batch_a > q_a]).item())
        worst_b.append(torch.mean(batch_b[batch_b > q_b]).item())
        # worst_a.append(torch.max(batch_a).item())
        # worst_b.append(torch.max(batch_b).item())
