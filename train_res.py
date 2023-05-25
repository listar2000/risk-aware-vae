from dataset import read_mnist, read_fashionmnist
from train import train_VAE, two_layer_config

import torch
from torch.utils.data import DataLoader
from utils import *

if __name__ == '__main__':

    fmnist_train, fmnist_val, mnist_test = read_fashionmnist()

    device = torch.device("cpu")

    config = {"img_size": 28 * 28,
              "latent_dim": 20,
              "layer_config": two_layer_config,
              "subsample": 1,
              "device": device,
              "batch_size": 256,
              "recon_loss_f": "mse",
              "risk_aware": "neutral",
              "risk_q": 1.0,
              "batch_aware": True,
              "save_model": True,
              "epochs": 10}

    for t in ["seeking", "abiding"]:
        for q in [0.05, 0.1, 0.2, 0.5]:
            config["risk_aware"] = t
            config["risk_q"] = q
            print(f"start training {t}, {q}")
            train_VAE(config, fmnist_train, fmnist_val)
