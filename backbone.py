"""
Codes for the MLPs behind VAE's encoders/decoders
"""
from typing import Dict, Union, Type, List, Any

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as tud
import torch.nn.functional as F
import numpy as np
import datetime
from tqdm import tqdm

vanilla_config = {
    "enc": [400],
    "mu_enc": [],
    "var_enc": [],
    "dec": [400],
    "enc_ac": nn.ReLU,  # enc_ac only uses the same activation
    "dec_ac": nn.ReLU,  # we allow more activation here
    "final_ac": nn.Sigmoid,  # activation on the final level
}


class VNet(nn.Module):
    """
    VAE Network Architecture

    Parameters
    ----------
    dx: int, input dimension
    dh: int, latent dimension
    """

    def __init__(self, dx, dh, config: dict = None):
        super(VNet, self).__init__()
        if config is None:
            config = vanilla_config
        # construct encoders
        enc_layers, prev_dn = [], dx
        for dn in config["enc"]:
            enc_layers.extend([nn.Linear(prev_dn, dn), config["enc_ac"]()])
            prev_dn = dn
        self.enc = nn.Sequential(*enc_layers)

        # construct mu & var encoders
        mu_layers, var_layers = [], []
        prev_mu_dn, prev_var_dn = prev_dn, prev_dn

        for dn in config["mu_enc"]:
            mu_layers.extend([nn.Linear(prev_mu_dn, dn), config["enc_ac"]()])
            prev_mu_dn = dn
        mu_layers.append(nn.Linear(prev_mu_dn, dh))
        self.mu_enc = nn.Sequential(*mu_layers)

        for dn in config["var_enc"]:
            var_layers.extend([nn.Linear(prev_var_dn, dn), config["enc_ac"]()])
            prev_var_dn = dn
        var_layers.append(nn.Linear(prev_var_dn, dh))
        self.var_enc = nn.Sequential(*var_layers)

        prev_dn = dh
        dec_layers = []
        for i, dn in enumerate(config["dec"]):
            dec_layers.extend([nn.Linear(prev_dn, dn), config["dec_ac"]()])
            prev_dn = dn

        dec_layers.extend([nn.Linear(prev_dn, dx), config["final_ac"]()])
        self.dec = nn.Sequential(*dec_layers)

    def encode(self, x):
        h = self.enc(x)
        return self.mu_enc(h), self.var_enc(h)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        return self.dec(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var


class VAE(object):
    """Variational AutoEncoder (VAE)

    Parameters
    ----------
    n_inputs: int, feature size of input data
    n_components: int, feature size of output
    lr: float, learning rate (default: 0.001)
    batch_size: int, batch size (default: 128)
    cuda: bool, whether to use GPU if available (default: True)
    path: string, path to save trained model (default: "vae.pth")
    kkl: float, float, weight on loss term -KL(q(z|x)||p(z)) (default: 1.0)
    kv: float, weight on variance term inside -KL(q(z|x)||p(z)) (default: 1.0)
    """

    def __init__(self, n_inputs, n_components, config, lr=1.0e-3, batch_size=64,
                 device=None, folder_path="./checkpoints", save_model=False, kkl=1.0, kv=1.0,
                 recon_loss_f="bce", risk_aware='neutral', risk_q=0.5):
        self.model = VNet(n_inputs, n_components, config=config)
        self.device = device or torch.device("cpu")
        self.model.to(self.device)
        self.batch_size = batch_size
        self.lr = lr
        self.kkl = kkl
        self.kv = kv
        self.folder_path = folder_path
        self.save_model = save_model
        # reconstruction loss
        self.recon_loss_f = F.binary_cross_entropy if recon_loss_f.lower() == 'bce' else F.mse_loss
        # risk-awareness
        assert risk_aware in ['neutral', 'seeking', 'abiding']
        self.risk_aware = risk_aware
        self.risk_q = risk_q
        # initialize weights
        self.initialize()

    def fit(self, train_data, dev_data, epochs):
        """Fit VAE from data Xr
        Parameters
        ----------
        :in:
        train_data: 2d array of shape (n_data, n_dim). Training data
        develop_data: 2d array of shape (n_data, n_dim). Dev data, used for early stopping
        epochs: int, number of training epochs
        """
        train_loader = tud.DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        dev_loader = tud.DataLoader(dev_data, batch_size=self.batch_size, shuffle=True)

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        best_dev_loss = np.inf
        for epoch in range(1, epochs + 1):
            train_loss = self._train(train_loader, optimizer)
            dev_loss, _ = self._evaluate(dev_loader)
            if dev_loss < best_dev_loss:
                self._save_model()
                best_dev_loss = dev_loss
            print('Epoch: %d, train loss: %.4f, dev loss: %.4f' % (
                epoch, train_loss, dev_loss))
        return

    def transform(self, test_data, file_path=None):
        """Transform X
        Parameters
        ----------
        :in:
        X: 2d array of shape (n_data, n_dim)
        :out:
        Z: 2d array of shape (n_data, n_components)
        """
        path_to_use = file_path or self.file_path
        try:
            self.model = torch.load(path_to_use)
        except Exception as err:
            print("Error loading '%s'\n[ERROR]: %s\nUsing initial model!" % (path_to_use, err))
        test_loader = tud.DataLoader(test_data, batch_size=self.batch_size, shuffle=False)
        _, pred = self._evaluate(test_loader)
        return pred

    def _train(self, train_loader, optimizer):
        self.model.train()
        train_loss = 0
        for batch_idx, data in enumerate(tqdm(train_loader)):
            # TODO: makes ure the train loader should not involve the labels
            data = data[0].to(self.device)
            data = data.view(data.size(0), -1)
            optimizer.zero_grad()
            recon_batch, mu, log_var = self.model(data)
            loss = self._loss_function(recon_batch, data, mu, log_var)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        return train_loss / (batch_idx + 1)

    def _evaluate(self, loader):
        self.model.eval()
        loss = 0
        fs = []
        with torch.no_grad():
            for batch_idx, data in enumerate(loader):
                data = data[0].to(self.device)
                data = data.view(data.size(0), -1)
                recon_batch, mu, log_var = self.model(data)
                loss += self._loss_function(recon_batch, data, mu, log_var).item()
                fs.append(mu)
        fs = torch.cat(fs).cpu().numpy()
        return loss / (batch_idx + 1), fs

    def _loss_function(self, recon_x, x, mu, log_var):
        """
        VAE Loss
        Parameters
        ----------
        :in:
        recon_x: 2d tensor of shape (batch_size, n_dim), reconstructed input
        x: 2d tensor of shape (batch_size, n_dim), input data
        mu: 2d tensor of shape (batch_size, n_components), latent mean
        log_var: 2d tensor of shape (batch_size, n_components), latent log-variance
        :out:
        loss: 1d tensor, VAE loss
        """
        n, d = mu.shape
        recon_loss = self.recon_loss_f(recon_x, x, reduction="none")
        # apply risk-awareness changes
        if self.risk_aware == 'seeking':
            q = torch.quantile(recon_loss, self.risk_q)
            recon_loss = recon_loss[recon_loss < q]
        elif self.risk_aware == 'abiding':
            q = torch.quantile(recon_loss, 1.0 - self.risk_q)
            recon_loss = recon_loss[recon_loss > q]

        recon_loss = recon_loss.sum() / recon_loss.size(0)
        kld = -0.5 * (d + self.kv * (log_var - log_var.exp()).sum() / n - mu.pow(2).sum() / n)
        loss = recon_loss + self.kkl * kld
        return loss

    def _save_model(self):
        if not self.save_model or not self.folder_path:
            return
        if not self.file_path:
            now = datetime.datetime.now()
            self.file_path = f'{self.folder_path}/vae_{now.strftime("%Y%m%d_%H%M%S")}.pth'
        torch.save(self.model, self.file_path)

    def initialize(self):
        """
        Model Initialization
        """
        def _init_weights(m):
            if type(m) == nn.Linear:
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0)

        self.model.apply(_init_weights)
        return
