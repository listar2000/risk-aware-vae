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
import datetime, os
from tqdm import tqdm


class VNet(nn.Module):
    """
    VAE Network Architecture

    Parameters
    ----------
    dx: int, input dimension
    dh: int, latent dimension
    subsample: int, how many sub samples to draw for one data point
    """

    def __init__(self, dx, dh, device, subsample=1, config: dict = None):
        super(VNet, self).__init__()
        self.device = device or torch.device("cpu")
        self.subsample = subsample
        assert config is not None
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

    def reparameterize(self, mu, log_var, subsample):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn(*std.shape, subsample, device=self.device)
        return eps.mul(std.unsqueeze(-1)).add_(mu.unsqueeze(-1))

    def decode(self, z):
        # Reshape from (batch_size, latent_dim, subsample) to (batch_size * subsample, latent_dim)
        batch_size, latent_dim, subsample = z.size()
        z_reshape = z.permute(0, 2, 1).contiguous().view(-1, latent_dim)

        # Decode each "data point"
        x_hat = self.dec(z_reshape)

        # Reshape back to (batch_size, subsample, input_dim)
        input_dim = x_hat.size(-1)
        x_hat = x_hat.view(batch_size, subsample, input_dim)
        return x_hat

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var, self.subsample)
        x_hat = self.decode(z)
        return x_hat, mu, log_var

    def evaluate(self, x):
        with torch.no_grad():
            x_hat, mu, log_var = self.forward(x)
            return x_hat.mean(1).unsqueeze(1), mu, log_var


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

    def __init__(self, n_inputs, n_components, config, lr=1.0e-3, batch_size=64, subsample=1,
                 device=None, folder_path="./checkpoints", save_model=False, kkl=1.0, kv=1.0,
                 recon_loss_f="bce", risk_aware='neutral', risk_q=0.5, ema_alpha=0.9, batch_aware=False):
        self.subsample = subsample
        self.model = VNet(n_inputs, n_components, device, subsample=subsample, config=config)
        self.device = device or torch.device("cpu")
        self.model.to(self.device)
        self.batch_size = batch_size
        self.lr = lr
        self.kkl = kkl
        self.kv = kv
        self.folder_path = folder_path
        self.file_path = None
        self.save_model = save_model
        # reconstruction loss
        self.recon_loss_f = F.binary_cross_entropy if recon_loss_f.lower() == 'bce' else F.mse_loss
        # risk-awareness
        assert risk_aware in ['neutral', 'seeking', 'abiding']
        self.risk_aware = risk_aware
        self.risk_q = risk_q
        self.batch_aware = batch_aware
        # exponential moving average for quantile
        self.ema_quantile = None
        self.ema_alpha = ema_alpha  # feel free to adjust this value
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
        num_workers = min(4, os.cpu_count())
        train_loader = tud.DataLoader(train_data, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
        dev_loader = tud.DataLoader(dev_data, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)

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
                recon_batch, mu, log_var = self.model.evaluate(data)
                loss += self._loss_function(recon_batch, data, mu, log_var, True).item()
                fs.append(mu)
        fs = torch.cat(fs).cpu().numpy()
        return loss / (batch_idx + 1), fs

    def _loss_function(self, recon_x, x, mu, log_var, is_eval=False):
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
        x_expand = x.unsqueeze(1).expand_as(recon_x)
        recon_loss = self.recon_loss_f(recon_x, x_expand, reduction="none")
        recon_loss = recon_loss.sum(-1)
        # compute the current batch quantile
        if self.risk_aware in ['seeking', 'abiding'] and not is_eval:
            if not self.batch_aware:
                current_q = torch.quantile(recon_loss,
                                           self.risk_q if self.risk_aware == 'seeking' else 1.0 - self.risk_q, dim=-1)
                if self.ema_quantile is None:
                    self.ema_quantile = current_q.detach().mean()  # initialize EMA quantile

                # update the EMA quantile
                self.ema_quantile = self.ema_alpha * current_q.detach() + (1 - self.ema_alpha) * self.ema_quantile
                q_expanded = self.ema_quantile.unsqueeze(-1).expand_as(recon_loss)
                q_mask = recon_loss < q_expanded if self.risk_aware == 'seeking' else recon_loss > q_expanded
                # Calculate the filtered reconstruction loss
                filtered_recon_loss = recon_loss * q_mask
                final_recon_loss = filtered_recon_loss.sum(1) / (q_mask.sum(1) + 1e-8)

                non_zero_mask = q_mask.sum(1) > 0

                final_recon_loss = final_recon_loss[non_zero_mask].mean()
                self.ema_quantile = self.ema_quantile.mean()
            else:
                recon_loss = recon_loss.mean(dim=-1)
                current_q = torch.quantile(recon_loss,
                                           self.risk_q if self.risk_aware == 'seeking' else 1.0 - self.risk_q)
                if self.ema_quantile is None:
                    self.ema_quantile = current_q.detach().mean()  # initialize EMA quantile
                q_mask = recon_loss < self.ema_quantile if self.risk_aware == 'seeking' \
                    else recon_loss > self.ema_quantile
                self.ema_quantile = self.ema_alpha * current_q.detach() + (1 - self.ema_alpha) * self.ema_quantile

                if not q_mask.any():
                    final_recon_loss = recon_loss.mean()
                else:
                    final_recon_loss = recon_loss[q_mask].mean()
        else:
            final_recon_loss = recon_loss.mean()
        kld = -0.5 * (d + self.kv * (log_var - log_var.exp()).sum() / n - mu.pow(2).sum() / n)
        loss = final_recon_loss + self.kkl * kld
        return loss

    def _save_model(self):
        if not self.save_model or not self.folder_path:
            return
        if not self.file_path:
            self.file_path = f'{self.folder_path}/{self._create_file_name()}'
        torch.save(self.model, self.file_path)

    def _create_file_name(self):
        p1 = f"b_{self.batch_size}_lr_{self.lr}_{self.risk_aware}_{self.risk_q}"
        p2 = f"_alpha_{self.ema_alpha}_ba_{self.batch_aware}.pth"
        return p1 + p2

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
