"""
Codes for the MLPs behind VAE's encoders/decoders
"""
from typing import Dict, Union, Type, List, Any

import torch
import torch.nn as nn

vanilla_config = {
    "enc": [400],
    "mu_enc": [],
    "var_enc": [],
    "dec": [400],
    "enc_ac": nn.ReLU,  # enc_ac only uses the same activation
    "dec_ac": [nn.ReLU, nn.Sigmoid],  # we allow more activation here
}

class VNet(nn.Module):
    """
    VAE Network Architecture

    Parameters
    ----------
    dx: int, input dimension
    dh: int, latent dimension
    """
    def __init__(self, dx, dh, config=None):
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
            dec_layers.extend([nn.Linear(prev_dn, dn), config["dec_ac"][i]()])
            prev_dn = dn
        self.dec = nn.Sequential(*dec_layers)

    def encode(self, x):
        h = self.enc(x)
        return self.mu_enc(h), self.var_enc(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        return self.dec(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar