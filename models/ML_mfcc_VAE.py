from .types_ import *
import torch
from torch import nn
from abc import abstractmethod


class Encoder(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 latent_dim: List = None,
                 w_in: int = 6,
                 h_in: int = 13,
                 hidden_dims: List = None):
        super(Encoder).__init__()
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.in_channels = in_channels
        self.w_in = w_in
        self.h_in = h_in

        modules = []

        if hidden_dims is None:
            hidden_dims = [2, 4, 8]

        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.network = nn.Sequential(*modules)

        modules = []
        if latent_dim is None:
            latent_dim = [64, 32, 16]

        in_features = hidden_dims[-1] * self.w_in * self.h_in
        for l_dim in latent_dim:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_features=in_features, out_features=l_dim),
                    nn.LeakyReLU()
                )
            )
            in_features = l_dim

        self.layer_mu = nn.Sequential(*modules)
        self.layer_var = nn.Sequential(*modules)

    def forward(self, features: Tensor, **kwargs) -> Tensor:
        latent_variable = self.network(features)
        mu = self.layer_mu(latent_variable)
        logvar = self.layer_var(latent_variable)

        return mu, logvar


def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
    """
    Will a single z be enough ti compute the expectation
    for the loss??
    :param mu: (Tensor) Mean of the latent Gaussian
    :param logvar: (Tensor) Standard deviation of the latent Gaussian
    :return:
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return eps * std + mu


class Decoder(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 latent_dim: List = None,
                 w_in: int = 6,
                 h_in: int = 13,
                 hidden_dims: List = None):
        super(Decoder).__init__()
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.in_channels = in_channels
        self.h_in = h_in
        self.w_in = w_in

        modules = []

        if hidden_dims is None:
            self.hidden_dims = [8, 4, 2]

        for h_dim in self.hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.network = nn.Sequential(*modules)

        self.layer_mu = nn.Linear(self.latent_dim * hidden_dims[-1], w_in * h_in)
        self.layer_var = nn.Linear(self.latent_dim * hidden_dims[-1], w_in * h_in)

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        output = self.network(input)
        output = output.view(-1, 1, self.w_in, self.h_in)

        return output


class ML_VAE(nn.Module):
    def __init__(self):
        super(ML_VAE).__init__()

    def forward(self, input):
        class_mu, class_logvar = Encoder(input)
        style_mu, style_logvar = Encoder(input)
        group_mu, group_logvar = accumulate_group_evidence(style_mu,
                                                           style_logvar)
        z_c = reparameterize(class_mu, class_logvar)
        z_s = reparameterize(group_mu, group_logvar)
        output_c = Decoder(z_c)
        output_s = Decoder(z_s)
        output_s = nn.functional.relu(output_s)
        output = output_c * output_s
        loss = loss_function(class_mu, class_logvar, group_mu, group_logvar, input, output)

        return loss, output


def accumulate_group_evidence(mu: Tensor, logvar: Tensor) -> Tensor:
    """
    :param mu: N * latent dim
    :param logvar:
    :return:
    """
    rows = mu.size(0)
    cols = mu.size(1)
    var = logvar.exp()
    mu_ = torch.zeros(rows, cols)
    "reverse the value of mu and logvar"
    for i in range(rows):
        for j in range(cols):
            if mu[i][j] != 0:
                mu_[i][j] = 1 / mu[i][j]
            else:
                mu_[i][j] = 1 / 1e-7

            var[i][j] = 1 / var[i][j]

    "Calculate the value of group_var"
    group_var = var.sum(dim=1)
    for j in range(cols):
        group_var[j] = 1 / group_var[j]

    "Calculate the value of group_mu"
    sum = torch.zeros(cols)
    for i in range(rows):
        sum += torch.matmul(mu[i], var[i].diag())

    group_mu = torch.matmul(sum, group_var.diag())
    group_logvar = group_var.log()

    return group_mu, group_logvar


def loss_function(mu_1, logvar_1, mu_2, logvar_2, input, output):
    kl_divergence_1 = -0.5 * torch.sum(1 + logvar_1) + 0.5 * (torch.sum(mu_1.pow(2)) +
                                                              torch.sum(logvar_1.exp().pow(2)))
    kl_divergence_2 = -0.5 * torch.sum(1 + logvar_2) + 0.5 * (torch.sum(mu_2.pow(2)) +
                                                              torch.sum(logvar_2.exp().pow(2)))
    reconstruction = (input.exp() - output.exp()).pow(2)
    loss = -kl_divergence_1 + kl_divergence_2 + reconstruction

    return loss
