# This is a draft version VAE for
# multi-level disentanglement of voice generation and transformation
from abc import ABC

import torch
from models.base import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *


class MLVAE(nn.Module, ABC):

    def __init__(self,
                 in_channels: int = 1,
                 style_dim: int = 16,
                 class_dim: int = 16,
                 hidden_dims: List = None,
                 beta: int = 4,
                 gamma: float = 1000.,
                 max_capacity: int = 25,
                 capacity_max_iter: int = 1e5,
                 loss_type: str = 'B',
                 l_in: int =200,
                 **kwargs) -> None:
        super(MLVAE, self).__init__()

        self.style_dim = style_dim
        self.class_dim = class_dim
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        self.l_in = l_in
        self.C_max = torch.Tensor([max_capacity])
        self.C_stop_iter = capacity_max_iter

        modules = []

        if hidden_dims is None:
            hidden_dims = [4, 8, 16]

        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        # Define the network of encoder
        self.class_encoder = nn.Sequential(*modules)
        self.style_encoder = nn.Sequential(*modules)
        self.class_mu = nn.Linear(self.l_in, 2)
        self.class_logvar = nn.Linear(self.l_in, 2)
        self.style_mu = nn.Linear(self.l_in, 2)
        self.style_logvar = nn.Linear(self.l_in, 2)

        # Define the network of decoder
        modules = []
        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=1,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.ReLU())
            )

        self.class_decoder = nn.Sequential(*modules)
        self.style_decoder = nn.Sequential(*modules)

    def encode(self, input: Tensor) -> Tensor:
        class_latent = self.class_encoder(input)
        style_latent = self.style_encoder(input)
        class_mu = self.class_mu(class_latent)
        class_logvar = self.class_mu(class_latent)
        style_mu = self.style_mu(style_latent)
        style_logvar = self.style_logvar(style_latent)

        return class_mu, class_logvar, style_mu, style_logvar

    def decode(self, content: Tensor, style: Tensor) -> Tensor:
        """
        :param content: Latent space of content
        :param style: Latent space of style
        :return: convolution of content and style
        The dimension of source is 1 * L1 because the content is a fixed
        obsevartion
        while            media  is N * L2
        The expected out put is N * (L1-L2+1)
        """
        source = torch.mean(self.class_decoder(content), dim=0)
        source = source.view(1, 1, -1)
        media = self.style_decoder(style)
        media = media.view(media.size(0), 1, -1)
        output = F.conv_transpose1d(media, source)
        return output

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
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

    def forward(self, input: Tensor, **kwargs) -> Tensor:

        assert self.l_in == input.shape[2]

        class_mu, class_logvar, style_mu, style_logvar = self.encode(input)
        group_mu, group_logvar = self.accumulate_group_evidence(class_mu, class_logvar)
        z_c = self.reparameterize(group_mu, group_logvar)
        z_s = self.reparameterize(style_mu, style_logvar)
        output = self.decode(z_c, z_s)
        recon_loss = (((input.flatten() - output.flatten()).abs()).pow(2)).sum()
        # eps = 1e-7
        likelihood = torch.matmul(input.flatten(), (output.flatten()).log()) \
                     + torch.matmul((1 - input.flatten()), ((1 - output.flatten()).log()))
        kld_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum()
        loss = -likelihood + kld_loss + recon_loss

        return z, output, loss

    def accumulate_group_evidence(self, mu: Tensor, logvar: Tensor) -> Tensor:
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





