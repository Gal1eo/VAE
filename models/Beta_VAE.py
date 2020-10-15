import torch
from models.base import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.bernoulli import Bernoulli
from torch.distributions import Independent

class BetaVAE(BaseVAE):

    num_iter = 0 # Global static variable to keep track of iterations

    def __init__(self,
                 in_channels: int = 1,
                 latent_dim: int = 16,
                 hidden_dims: List = None,
                 beta: int = 4,
                 gamma: float = 1000.,
                 max_capacity: int = 25,
                 capacity_max_iter: int = 1e5,
                 loss_type: str = 'B',
                 h_in: int = 28,
                 w_in: int = 28,
                 **kwargs) -> None:
        super(BetaVAE, self).__init__()

        self.latent_dim = latent_dim
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        self.h_in = h_in
        self.w_in = w_in
        self.C_max = torch.Tensor([max_capacity])
        self.C_stop_iter = capacity_max_iter



        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        # flatten will reshape the size of input, therefore the neurons will change
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)


        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1])

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.ReLU())
            )

        self.decoder = nn.Sequential(*modules)
        """
        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels=3,
                                      kernel_size=3, padding=1),
                            nn.Tanh())
        """
        self.final_layer_mu = nn.Linear(hidden_dims[-1]*16*16, 784)
        self.final_layer_be = nn.Sequential(nn.Linear(hidden_dims[-1] * 16 * 16, 784),
                                            nn.Sigmoid())
        self.final_layer_var = nn.Linear(hidden_dims[-1]*16*16, 784)

    def encode(self, input: Tensor) -> MultivariateNormal:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        logvar = self.fc_var(result)
        mu = mu.flatten()
        var = logvar.flatten()
        var = var.exp()
        var = torch.diag(var)
        distribution = MultivariateNormal(mu, var)
        return distribution

    def decode(self, z: Tensor) -> Tensor:
        """
        :param z: Samples from the distribution of q_phi
        :return: The distribution of MultivariateNormal or Bernoulli
        """

        result = self.decoder_input(z)
        result = result.view(-1, 512, 1, 1)
        result = self.decoder(result)
        result = torch.flatten(result, start_dim=1)
        # result = self.final_layer(result)
        # mu = self.final_layer_mu(result)
        # mu = mu.flatten()
        # logvar = self.final_layer_var(result)
        # logvar = logvar.flatten()
        # var = logvar.exp()
        # var = torch.diag(var)
        # distribution = MultivariateNormal(mu, var)
        probs = self.final_layer_be(result)
        probs = probs.flatten()
        distribution = Bernoulli(probs=probs)

        return distribution

    @staticmethod
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

    def forward(self, input: Tensor, **kwargs) -> Tensor:

        assert self.h_in == input.shape[2]
        assert self.w_in == input.shape[3]

        z_dist = self.encode(input)
        z = z_dist.sample()
        z = z.view(input.shape[0], -1)
        x_dist = self.decode(z)
        loss = self.loss_function(input, z_dist, x_dist)

        return z_dist, x_dist, loss

    '''
    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        self.num_iter += 1
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset

        # recons_loss =F.mse_loss(recons, input)
        recons_loss = -de

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        if self.loss_type == 'H': # https://openreview.net/forum?id=Sy2fzU9gl
            loss = recons_loss + self.beta * kld_weight * kld_loss
        elif self.loss_type == 'B': # https://arxiv.org/pdf/1804.03599.pdf
            self.C_max = self.C_max.to(input.device)
            C = torch.clamp(self.C_max/self.C_stop_iter * self.num_iter, 0, self.C_max.data[0])
            loss = recons_loss + self.gamma * kld_weight* (kld_loss - C).abs()
        else:
            raise ValueError('Undefined loss type.')

        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':kld_loss}
    '''

    def loss_function(self, x: Tensor,
                      posterior_x_z: MultivariateNormal,
                      posterior_z_x: MultivariateNormal, **kwargs) -> Tensor:

        """
        :param x: Training data
        :param prior: The true distribution of laten variable z
        :param posterior_x_z: The conditional distribution of encoder
        :param posterior_z_x: The conditional distribution of decoder
        :param kwargs:
        :return: KL divergence minus likehood
        """
        batch_size = x.shape[0]
        if torch.cuda.is_available():
            dev = 'cuda'

        else:
            dev = 'cpu'

        prior = MultivariateNormal(torch.zeros(self.latent_dim * batch_size, device=dev)
                                   , torch.eye(self.latent_dim * batch_size, device=dev))
        KLD_loss = self.KL_Guassian(posterior_x_z, prior)
        x = x.flatten()
        likelihood = torch.sum(posterior_z_x.log_prob(x))
        elbo = (-KLD_loss + likelihood)/batch_size

        return -elbo

    def KL_Guassian(self,
                    posterior: MultivariateNormal,
                    prior: MultivariateNormal, **kwargs) -> Tensor:

        """
        :param prior: The true distribution of laten variable
        :param posterior: The conditional distribution of
        laten variable given the observation of input x
        :param kwargs:
        :return: The KL divergence of D(posterior, prior)
        """
        loss_1 = -0.5 * torch.sum((prior.covariance_matrix.sum(1)).log() + (posterior.covariance_matrix.sum(1)).log())
        loss_2 = -0.5 * torch.sum(torch.sum(posterior.mean ** 2) + posterior.covariance_matrix)
        loss = loss_1 - loss_2

        return loss

    '''
    def compute_Guassian(self,
                         prob: MultivariateNormal) -> Tensor:

        """
        :param prob:
        :return:
        """

        result = -0.5 * (torch.sum(prob.mean**2)+torch.sum(prob.covariance_matrix))

        return result
    '''

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        dist = self.decode(z)
        samples = dist.sample()
        samples = samples.view(num_samples, -1)

        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[1]