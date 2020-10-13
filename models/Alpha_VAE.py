import torch
from models.base import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *

class AlphaVAE(BaseVAE):

    def __init__(self):

        super(AlphaVAE, self).__init__()

        modules = []

        for hid_dim in hidden_dim:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(),
                    nn.LeakyReLU()
                )
            )

        self.layer_Conv = nn.Sequential(*modules)
        self.layer_mu   = nn.Linear(, laten_dim)
        self.layer_logvar = nn.Linear(, laten_dim)

    def encoder(self):

        self.encoder()


    def decode(self):


    def forward(sel, input: Tensor, **kwargs) -> Tensor:


    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:

    def sample(self, batch_size: int, current_device: int, **kwargs) -> Tensor:

    def generate(self, x: Tensor, **kwargs) -> Tensor:

