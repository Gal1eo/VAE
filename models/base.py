from .types_ import *
from torch import nn
from abc import abstractmethod


class BaseVAE(nn.Module):

    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size: int, current_device: int, **kwargs) -> Tensor:
        raise RuntimeWarning()

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        pass

    @abstractmethod
    def compute_Guassian(self,
                         prob: MultivariateNormal) -> Tensor:
        pass

    @abstractmethod
    def KL_Guassian(self,
                    prior: MultivariateNormal,
                    posterior: MultivariateNormal, **kwargs) -> Tensor:
        pass

    @abstractmethod



