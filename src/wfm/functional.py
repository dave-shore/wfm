from typing import Union
import torch
import numpy as np
from jaxlib.xla_extension import ArrayImpl
import jax.numpy as jnp
from .base import ALLOWED_LIBRARIES


class FitzhughNagumo():

    def __init__(self, gamma, p, library) -> None:
        self.gamma = gamma
        self.p = p
        self.library = library.lower()
        assert self.library in ALLOWED_LIBRARIES
        
        if isinstance(gamma, (np.ndarray, np.matrix, torch.Tensor, ArrayImpl)) or isinstance(p, (np.ndarray, np.matrix, torch.Tensor, ArrayImpl)):
            if isinstance(p, float):
                self.p = np.full_like(gamma, p)
            elif isinstance(gamma, float):
                self.gamma = np.full_like(p, gamma)

            self.dim = min(gamma.shape)
            self.gamma = self.gamma[:self.dim,:self.dim]
            self.p = self.p[:self.dim, :1]
        else:
            self.dim = 1

        self._cast_library()

    def _cast_library(self):

        if self.library == "torch":
            self.gamma = torch.nn.Parameter(torch.from_numpy(self.gamma) if self.dim > 1 else self.gamma)
            self.p = torch.nn.Parameter(torch.from_numpy(self.p) if self.dim > 1 else self.p)
        
        if self.library == "jax":
            self.gamma = jnp.asarray(self.gamma)
            self.p = jnp.asarray(self.p)


    def __call__(self, v: Union[np.ndarray, torch.Tensor, ArrayImpl]) -> Union[np.ndarray, torch.Tensor, ArrayImpl]:
        
        if self.dim > 1:
            return self.gamma @ v * (v - self.p) * (1 - v)
        else:
            return self.gamma * v * (v - self.p) * (1 - v)