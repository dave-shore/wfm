from typing import Union
import torch
import numpy as np


class FitzhughNagumo():

    def __init__(self, gamma, p) -> None:
        self.gamma = gamma
        self.p = p
        
        if isinstance(gamma, (np.ndarray, np.matrix, torch.Tensor)) or isinstance(p, (np.ndarray, np.matrix, torch.Tensor)):
            if isinstance(p, float):
                self.p = np.full_like(gamma, p)
            elif isinstance(gamma, float):
                self.gamma = np.full_like(p, gamma)

            self.dim = min(gamma.shape)
            self.gamma = self.gamma[:self.dim,:self.dim]
            self.p = self.p[:self.dim, :1]
        else:
            self.dim = 1

    def __call__(self, v: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        
        if self.dim > 1:
            return self.gamma @ v * (v - self.p) * (1 - v)
        else:
            return self.gamma * v * (v - self.p) * (1 - v)