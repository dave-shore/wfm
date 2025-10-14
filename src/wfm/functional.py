from typing import Union, List, Any
import torch
import numpy as np
from jaxlib.xla_extension import ArrayImpl
import jax.numpy as jnp
from .base import ALLOWED_LIBRARIES


class ParamsProxy:
    """A proxy class that maintains references to actual object attributes."""
    
    def __init__(self, obj, attr_names: List[str]):
        self._obj = obj
        self._attr_names = attr_names
    
    def __getitem__(self, index):
        return getattr(self._obj, self._attr_names[index])
    
    def __setitem__(self, index, value):
        setattr(self._obj, self._attr_names[index], value)
    
    def __len__(self):
        return len(self._attr_names)
    
    def __iter__(self):
        for attr_name in self._attr_names:
            yield getattr(self._obj, attr_name)
    
    def __repr__(self):
        return f"ParamsProxy({[getattr(self._obj, attr_name) for attr_name in self._attr_names]})"


class FieldFunction():

    def __init__(self, *args, library: str = "numpy", **kwargs):

        self.library = library.lower()
        assert self.library in ALLOWED_LIBRARIES
        self.params = []

    def _cast_library(self):

        if self.library == "torch":
            for i in range(len(self.params)):
                param = self.params[i]
                self.params[i] = torch.nn.Parameter(torch.as_tensor(param) if isinstance(param, (list, np.ndarray)) else param, requires_grad = True)
        
        if self.library == "jax":
            for i in range(len(self.params)):
                self.params[i] = jnp.asarray(self.params[i])

    def __call__(self, *args, **kwargs):
        pass


class FitzhughNagumo(FieldFunction):

    def __init__(self, gamma, p, library) -> None:

        super().__init__(gamma, p, library = library)
        self.gamma = gamma
        self.p = p
        
        if isinstance(gamma, (np.ndarray, np.matrix, torch.Tensor, ArrayImpl)) or isinstance(p, (np.ndarray, np.matrix, torch.Tensor, ArrayImpl)):
            if isinstance(p, float):
                self.p = np.full_like(gamma, p)
            elif isinstance(gamma, float):
                self.gamma = np.full_like(p, gamma)

            self.dim = min(gamma.shape)
            self.gamma = self.gamma[:self.dim,:self.dim]
            self.p = self.p[:1, :self.dim]
        else:
            self.dim = 1

        self.params = ParamsProxy(self, ['gamma', 'p'])
        self._cast_library()


    def __call__(self, u: Union[np.ndarray, torch.Tensor, ArrayImpl]) -> Union[np.ndarray, torch.Tensor, ArrayImpl]:
        
        if self.dim > 1:
            a = u @ self.gamma
        else:
            a = self.gamma * u
        
        b = (u - self.p) * (1 - u)
        return a * b


class LinearDecay(FieldFunction):
    
    def __init__(self, rate, library):
        super().__init__(rate, library = library)
        self.rate = rate
        self.params = ParamsProxy(self, ['rate'])
        self._cast_library()

    def __call__(self, u: Union[np.ndarray, torch.Tensor, ArrayImpl]) -> Union[np.ndarray, torch.Tensor, ArrayImpl]:

        return -self.rate * u
    
    
class LogisticGrowth(FieldFunction):
    
    def __init__(self, growth_rate, carrying_capacity, library):
        super().__init__(growth_rate, carrying_capacity, library = library)
        self.growth_rate = growth_rate
        self.carrying_capacity = carrying_capacity
        self.params = ParamsProxy(self, ['growth_rate', 'carrying_capacity'])
        self._cast_library()

    def __call__(self, u: Union[np.ndarray, torch.Tensor, ArrayImpl]) -> Union[np.ndarray, torch.Tensor, ArrayImpl]:

        return self.growth_rate * u * (1 - u / self.carrying_capacity)
    


class SourceSink(FieldFunction):
    
    def __init__(self, source_strength, sink_rate, library):
        super().__init__(source_strength, sink_rate, library = library)
        self.source_strength = source_strength
        self.sink_rate = sink_rate
        self.params = ParamsProxy(self, ['source_strength', 'sink_rate'])
        self._cast_library()

    def __call__(self, u: Union[np.ndarray, torch.Tensor, ArrayImpl]) -> Union[np.ndarray, torch.Tensor, ArrayImpl]:

        return self.source_strength - self.sink_rate * u
