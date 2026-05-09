from typing import Union, List, Tuple
import torch
import numpy as np
from jaxlib.xla_extension import ArrayImpl
import jax.numpy as jnp
from .base import ALLOWED_LIBRARIES
import torch.nn as nn
from math import floor


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


class ConvReduction(nn.Module):

    def __init__(self, kernel_size: int | Tuple[int, ...] = 3, stride: int | Tuple[int, ...] = 1, padding: int | Tuple[int, ...] = 1, channels: int = 3, ndim: int = 3):
        super().__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, ) * ndim
        self.stride = stride if isinstance(stride, tuple) else (stride, ) * ndim
        self.padding = padding if isinstance(padding, tuple) else (padding, ) * ndim
        self.channels = channels
        self.ndim = ndim

        if ndim == 3:
            self.conv = nn.Conv3d(channels, channels, kernel_size = kernel_size, stride = stride, padding = padding)
            self.reverse_conv = nn.ConvTranspose3d(channels, channels, kernel_size = kernel_size, stride = stride, padding = padding)
        elif ndim == 2:
            self.conv = nn.Conv2d(channels, channels, kernel_size = kernel_size, stride = stride, padding = padding)
            self.reverse_conv = nn.ConvTranspose2d(channels, channels, kernel_size = kernel_size, stride = stride, padding = padding)
        elif ndim == 1:
            self.conv = nn.Conv1d(channels, channels, kernel_size = kernel_size, stride = stride, padding = padding)
            self.reverse_conv = nn.ConvTranspose1d(channels, channels, kernel_size = kernel_size, stride = stride, padding = padding)
        else:
            raise ValueError(f"Invalid number of dimensions: {ndim}. Must be 1, 2 or 3.")

        self.reverse_conv.weight = torch.nn.Parameter(torch.linalg.pinv(self.conv.weight.data))
        self.reverse_conv.bias = torch.nn.Parameter(- self.conv.bias.data)

    
    def inverse_operation(self, x: torch.Tensor, input_preconv_shape: Tuple[int, ...]):

        channel_dim_index = max(i for i, s in enumerate(input_preconv_shape) if s == self.channels) if self.channels in input_preconv_shape else None
        original_shape = x.shape
        # x.shape = (batch_size, seq_len, hidden_size)

        x = x.flatten(end_dim = -2)

        input_post_conv_shape = (
            -1, 
            self.channels, 
            *[floor(1 + (s + 2*self.padding - self.kernel_size) / self.stride) for s in input_preconv_shape[-self.ndim:]]
        )

        output = self.reverse_conv(x.reshape(input_post_conv_shape))
        output = output.reshape(*original_shape[:-1], *output.shape[-self.ndim-1:])
        if channel_dim_index is not None:
            output = output.transpose(1, channel_dim_index)

        return output


    def forward(self, x: torch.Tensor):

        original_shape = x.shape

        if x.ndim > self.ndim + 2:
            x = x.flatten(end_dim = - self.ndim - 2)

        channel_dim_index = max(i for i, s in enumerate(x.shape) if s == self.channels)
        x = x.transpose(1, channel_dim_index)

        output = self.conv(x).flatten(start_dim = 1)

        output = output.reshape(*original_shape[:-self.ndim - 2], -1)
        if channel_dim_index is not None:
            output = output.transpose(1, channel_dim_index)

        return output

class FieldFunction():

    def __init__(self, *args, library: str = "numpy", **kwargs):

        self.library = library.lower()
        assert self.library in ALLOWED_LIBRARIES
        self.params = []
        self.dtype = torch.get_default_dtype() if self.library == "torch" else np.float32

    def _cast_library(self):

        if self.library == "torch":
            for i in range(len(self.params)):
                param = self.params[i]
                self.params[i] = torch.nn.Parameter(torch.as_tensor(param, dtype = self.dtype) if isinstance(param, (list, np.ndarray)) else param, requires_grad = True)
        
        if self.library == "jax":
            for i in range(len(self.params)):
                self.params[i] = jnp.asarray(self.params[i], dtype = self.dtype)

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

        if self.library == "torch":
            u = u.to(self.dtype)
        else:
            u = u.astype(self.dtype)
        
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

        if self.library == "torch":
            u = u.to(self.dtype)
        else:
            u = u.astype(self.dtype)

        return -self.rate * u
    
    
class LogisticGrowth(FieldFunction):
    
    def __init__(self, growth_rate, carrying_capacity, library):
        super().__init__(growth_rate, carrying_capacity, library = library)
        self.growth_rate = growth_rate
        self.carrying_capacity = carrying_capacity
        self.params = ParamsProxy(self, ['growth_rate', 'carrying_capacity'])
        self._cast_library()

    def __call__(self, u: Union[np.ndarray, torch.Tensor, ArrayImpl]) -> Union[np.ndarray, torch.Tensor, ArrayImpl]:

        if self.library == "torch":
            u = u.to(self.dtype)
        else:
            u = u.astype(self.dtype)

        return self.growth_rate * u * (1 - u / self.carrying_capacity)
    


class SourceSink(FieldFunction):
    
    def __init__(self, source_strength, sink_rate, library):
        super().__init__(source_strength, sink_rate, library = library)
        self.source_strength = source_strength
        self.sink_rate = sink_rate
        self.params = ParamsProxy(self, ['source_strength', 'sink_rate'])
        self._cast_library()

    def __call__(self, u: Union[np.ndarray, torch.Tensor, ArrayImpl]) -> Union[np.ndarray, torch.Tensor, ArrayImpl]:

        if self.library == "torch":
            u = u.to(self.dtype)
        else:
            u = u.astype(self.dtype)

        return self.source_strength - self.sink_rate * u
