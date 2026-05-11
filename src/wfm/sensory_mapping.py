from .base import *
from .functional import ConvReduction
import torch
from typing import Tuple
import torch.nn as nn
import random
import jax.numpy as jnp
import numpy as np
import math

class SensoryFocusMatrixTorch(nn.Module):

    def __init__(self, target_shape: Tuple[int, int, int], device: torch.device):

        super().__init__()
        self.target_height, self.target_width, self.target_channels = target_shape
        self.device = device

        self.fixed_focus = torch.nn.functional.pad(
            torch.ones(self.target_height, self.target_width, device = self.device),
            (self.target_width, self.target_height, self.target_width, self.target_height),
            mode = "constant",
            value = 0
        ).reshape(1,1,self.target_height*3, self.target_width*3, 1)

        self.learnable_focus = torch.nn.Parameter(
            torch.zeros(self.target_height*3, self.target_width*3, device = self.device)
        )

        self.balance = torch.nn.Parameter(torch.ones(self.target_height*3, self.target_width*3, device = self.device))

    def forward(self, X: torch.Tensor):
        """Process sensory input to sensory focus matrix."""

        while X.dim() < 5:
            X = X.unsqueeze(0)
        # X.shape = (batch_size, sequence_length, height*3, width*3, channels)

        alpha = torch.nn.functional.sigmoid(self.balance).reshape(1,1,self.target_height*3, self.target_width*3, 1)
        beta = torch.nn.functional.tanh(self.learnable_focus).reshape(1,1,self.target_height*3, self.target_width*3, 1)

        return X * (alpha * self.fixed_focus + (1 - alpha) * beta)


class SurfaceMappingTorch(nn.Module):

    def __init__(self, target_shape: Tuple[int, int, int], fourier_dim: int, dropout: float = 0.1, rnn_type: str = "GRU", device: torch.device = torch.device("cpu")):
        """
        Learning the surface mapping: from sequence Lx12x12x3 to Sequence of D-dimensional Fourier mappings from R^2 to R^3, with total size Lx3xDx2 (D < 144 / 2 = 72)
        """

        super().__init__()

        self.target_height, self.target_width, self.target_channels = target_shape
        if fourier_dim > self.target_height * self.target_width // 2:
            raise Warning(f"Fourier dimension {fourier_dim} is greater than the maximum possible dimension {self.target_height * self.target_width // 2}")
        
        self.fourier_dim = fourier_dim
        self.device = device
        self.dropout = dropout

        # 3 learnable mappings (one for each field dimension) between input sequence and Fourier mappings
        self.conv = ConvReduction(kernel_size = (self.target_height // 3, self.target_width // 3), stride = (self.target_height // 3, self.target_width // 3), channels = self.target_channels, ndim = 2)
        D_out = np.floor((self.target_height + 2*self.conv.padding[0] - self.conv.kernel_size[0]) / self.conv.stride[0] + 1)
        H_out = np.floor((self.target_width + 2*self.conv.padding[1] - self.conv.kernel_size[1]) / self.conv.stride[1] + 1)
        self.conv_output_size = int(D_out * H_out * self.target_channels)
        self.hidden_size = int(self.conv_output_size // 2 + self.fourier_dim * 3) # Avg between conv output and Fourier mappings
        self.metamappings = nn.Sequential(
            nn.Flatten(start_dim = 2),
            getattr(nn, rnn_type)(self.conv_output_size, self.hidden_size, batch_first = True, device = self.device, dropout = dropout)
        )

        self.feature_mapping = nn.Sequential(
            nn.SiLU(),
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, self.fourier_dim * 6, device = self.device),
            nn.Dropout(dropout)
        )

        self._initialize_metamappings()

    def _initialize_metamappings(self):
        """
        Initialize the learnable mappings.
        """

        for p in self.parameters():
            if p.dim() > 1:
                method = random.choice(["xavier_uniform_", "xavier_normal_", "kaiming_uniform_", "kaiming_normal_"])
            else:
                method = "trunc_normal_"
            getattr(nn.init, method)(p)

    def forward(self, X: torch.Tensor):
        """
        Process input sequence to Fourier mappings.
        """
        while X.dim() < 5:
            X = X.unsqueeze(0)
        # X.shape = (batch_size, sequence_length, height, width, channels)

        X_conv = self.conv(X)

        rnn_mappings, _ = self.metamappings(X_conv)
        fourier_mappings = self.feature_mapping(rnn_mappings).reshape(X.shape[0], X.shape[1], 3, self.fourier_dim, 2)
        # fourier_mappings.shape = (batch_size, sequence_length, 3, fourier_dim, 2)

        return fourier_mappings


def create_peripheral_array_simple(X):
    """
    Transform array from (3,24,4,4,3) to (3,22,12,12,3) using numpy built-ins
    - Dimension 1: No padding (24→22, slice to remove 2) 
    - Dimension 2: Each 4x4 sub-tensor gets padded with previous and next (4→12)
    - Dimension 3: Padded with zeros (4→12)
    """

    batch_size, sequence_length, height, width, channels = X.shape

    # Step 1: Create shifted versions for temporal context
    prev_X = X[:, :-2, :, :, :]    # (3, 22, 4, 4, 3) - previous frames
    curr_X = X[:, 1:-1, :, :, :]   # (3, 22, 4, 4, 3) - current frames  
    next_X = X[:, 2:, :, :, :]     # (3, 22, 4, 4, 3) - next frames

    library = "torch" if isinstance(X, torch.Tensor) else "jax" if isinstance(X, jnp.ndarray) else "numpy"
    concat_function = torch.cat if library == "torch" else jnp.concatenate if library == "jax" else np.concatenate
    pad_function = torch.nn.functional.pad if library == "torch" else jnp.pad if library == "jax" else np.pad
    
    # Step 2: Concatenate along dimension 2 (axis=2): prev + curr + next
    temporal_concat = concat_function([prev_X, curr_X, next_X], axis=2)  # (3, 22, 12, 4, 3)
    
    # Step 3: Pad dimension 3 (axis=3) with zeros: 4 → 12
    result = pad_function(temporal_concat, ((0,0), (0,0), (0,0), (width,width), (0,0)), 
                   mode='constant')  # (3, 22, 12, 12, 3)
    
    return result


def factorial(n: float) -> int:
    return math.factorial(max(0,int(n)))


def associated_legendre_transform(X: torch.Tensor | jnp.ndarray | np.ndarray, m: int, l: int) -> torch.Tensor | jnp.ndarray | np.ndarray:
    """
    Associated Legendre transform of a set of weights over a set of coordinates.
    """
    assert m >= 0
    assert l >= 0
    assert m <= l

    l = int(l)
    m = int(m)

    k = torch.arange(m, l + 1, dtype = torch.int64)
    b = torch.prod(torch.stack([k-i for i in range(m)]), dim = 0)
    b = b.reshape(-1,*[1 for _ in range(X.dim())])
    exp_k = k.reshape(-1,*[1 for _ in range(X.dim())]) - m
    k_min = int(k.min().item())
    if l - k_min > 0:
        binom_l_k = torch.prod(
            torch.stack([
                torch.where(k + i < l, k + i, 1)
                for i in range(l - k_min)
            ]),
            dim = 0
        )
        binom_l_k = binom_l_k / (l-k).apply_(factorial)
    else:
        binom_l_k = torch.ones_like(k)
    binom_l_k = binom_l_k.reshape(-1,*[1 for _ in range(X.dim())])
    l_plus = (k + l - 1) // 2
    l_plus_max = int(l_plus.max().item())
    if l_plus_max - l > 0:
        binom_l_plus_l = torch.prod(
            torch.stack([
                torch.where(l_plus - i > l, l_plus - i, 1)
                for i in range(l_plus_max - l)
            ]),
            dim = 0
        )
        binom_l_plus_l = binom_l_plus_l / (l_plus - l).apply_(factorial)
    else:
        binom_l_plus_l = torch.ones_like(l_plus)
    binom_l_plus_l = binom_l_plus_l.reshape(-1,*[1 for _ in range(X.dim())])
    S = torch.sum(b * binom_l_k * binom_l_plus_l * X.unsqueeze(0)**exp_k, dim = 0)

    return (-1)**m * (2**l) * (1 - X**2)**(m/2) * S


def fourier_transform(weights: torch.Tensor | jnp.ndarray | np.ndarray, coords: torch.Tensor | jnp.ndarray | np.ndarray) -> torch.Tensor | jnp.ndarray | np.ndarray:
    """
    Fourier transform of a set of weights over a set of coordinates.
    """
    assert type(weights) == type(coords)
    assert weights.dtype == coords.dtype, f"Weights and coords must have the same dtype, got {weights.dtype} and {coords.dtype}."

    batch_size, sequence_length, _, fourier_dim, _ = weights.shape
    mesh_width, mesh_height, _ = coords.shape # coords.shape = (mesh_width, mesh_height, 2)

    library = "torch" if isinstance(weights, torch.Tensor) else "jax" if isinstance(weights, jnp.ndarray) else "numpy"

    cos_function = torch.cos if library == "torch" else jnp.cos if library == "jax" else np.cos
    phi_mods = np.arange(weights.shape[-2]).reshape(1,1,-1,1)
    theta_mods = np.ones_like(phi_mods)
    frequency_mods = np.concatenate([theta_mods, phi_mods], axis = -1)
    mod_coords = np.expand_dims(coords, axis = -2) * frequency_mods
    # mod_coords.shape = (mesh_width, mesh_height, fourier_dim, 2)

    if library == "torch":
        mod_coords = torch.from_numpy(mod_coords)
    elif library == "jax":
        mod_coords = jnp.array(mod_coords)

    legendre_dim = int(np.ceil(np.sqrt(2 * fourier_dim + 1)).astype(np.int32))
    legendre_degrees = np.concatenate([np.arange(1,m+1) for m in range(legendre_dim)])
    legendre_degrees = legendre_degrees[:fourier_dim]

    cosines = cos_function(mod_coords)
    
    if library == "torch":
        cosines = cosines.to(weights.device, dtype = weights.dtype)
        legendre_degrees = torch.from_numpy(legendre_degrees).to(weights.device, dtype = weights.dtype)
    elif library == "jax":
        cosines = cosines.astype(weights.dtype)
        legendre_degrees = jnp.array(legendre_degrees).astype(weights.dtype)

    cosines = torch.stack([
        associated_legendre_transform(cosines[:,:,i,:], m = int(m), l = legendre_degrees[i - int(m)].item() + 1)
        if i > 0 else
        associated_legendre_transform(cosines[:,:,i,:], m = int(m), l = 1)
    for i,m in enumerate(legendre_degrees)], dim = -2)

    F = torch.tensordot(weights, cosines, dims = ([-2,-1],[-2,-1])) if library == "torch" else jnp.tensordot(weights, cosines, dims = ([-2,-1],[-2,-1])) if library == "jax" else np.tensordot(weights, cosines, dims = ([-2,-1],[-2,-1]))
    # F.shape = (batch_size, sequence_length, 3, mesh_width, mesh_height)

    return F
