"""
Diffusion-Advection-Reaction Module

This module provides functionality to define and discretize diffusion-advection-reaction
differential equations over spherical meshes.
"""

import torch
from typing import Optional, Union, Dict, Any, List, Tuple, Type, Callable
import numpy as np
from .spherical_mesh import GenericMesh, SphericalMesh
from .functional import FitzhughNagumo
from .base import *
import torch.nn as nn
import jax.numpy as jnp

REACTION_FUNCTIONS = {
    "fitzhugh-nagumo": FitzhughNagumo,
}


class PDETorch(nn.Module):
    """
    A class representing a partial differential equation of general form:
    
    ∂^2u/∂t^2 + V*∂u/∂t = ∇·(D*∇u) - ∇·(Au) + R(u)
    
    Where:
    - u is a vector field, in general from C^n to C^m
    - V is a linear velocity operator, V : C^m -> C^m
    - ∇u is the jacobian of u, ∇u \in C^{n*m}
    - D is a diffusion tensor, D \in C^{n*m}, or function D : C^n -> C^{n*m}
    - A is an advection tensor, A \in C^{m*m}, or function A : C^m -> C^{m*m}
    - R is the reaction function, R : C^m -> C^m
    """
    
    def __init__(
        self,
        velocity: Optional[torch.Tensor] = None,
        diffusion: Optional[Union[float, torch.Tensor, Type]] = None,
        diffusion_function_kwargs: Dict[str, Union[float, np.ndarray, torch.Tensor]] = None,
        advection: Optional[Union[float, torch.Tensor, Type]] = None,
        advection_function_kwargs: Dict[str, Union[float, np.ndarray, torch.Tensor]] = None,
        reaction: Optional[Union[Type, str]] = None,
        reaction_function_kwargs: Dict[str, Union[float, np.ndarray, torch.Tensor]] = None,
        boundary_conditions: Optional[List[Tuple[float, torch.Tensor, jnp.ndarray, np.ndarray]] | Callable] = None,
        mesh: Optional[SphericalMesh] = None
    ):
        """
        Initialize the differential equation.
        
        Args:
            velocity: Velocity field (tensor)
            diffusion: Diffusion coefficient (scalar, tensor, or class to instantiate)
            diffusion_function_kwargs: Keyword arguments for diffusion class instantiation
            advection: Advection coefficient (scalar, tensor, or class to instantiate)
            advection_function_kwargs: Keyword arguments for advection class instantiation
            reaction: Reaction term (class to instantiate, or string key for predefined reactions)
            reaction_function_kwargs: Keyword arguments for reaction class instantiation
            boundary_conditions: List of boundary conditions or callable
            mesh: Spherical mesh for discretization
            
        Note:
            - diffusion, advection, and reaction parameters must be classes that can be instantiated,
              not functions. Pass the class itself, and it will be instantiated with the provided kwargs.
            - For reaction, you can also pass a string key for predefined reaction functions.
        """
        super().__init__()
        
        self.V = velocity
        
        # Handle diffusion - must be a class to instantiate or a scalar/tensor
        if isinstance(diffusion, type):
            self.D = diffusion(**(diffusion_function_kwargs or {}))
        elif isinstance(diffusion, str):
            raise ValueError("Diffusion must be a class, not a string. Use the class directly.")
        else:
            self.D = diffusion  # scalar or tensor
        self.D_kwargs = diffusion_function_kwargs
        
        # Handle advection - must be a class to instantiate or a scalar/tensor  
        if isinstance(advection, type):
            self.A = advection(**(advection_function_kwargs or {}))
        elif isinstance(advection, str):
            raise ValueError("Advection must be a class, not a string. Use the class directly.")
        else:
            self.A = advection  # scalar or tensor
        self.A_kwargs = advection_function_kwargs
        
        # Handle reaction - must be a class to instantiate or a string key
        if isinstance(reaction, type):
            self.R = reaction(**(reaction_function_kwargs or {}))
        elif isinstance(reaction, str):
            if reaction in REACTION_FUNCTIONS:
                reaction_class = REACTION_FUNCTIONS[reaction]
                self.R = reaction_class(**(reaction_function_kwargs or {}))
            else:
                raise ValueError(f"Unknown reaction function: {reaction}. Available: {list(REACTION_FUNCTIONS.keys())}")
        else:
            self.R = reaction
        self.R_kwargs = reaction_function_kwargs

        self.boundary_conditions = boundary_conditions or []
        self.mesh = mesh
        
    def set_mesh(self, mesh: SphericalMesh):
        """Set the mesh for discretization."""
        self.mesh = mesh

        
class BoundaryDecoderTorch(nn.Module):

    def __init__(self, mesh, boundary_conditions, kernel_size, output_sizes, dropout: float = 0.1, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.mesh = mesh
        self.boundary_conditions = boundary_conditions
        self.dropout = dropout

         # Input to border translation (from encoder to decoder) is of shape (encoder_tokens, 3, *other_dimensions)
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 1, kernel_size=kernel_size),
            nn.BatchNorm2d(1)
        )
        # The spatial convolution outputs an encoder_tokens-long sequence of matrices of shape (H,W)

        self.reverse_conv = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 3, kernel_size=kernel_size, padding = (2*kernel_size[0]- 2, 2*kernel_size[1]-2), stride = (2,2)),
            nn.SELU(),
            nn.Dropout(self.dropout)
        )

        self.mid_proj = nn.ModuleList([
            nn.LazyLinear(size) for size in output_sizes
        ])

    def dbc_generator(self, memory: torch.Tensor, initial_state: torch.Tensor):

        if memory.dim() != 4:
            raise ValueError(f"Memory has to be a Bilinear Associative Memory tensor of shape (H,W,H,W), instead got {memory.shape}")

        K_H, K_W, V_H, V_W = memory.shape
        Q = initial_state.reshape(K_H, K_W)

        while True:
            V = torch.einsum("ij,ijkl->kl", Q, memory).unsqueeze(0)
            # shape = 1, V_H, V_W

            # Must yield the current DBC (shape = 3, V_H + kernel_size[0] - 1, V_W + kernel_size[1] - 1) and the new Q
            dbc = self.reverse_conv(V)
            for dim,proj in enumerate(self.mid_proj):
                dbc = proj(dbc.transpose(dim, -1)).transpose(dim, -1)
            yield dbc

            Q += V.squeeze()


    def forward(self):

        if not isinstance(self.mesh, GenericMesh):
            raise TypeError("The PDE must have a valid mesh.")

        if not hasattr(self.mesh, "get_boundary_points"):
            raise AttributeError("The mesh should have the 'get_boundary_points' method accessible")
        
        if isinstance(self.boundary_conditions, Callable):
            boundary_conditions = [tuple(self.boundary_conditions(bp) for bp in tup) for tup in self.mesh.get_boundary_points()]
        else:
            boundary_conditions = self.boundary_conditions

        decoder_boundary_conditions = []
        for encoder_bc in boundary_conditions:
            decoder_bc = []
            for ebc in encoder_bc:
                proj_ebc = self.spatial_conv(ebc.transpose(1,-1)).squeeze()
                bam = torch.sum(
                    proj_ebc.unsqueeze(-1).unsqueeze(-1) * proj_ebc.unsqueeze(1).unsqueeze(1),
                    dim = 0
                )
                dbc = self.dbc_generator(bam, ebc)
                decoder_bc.extend([x for x in dbc])
            decoder_boundary_conditions.append(tuple(decoder_bc))


        





