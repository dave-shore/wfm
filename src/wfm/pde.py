"""
Diffusion-Advection-Reaction Module

This module provides functionality to define and discretize diffusion-advection-reaction
differential equations over spherical meshes.
"""

import torch
from typing import Optional, Union, Dict, Any, List, Tuple, Type, Callable
import numpy as np
from .spherical_mesh import SphericalMesh
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
        

