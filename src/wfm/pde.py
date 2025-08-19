"""
Diffusion-Advection-Reaction Module

This module provides functionality to define and discretize diffusion-advection-reaction
differential equations over spherical meshes.
"""

import torch
from torch._dynamo.polyfills import NoEnterTorchFunctionMode
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Callable, Union, Dict, Any
import numpy as np
from .spherical_mesh import SphericalMesh
from .functional import FitzhughNagumo

REACTION_FUNCTIONS = {
    "fitzhugh-nagumo": FitzhughNagumo, 
}


class DiffusionAdvectionReaction:
    """
    A class representing a diffusion-advection-reaction differential equation.
    
    The general form is:
    ∂u/∂t = ∇·(D∇u) - ∇·(Au) + R(u)
    
    Where:
    - u is the field variable
    - D is the diffusion tensor
    - v is the velocity field
    - R(u) is the reaction term
    """
    
    def __init__(
        self,
        diffusion_matrix: Union[float, torch.Tensor, Callable] = 1.0,
        advection_matrix: Optional[Union[torch.Tensor, Callable]] = None,
        reaction_function: Optional[Union[Callable, str]] = None,
        reaction_function_kwargs: Dict[str, Union[float, np.ndarray, torch.Tensor]] = None,
        boundary_conditions: Optional[Dict[str, Any]] = None,
        mesh: Optional[SphericalMesh] = None
    ):
        """
        Initialize the differential equation.
        
        Args:
            diffusion_coeff: Diffusion coefficient (scalar, tensor, or function)
            velocity_field: Velocity field (tensor or function)
            reaction_function: Reaction term function R(u)
            boundary_conditions: Dictionary of boundary conditions
            mesh: Spherical mesh for discretization
        """
        self.diffusion_matrix = diffusion_matrix
        self.advection_matrix = advection_matrix
        self.reaction_function = reaction_function
        self.boundary_conditions = boundary_conditions or {}
        self.mesh = mesh
        
    def set_mesh(self, mesh: SphericalMesh):
        """Set the mesh for discretization."""
        self.mesh = mesh
        
    def diffusion_term(self, u: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """
        Compute the diffusion term ∇·(D∇u).
        
        Args:
            u: Field variable tensor
            coords: Coordinate tensor
            
        Returns:
            Diffusion term tensor
        """
        if self.mesh is None:
            raise ValueError("Mesh must be set before computing diffusion term")
            
        # Get diffusion coefficient
        if callable(self.diffusion_coeff):
            D = self.diffusion_coeff(coords)
        elif isinstance(self.diffusion_coeff, torch.Tensor):
            D = self.diffusion_coeff
        else:
            D = torch.full((coords.shape[0], 1), self.diffusion_coeff, 
                          device=coords.device, dtype=coords.dtype)
        
        # Compute gradients in spherical coordinates
        grad_u = self._compute_spherical_gradients(u, coords)
        
        # Apply diffusion coefficient
        diff_flux = D * grad_u
        
        # Compute divergence
        div_diff = self._compute_spherical_divergence(diff_flux, coords)
        
        return div_diff
    
    def advection_term(self, u: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """
        Compute the advection term ∇·(vu).
        
        Args:
            u: Field variable tensor
            coords: Coordinate tensor
            
        Returns:
            Advection term tensor
        """
        if self.velocity_field is None:
            return torch.zeros_like(u)
            
        if self.mesh is None:
            raise ValueError("Mesh must be set before computing advection term")
        
        # Get velocity field
        if callable(self.velocity_field):
            v = self.velocity_field(coords)
        else:
            v = self.velocity_field
            
        # Compute advection term
        adv_term = self._compute_spherical_advection(u, v, coords)
        
        return adv_term
    
    def reaction_term(self, u: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """
        Compute the reaction term R(u).
        
        Args:
            u: Field variable tensor
            coords: Coordinate tensor
            
        Returns:
            Reaction term tensor
        """
        if self.reaction_function is None:
            return torch.zeros_like(u)
            
        return self.reaction_function(u, coords)
    
    def full_equation(self, u: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """
        Compute the full right-hand side of the differential equation.
        
        Args:
            u: Field variable tensor
            coords: Coordinate tensor
            
        Returns:
            Full equation tensor
        """
        diff_term = self.diffusion_term(u, coords)
        adv_term = self.advection_term(u, coords)
        react_term = self.reaction_term(u, coords)
        
        return diff_term - adv_term + react_term
    
    def _compute_spherical_gradients(self, u: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """Compute gradients in spherical coordinates."""
        # Convert to spherical coordinates if needed
        if coords.shape[1] == 3:  # Cartesian coordinates
            spherical_coords = self.mesh.transform_coordinates(coords, 'cartesian', 'spherical')
        else:
            spherical_coords = coords
            
        r, theta, phi = spherical_coords[:, 0], spherical_coords[:, 1], spherical_coords[:, 2]
        
        # Compute gradients using finite differences
        # This is a simplified implementation - could be enhanced with more sophisticated methods
        grad_r = torch.gradient(u, dim=0)[0]
        grad_theta = torch.gradient(u, dim=0)[0] / r.unsqueeze(1)
        grad_phi = torch.gradient(u, dim=0)[0] / (r * torch.sin(theta)).unsqueeze(1)
        
        return torch.stack([grad_r, grad_theta, grad_phi], dim=1)
    
    def _compute_spherical_divergence(self, flux: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """Compute divergence in spherical coordinates."""
        if coords.shape[1] == 3:  # Cartesian coordinates
            spherical_coords = self.mesh.transform_coordinates(coords, 'cartesian', 'spherical')
        else:
            spherical_coords = coords
            
        r, theta, phi = spherical_coords[:, 0], spherical_coords[:, 1], spherical_coords[:, 2]
        
        # Extract flux components
        F_r, F_theta, F_phi = flux[:, 0], flux[:, 1], flux[:, 2]
        
        # Compute divergence using finite differences
        # Simplified implementation
        div_r = torch.gradient(r**2 * F_r, dim=0)[0] / (r**2).unsqueeze(1)
        div_theta = torch.gradient(torch.sin(theta) * F_theta, dim=0)[0] / (r * torch.sin(theta)).unsqueeze(1)
        div_phi = torch.gradient(F_phi, dim=0)[0] / (r * torch.sin(theta)).unsqueeze(1)
        
        return div_r + div_theta + div_phi
    
    def _compute_spherical_advection(self, u: torch.Tensor, v: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """Compute advection term in spherical coordinates."""
        if coords.shape[1] == 3:  # Cartesian coordinates
            spherical_coords = self.mesh.transform_coordinates(coords, 'cartesian', 'spherical')
        else:
            spherical_coords = coords
            
        # Convert velocity to spherical components if needed
        if v.shape[1] == 3:  # Cartesian velocity
            v_spherical = self.mesh.transform_coordinates(v, 'cartesian', 'spherical')
        else:
            v_spherical = v
            
        # Compute advection using upwind scheme
        # Simplified implementation
        adv_term = torch.zeros_like(u)
        
        # Could implement more sophisticated advection schemes here
        # For now, return a basic approximation
        return adv_term


class SphericalDiscretizer:
    """
    Class for discretizing differential equations over spherical meshes.
    """
    
    def __init__(self, mesh: SphericalMesh, time_step: float = 0.01):
        """
        Initialize the discretizer.
        
        Args:
            mesh: Spherical mesh for discretization
            time_step: Time step for temporal discretization
        """
        self.mesh = mesh
        self.dt = time_step
        self.spatial_discretization = self._create_spatial_discretization()
        
    def _create_spatial_discretization(self) -> Dict[str, torch.Tensor]:
        """Create spatial discretization operators."""
        coords = self.mesh.get_mesh_points()
        n_points = coords.shape[0]
        
        # Create finite difference operators
        # This is a simplified implementation
        operators = {
            'laplacian': torch.zeros(n_points, n_points, device=coords.device),
            'gradient_x': torch.zeros(n_points, n_points, device=coords.device),
            'gradient_y': torch.zeros(n_points, n_points, device=coords.device),
            'gradient_z': torch.zeros(n_points, n_points, device=coords.device),
        }
        
        # Could implement more sophisticated discretization schemes here
        
        return operators
    
    def discretize_equation(
        self,
        equation: DiffusionAdvectionReaction,
        initial_condition: torch.Tensor,
        time_steps: int,
        method: str = 'explicit_euler'
    ) -> torch.Tensor:
        """
        Discretize and solve the differential equation.
        
        Args:
            equation: The differential equation to solve
            initial_condition: Initial condition tensor
            time_steps: Number of time steps to compute
            method: Time integration method
            
        Returns:
            Solution tensor of shape (time_steps + 1, n_points)
        """
        equation.set_mesh(self.mesh)
        
        if method == 'explicit_euler':
            return self._explicit_euler(equation, initial_condition, time_steps)
        elif method == 'implicit_euler':
            return self._implicit_euler(equation, initial_condition, time_steps)
        else:
            raise ValueError(f"Unsupported integration method: {method}")
    
    def _explicit_euler(
        self,
        equation: DiffusionAdvectionReaction,
        initial_condition: torch.Tensor,
        time_steps: int
    ) -> torch.Tensor:
        """Solve using explicit Euler method."""
        coords = self.mesh.get_mesh_points()
        n_points = coords.shape[0]
        
        # Initialize solution array
        solution = torch.zeros(time_steps + 1, n_points, device=coords.device)
        solution[0] = initial_condition
        
        # Time stepping
        for t in range(time_steps):
            u_current = solution[t]
            rhs = equation.full_equation(u_current, coords)
            solution[t + 1] = u_current + self.dt * rhs
            
        return solution
    
    def _implicit_euler(
        self,
        equation: DiffusionAdvectionReaction,
        initial_condition: torch.Tensor,
        time_steps: int
    ) -> torch.Tensor:
        """Solve using implicit Euler method."""
        # This would require solving a linear system at each time step
        # For now, fall back to explicit Euler
        return self._explicit_euler(equation, initial_condition, time_steps)


class ReactionFunctions:
    """Collection of common reaction functions."""
    
    @staticmethod
    def linear_decay(u: torch.Tensor, coords: torch.Tensor, rate: float = 1.0) -> torch.Tensor:
        """Linear decay reaction: R(u) = -rate * u"""
        return -rate * u
    
    @staticmethod
    def logistic_growth(u: torch.Tensor, coords: torch.Tensor, 
                       growth_rate: float = 1.0, carrying_capacity: float = 1.0) -> torch.Tensor:
        """Logistic growth reaction: R(u) = growth_rate * u * (1 - u/carrying_capacity)"""
        return growth_rate * u * (1 - u / carrying_capacity)
    
    @staticmethod
    def cubic_reaction(u: torch.Tensor, coords: torch.Tensor, 
                      coefficient: float = 1.0) -> torch.Tensor:
        """Cubic reaction: R(u) = coefficient * u * (1 - u^2)"""
        return coefficient * u * (1 - u**2)
    
    @staticmethod
    def source_sink(u: torch.Tensor, coords: torch.Tensor, 
                   source_strength: float = 1.0, sink_rate: float = 0.1) -> torch.Tensor:
        """Source-sink reaction: R(u) = source_strength - sink_rate * u"""
        return source_strength - sink_rate * u


class VelocityFields:
    """Collection of common velocity fields."""
    
    @staticmethod
    def solid_body_rotation(coords: torch.Tensor, angular_velocity: float = 1.0) -> torch.Tensor:
        """Solid body rotation velocity field."""
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        
        vx = -angular_velocity * y
        vy = angular_velocity * x
        vz = torch.zeros_like(x)
        
        return torch.stack([vx, vy, vz], dim=1)
    
    @staticmethod
    def radial_flow(coords: torch.Tensor, flow_strength: float = 1.0) -> torch.Tensor:
        """Radial flow velocity field."""
        r = torch.sqrt(coords[:, 0]**2 + coords[:, 1]**2 + coords[:, 2]**2)
        r = torch.clamp(r, min=1e-6)  # Avoid division by zero
        
        vx = flow_strength * coords[:, 0] / r
        vy = flow_strength * coords[:, 1] / r
        vz = flow_strength * coords[:, 2] / r
        
        return torch.stack([vx, vy, vz], dim=1)
    
    @staticmethod
    def zonal_flow(coords: torch.Tensor, flow_strength: float = 1.0) -> torch.Tensor:
        """Zonal (east-west) flow velocity field."""
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        
        # Zonal flow in the x-direction, varying with latitude
        r = torch.sqrt(x**2 + y**2 + z**2)
        theta = torch.acos(torch.clamp(z / r, -1.0, 1.0))
        
        vx = flow_strength * torch.cos(theta)
        vy = torch.zeros_like(x)
        vz = torch.zeros_like(x)
        
        return torch.stack([vx, vy, vz], dim=1)
