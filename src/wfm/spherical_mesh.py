"""
Spherical Mesh Module

This module provides functionality to create and manipulate 3-dimensional spherical meshes
centered at the origin [0,0,0], with collision avoidance for the center and poles.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Union
import math


class SphericalMesh:
    """
    A 3D spherical mesh centered at [0,0,0] with collision avoidance.
    
    This class creates a spherical mesh by discretizing the sphere into
    latitude and longitude bands, excluding the center and poles to avoid
    numerical instabilities and collisions.
    """
    
    def __init__(
        self,
        radius: float = 1.0,
        n_lat: int = 64,
        n_lon: int = 128,
        exclude_poles: bool = True,
        pole_exclusion_angle: float = 0.1,
        center_exclusion_radius: float = 0.05,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the spherical mesh.
        
        Args:
            radius: Radius of the sphere
            n_lat: Number of latitude bands (excluding poles if exclude_poles=True)
            n_lon: Number of longitude points per latitude band
            exclude_poles: Whether to exclude points near the poles
            pole_exclusion_angle: Angle in radians to exclude from poles
            center_exclusion_radius: Radius to exclude from center
            device: PyTorch device for tensor operations
        """
        self.radius = radius
        self.n_lat = n_lat
        self.n_lon = n_lon
        self.exclude_poles = exclude_poles
        self.pole_exclusion_angle = pole_exclusion_angle
        self.center_exclusion_radius = center_exclusion_radius
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Generate mesh points
        self.points, self.indices = self._generate_mesh()
        self.cartesian_coords = self._cartesian_coordinates()
        self.spherical_coords = self._spherical_coordinates()
        
    def _generate_mesh(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate the mesh points and their indices."""
        # Create latitude angles, excluding poles if requested
        if self.exclude_poles:
            lat_start = self.pole_exclusion_angle
            lat_end = math.pi - self.pole_exclusion_angle
            lat_angles = torch.linspace(lat_start, lat_end, self.n_lat, device=self.device)
        else:
            lat_angles = torch.linspace(0, math.pi, self.n_lat, device=self.device)
        
        # Create longitude angles
        lon_angles = torch.linspace(0, 2 * math.pi, self.n_lon, device=self.device)
        
        # Create meshgrid
        lat_grid, lon_grid = torch.meshgrid(lat_angles, lon_angles, indexing='ij')
        
        # Flatten and stack
        lat_flat = lat_grid.flatten()
        lon_flat = lon_grid.flatten()
        
        # Create indices for tracking
        indices = torch.stack([lat_flat, lon_flat], dim=1)
        
        return torch.stack([lat_flat, lon_flat], dim=1), indices
    
    def _cartesian_coordinates(self) -> torch.Tensor:
        """Convert spherical coordinates to Cartesian coordinates."""
        lat = self.points[:, 0]
        lon = self.points[:, 1]
        
        x = self.radius * torch.sin(lat) * torch.cos(lon)
        y = self.radius * torch.sin(lat) * torch.sin(lon)
        z = self.radius * torch.cos(lat)
        
        return torch.stack([x, y, z], dim=1)
    
    def _spherical_coordinates(self) -> torch.Tensor:
        """Get spherical coordinates (r, theta, phi)."""
        lat = self.points[:, 0]
        lon = self.points[:, 1]
        
        r = torch.full_like(lat, self.radius)
        theta = lat  # Colatitude
        phi = lon     # Longitude
        
        return torch.stack([r, theta, phi], dim=1)
    
    def get_mesh_points(self) -> torch.Tensor:
        """Get the Cartesian coordinates of all mesh points."""
        return self.cartesian_coords
    
    def get_spherical_coords(self) -> torch.Tensor:
        """Get the spherical coordinates of all mesh points."""
        return self.spherical_coords
    
    def get_mesh_indices(self) -> torch.Tensor:
        """Get the indices of all mesh points."""
        return self.indices
    
    def get_mesh_shape(self) -> Tuple[int, int]:
        """Get the shape of the mesh (n_lat, n_lon)."""
        return (self.n_lat, self.n_lon)
    
    def get_total_points(self) -> int:
        """Get the total number of points in the mesh."""
        return self.points.shape[0]
    
    def transform_coordinates(
        self,
        points: torch.Tensor,
        from_coord: str = 'cartesian',
        to_coord: str = 'spherical'
    ) -> torch.Tensor:
        """
        Transform coordinates between different coordinate systems.
        
        Args:
            points: Input points tensor of shape (N, 3)
            from_coord: Source coordinate system ('cartesian' or 'spherical')
            to_coord: Target coordinate system ('cartesian' or 'spherical')
            
        Returns:
            Transformed coordinates tensor
        """
        if from_coord == 'cartesian' and to_coord == 'spherical':
            return self._cartesian_to_spherical(points)
        elif from_coord == 'spherical' and to_coord == 'cartesian':
            return self._spherical_to_cartesian(points)
        else:
            raise ValueError(f"Unsupported coordinate transformation: {from_coord} -> {to_coord}")
    
    def _cartesian_to_spherical(self, points: torch.Tensor) -> torch.Tensor:
        """Convert Cartesian coordinates to spherical coordinates."""
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        
        r = torch.sqrt(x**2 + y**2 + z**2)
        theta = torch.acos(torch.clamp(z / r, -1.0, 1.0))  # Colatitude
        phi = torch.atan2(y, x)  # Longitude
        
        # Handle negative phi values
        phi = torch.where(phi < 0, phi + 2 * math.pi, phi)
        
        return torch.stack([r, theta, phi], dim=1)
    
    def _spherical_to_cartesian(self, points: torch.Tensor) -> torch.Tensor:
        """Convert spherical coordinates to Cartesian coordinates."""
        r, theta, phi = points[:, 0], points[:, 1], points[:, 2]
        
        x = r * torch.sin(theta) * torch.cos(phi)
        y = r * torch.sin(theta) * torch.sin(phi)
        z = r * torch.cos(theta)
        
        return torch.stack([x, y, z], dim=1)
    
    def is_valid_point(self, points: torch.Tensor, coord_system: str = 'cartesian') -> torch.Tensor:
        """
        Check if points are valid (not in excluded regions).
        
        Args:
            points: Points to check, shape (N, 3)
            coord_system: Coordinate system of input points
            
        Returns:
            Boolean tensor indicating valid points
        """
        if coord_system == 'cartesian':
            spherical = self._cartesian_to_spherical(points)
        else:
            spherical = points
        
        r, theta, phi = spherical[:, 0], spherical[:, 1], spherical[:, 2]
        
        # Check center exclusion
        center_valid = r > self.center_exclusion_radius
        
        # Check pole exclusion
        if self.exclude_poles:
            pole_valid = (theta > self.pole_exclusion_angle) & (theta < math.pi - self.pole_exclusion_angle)
        else:
            pole_valid = torch.ones_like(center_valid, dtype=torch.bool)
        
        return center_valid & pole_valid
    
    def get_mesh_tensor(self, feature_dim: int = 1) -> torch.Tensor:
        """
        Get the mesh as a tensor with additional feature dimensions.
        
        Args:
            feature_dim: Number of feature dimensions to add
            
        Returns:
            Mesh tensor of shape (n_points, 3 + feature_dim)
        """
        features = torch.zeros(self.get_total_points(), feature_dim, device=self.device)
        return torch.cat([self.cartesian_coords, features], dim=1)


class SphericalMeshBuilder:
    """
    Builder class for creating customized spherical meshes.
    """
    
    @staticmethod
    def create_uniform_mesh(
        radius: float = 1.0,
        n_lat: int = 64,
        n_lon: int = 128,
        device: Optional[torch.device] = None
    ) -> SphericalMesh:
        """Create a uniform spherical mesh."""
        return SphericalMesh(
            radius=radius,
            n_lat=n_lat,
            n_lon=n_lon,
            exclude_poles=True,
            device=device
        )
    
    @staticmethod
    def create_adaptive_mesh(
        radius: float = 1.0,
        min_lat_bands: int = 32,
        max_lat_bands: int = 128,
        min_lon_points: int = 64,
        max_lon_points: int = 256,
        device: Optional[torch.device] = None
    ) -> SphericalMesh:
        """Create an adaptive mesh with varying resolution."""
        # Simple adaptive strategy - could be enhanced with more sophisticated algorithms
        n_lat = (min_lat_bands + max_lat_bands) // 2
        n_lon = (min_lon_points + max_lon_points) // 2
        
        return SphericalMesh(
            radius=radius,
            n_lat=n_lat,
            n_lon=n_lon,
            exclude_poles=True,
            device=device
        )
    
    @staticmethod
    def create_high_resolution_mesh(
        radius: float = 1.0,
        device: Optional[torch.device] = None
    ) -> SphericalMesh:
        """Create a high-resolution mesh for detailed simulations."""
        return SphericalMesh(
            radius=radius,
            n_lat=256,
            n_lon=512,
            exclude_poles=True,
            device=device
        )
