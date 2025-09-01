"""
Spherical Mesh Module

This module provides functionality to create and manipulate 3-dimensional spherical meshes
centered at the origin [0,0,0], with collision avoidance for the center and poles.
"""

from jax._src.ad_util import zero_from_primal
import torch
import torch.nn as nn
import jax
import numpy as np
import jax.numpy as jnp
from typing import Tuple, Optional, Union
import math
from .base import *


def spherical2cartesian(rad, lon, lat):

    x = rad * np.sin(lat) * np.cos(lon)
    y = rad * np.sin(lat) * np.sin(lon)
    z = rad * np.cos(lat)

    return x, y, z

def cartesian2spherical(x, y, z):

    rad = np.sqrt(x**2 + y**2 + z**2)
    lon = np.arctan(y / (x + EPS))
    lat = np.arccos(z / (rad + EPS))

    return rad, lon, lat


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
        n_radial: int = 100,
        n_lat: int = 180,
        n_lon: int = 360,
        exclude_poles: bool = True,
        pole_exclusion_angle: float = 0.1,
        center_exclusion_radius: float = 0.05,
        library: str = "numpy",
        dtype = np.complex64,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the spherical mesh.
        
        Args:
            radius: Radius of the sphere
            n_lat: Number of latitude bands (excluding poles if exclude_poles=True)
            n_lon: Number of longitude points per latitude band
            exclude_poles: Deprecated.
            pole_exclusion_angle: Angle in radians to exclude from poles
            center_exclusion_radius: Radius to exclude from center
            library: Can be "numpy", "torch", or "jax"
            device: PyTorch device for tensor operations
        """
        self.n_radial = n_radial
        self.max_radius = radius
        self.n_lat = n_lat
        self.n_lon = n_lon
        self.exclude_poles = (pole_exclusion_angle > 0.0)
        self.pole_exclusion_angle = pole_exclusion_angle
        self.center_exclusion_radius = center_exclusion_radius
        self.library = library.lower() if library in ALLOWED_LIBRARIES else "numpy"
        self.c_dtype = dtype
        self.r_dtype = getattr(np, f"float{np.nbytes[dtype] // 2 * 8}")
        if self.library == "torch":
            self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif self.library == "jax":
            try:
                self.device = jax.devices("gpu")[0]
            except RuntimeError:
                self.device = jax.devices("cpu")[0]
        else:
            self.device = "cpu"
        
        # Generate mesh and coordinates in NumPy, then convert once to the selected backend
        points_np, indices_np = self._generate_mesh_np()
        self.points = self._to_backend(points_np)
        self.indices = self._to_backend(indices_np)

    def _to_backend(self, array_np: np.ndarray):
        """Convert a NumPy array to the selected backend (torch/jax/numpy)."""
        
        if self.library == "torch":
            t = torch.from_numpy(array_np)
            return t.to(self.device) if isinstance(self.device, torch.device) else t
        elif self.library == "jax":
            return jax.device_put(array_np, self.device) if self.device is not None else jnp.array(array_np)
        else:
            return array_np

    def _generate_mesh_np(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate mesh latitude/longitude points and indices using NumPy only."""
        lat_start = self.pole_exclusion_angle
        lat_end = math.pi - self.pole_exclusion_angle
        lat_angles = np.linspace(lat_start, lat_end, self.n_lat, dtype=self.r_dtype)

        lon_angles = np.linspace(0.0, 2 * math.pi, self.n_lon, dtype=self.r_dtype)
        radii = np.linspace(self.center_exclusion_radius, self.max_radius, self.n_radial, dtype = self.r_dtype)

        rad_grid, lon_grid, lat_grid = np.meshgrid(radii, lon_angles, lat_angles, indexing='ij')

        rad_flat = rad_grid.reshape(self.n_radial, self.n_lon, self.n_lat)
        lat_flat = lat_grid.reshape(self.n_radial, self.n_lon, self.n_lat)
        lon_flat = lon_grid.reshape(self.n_radial, self.n_lon, self.n_lat)
        points_np = np.stack([rad_flat, lat_flat, lon_flat], axis=-1)

        indices_np = points_np.copy()
        return points_np, indices_np

    def _cartesian_coordinates_np(self, points_np: np.ndarray) -> np.ndarray:
        """Compute Cartesian coordinates from spherical (lat, lon) using NumPy only."""

        points_np = points_np.reshape((-1,3))

        rad = points_np[:, 0]
        lon = points_np[:, 1]
        lat = points_np[:, 2]

        x, y, z = spherical2cartesian(rad, lon, lat)
        return np.stack([x, y, z], axis=1).astype(self.r_dtype, copy=False)

    def _spherical_coordinates_np(self, points_np: np.ndarray) -> np.ndarray:
        """Return spherical coordinates (r, theta, phi) using NumPy only."""

        points_np = points_np.reshape((-1,3))

        x = points_np[:, 0]
        y = points_np[:, 1]
        z = points_np[:, 2]

        rad, lon, lat = cartesian2spherical(x, y, z)
        return np.stack([rad, lon, lat], axis=1).astype(self.r_dtype, copy=False)
        
    def _generate_mesh(self) -> Tuple:
        """Generate mesh in NumPy and convert to the selected backend; kept for API compatibility."""
        points_np, indices_np = self._generate_mesh_np()
        return self._to_backend(points_np), self._to_backend(indices_np)
    
    def _cartesian_coordinates(self):
        """Compute Cartesian coordinates via NumPy, then convert; kept for API compatibility."""
        points_np, _ = self._generate_mesh_np()

        shape = points_np.shape

        return self._to_backend(self._cartesian_coordinates_np(points_np)).reshape(shape)
    
    def _spherical_coordinates(self):
        """Compute spherical coordinates via NumPy, then convert; kept for API compatibility."""
        points_np, _ = self._generate_mesh_np()

        shape = points_np.shape

        return self._to_backend(self._spherical_coordinates_np(points_np)).reshape(shape)
    
    def get_mesh_shape(self) -> Tuple[int, int]:
        """Get the shape of the mesh (n_lat, n_lon)."""
        return (self.n_radial, self.n_lat, self.n_lon)
    
    def is_valid_point(self, points, coord_system: str = 'spherical'):
        """
        Check if points are valid (not in excluded regions).
        
        Args:
            points: Points to check, shape (N, 3)
            coord_system: Coordinate system of input points
            
        Returns:
            Boolean tensor indicating valid points
        """

        if isinstance(points, torch.Tensor):
            points = points.detach().cpu().numpy()

        if coord_system == 'cartesian':
            spherical = cartesian2spherical(points)
        else:
            spherical = points

        r, theta, _ = spherical[:, 0], spherical[:, 1], spherical[:, 2]

        if self.library == "torch":
            center_valid = r > self.center_exclusion_radius
            if self.exclude_poles:
                pole_valid = (theta > self.pole_exclusion_angle) & (theta < math.pi - self.pole_exclusion_angle)
            else:
                pole_valid = torch.ones_like(center_valid, dtype=torch.bool)
            return center_valid & pole_valid
        elif self.library == "jax":
            center_valid = r > self.center_exclusion_radius
            if self.exclude_poles:
                pole_valid = (theta > self.pole_exclusion_angle) & (theta < math.pi - self.pole_exclusion_angle)
            else:
                pole_valid = jnp.ones_like(center_valid, dtype=bool)
            return center_valid & pole_valid
        else:
            center_valid = r > self.center_exclusion_radius
            if self.exclude_poles:
                pole_valid = (theta > self.pole_exclusion_angle) & (theta < math.pi - self.pole_exclusion_angle)
            else:
                pole_valid = np.ones_like(center_valid, dtype=bool)
            return center_valid & pole_valid

    def plot_points(
        self,
        coord_system: str = "spherical",
        sample: Optional[int] = None,
        point_size: Union[int, float] = 0.1,
        color: Union[str, Tuple[float, float, float]] = "blue",
        alpha: float = 0.8,
        title: Optional[str] = None,
        save_path: Optional[str] = "meshgrid.html"
    ):
        """
        Plot `self.points` in 3D space using Plotly.

        Args:
            coord_system: Either "spherical" (r, lat, lon) or "cartesian" for the data in `self.points`.
            sample: If provided, randomly subsample this many points for plotting.
            point_size: Marker size for the scatter.
            color: Color for points (name like "royalblue" or RGB/RGBA tuple).
            alpha: Point transparency.
            title: Optional plot title.
            save_path: If provided, save to HTML (if path ends with .html) or to an image
                format supported by Plotly's Kaleido (e.g., .png, .jpg). Requires `kaleido` for images.

        Returns:
            The Plotly Figure object.
        """

        # Lazy import to avoid hard dependency unless this method is used
        try:
            import plotly.graph_objects as go
        except Exception as exc:
            raise RuntimeError("Plotly is required for plotting. Please install it: pip install plotly") from exc

        # Convert `self.points` to NumPy
        if self.library == "torch" and isinstance(self.points, torch.Tensor):
            points_np = self.points.detach().cpu().numpy()
        elif self.library == "jax":
            points_np = np.array(self.points)
        else:
            points_np = np.asarray(self.points)

        points_np = points_np.reshape((-1,3))
        num_points = points_np.shape[0]

        # Optional subsampling
        if sample is not None and 0 < sample < num_points:
            rng = np.random.default_rng()
            idx = rng.choice(num_points, size=sample, replace=False)
            points_np = points_np[idx]

        # Prepare Cartesian coordinates
        if coord_system.lower() == "spherical":
            # Our `self.points` are stored as [r, lat, lon]
            r = points_np[:, 0]
            lat = points_np[:, 1]
            lon = points_np[:, 2]
            x, y, z = spherical2cartesian(r, lon, lat)
        elif coord_system.lower() == "cartesian":
            x, y, z = points_np[:, 0], points_np[:, 1], points_np[:, 2]
        else:
            raise ValueError("coord_system must be either 'spherical' or 'cartesian'")

        # Convert color if tuple provided
        marker_color = color
        if isinstance(color, tuple):
            if len(color) == 3:
                r, g, b = color
                if 0.0 <= r <= 1.0 and 0.0 <= g <= 1.0 and 0.0 <= b <= 1.0:
                    r, g, b = int(r * 255), int(g * 255), int(b * 255)
                marker_color = f"rgb({r},{g},{b})"
            elif len(color) == 4:
                r, g, b, a = color
                if 0.0 <= r <= 1.0 and 0.0 <= g <= 1.0 and 0.0 <= b <= 1.0:
                    r, g, b = int(r * 255), int(g * 255), int(b * 255)
                marker_color = f"rgba({r},{g},{b},{a})"

        scatter = go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="markers",
            marker=dict(size=point_size, color=marker_color, opacity=alpha)
        )

        fig = go.Figure(data=[scatter], title = title)

        if save_path:
            if save_path.lower().endswith(".html"):
                fig.write_html(save_path, include_plotlyjs="cdn")
            else:
                try:
                    fig.write_image(save_path)
                except Exception as exc:
                    raise RuntimeError(
                        "Saving static images requires the 'kaleido' package. Install it with: pip install -U kaleido"
                    ) from exc

        return fig