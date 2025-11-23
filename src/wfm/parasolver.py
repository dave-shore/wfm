import torch
import torch.nn as nn
import numpy as np
from scipy import ndimage
import jax
import jax.numpy as jnp
from tqdm import trange
from itertools import chain
import gc
from typing import Tuple, Dict, Any, Optional, Callable, List, Union
from .spherical_mesh import SphericalMesh
from .base import *


def fit_tensor_shape(tensor: torch.Tensor, desired_shape: Tuple[int, ...]) -> torch.Tensor:
    """
    Reshape a tensor to a desired shape by tiling (if smaller) or slicing (if larger).
    
    Args:
        tensor: Input tensor to reshape
        desired_shape: Target shape tuple
        
    Returns:
        Tensor with the desired shape, obtained by tiling or slicing the input tensor
    """
    if tensor.shape == desired_shape:
        return tensor
    
    result = tensor.clone()
    
    for dim, desired_size in enumerate(desired_shape):
        current_size = result.shape[dim]
        
        if current_size < desired_size:
            # Need to tile/repeat to expand
            # Calculate how many times we need to repeat
            repeat_factor = (desired_size + current_size - 1) // current_size  # Ceiling division
            result = torch.repeat_interleave(result, repeats=repeat_factor, dim=dim)
            # Slice to exact desired size
            result = result.narrow(dim, 0, desired_size)
            
        elif current_size > desired_size:
            # Need to downsample by taking evenly spaced elements
            # Use linspace to get evenly distributed indices
            indices = torch.linspace(0, current_size - 1, desired_size, device=result.device).long()
            result = torch.index_select(result, dim, indices)
    
    return result


def get_jacobian_on_field(func: torch.Tensor | jnp.ndarray | np.ndarray | Callable, field: torch.Tensor | jnp.ndarray | np.ndarray, interpolation_factor: int = 4, **kwargs):

    field = field.squeeze()

    if not isinstance(func, Callable):
        field_out_dim = field.shape[-1]
        return func.reshape(1,1,1, field_out_dim, field_out_dim)

    field_shape = field.shape[:-1]
    field_out_dim = field.shape[-1]
    start_idx = (
        (field_shape[0] % interpolation_factor) // 2, 
        (field_shape[1] % interpolation_factor) // 2, 
        (field_shape[2] % interpolation_factor) // 2
    )
    field = field[start_idx[0]::interpolation_factor, start_idx[1]::interpolation_factor, start_idx[2]::interpolation_factor, :]
    field_reduced_shape = field.shape[:-1]
    
    if hasattr(func, "library"):
        field = field.reshape(-1, field_out_dim)
        if func.library == "torch":
            J = torch.cat([
                torch.func.vmap(
                    torch.func.jacfwd(func)
                )(field[i:i+BASE_BATCH_SIZE], **kwargs)
                for i in range(0, field.shape[0], BASE_BATCH_SIZE)
            ], dim = 0).reshape(*field_reduced_shape, field_out_dim, field_out_dim)
            return torch.nn.functional.interpolate(J.permute(4,3,2,1,0), size = field_shape[::-1], mode = "trilinear").permute(4,3,2,1,0)

        elif func.library == "jax":
            J = jnp.concatenate([
                jax.vmap(
                    jax.jacfwd(func)
                )(field[i:i+BASE_BATCH_SIZE], **kwargs)
                for i in range(0, field.shape[0], BASE_BATCH_SIZE)
            ], axis = 0).reshape(*field_reduced_shape, field_out_dim, field_out_dim)
            return jax.nn.interpolate(J.transpose(4,3,2,1,0), size = field_shape[::-1], mode = "trilinear").transpose(4,3,2,1,0)

        else:
            J = np.concatenate([
                np.gradient(func(field[i:i+BASE_BATCH_SIZE], **kwargs), axis = -1)
                for i in range(0, field.shape[0], BASE_BATCH_SIZE)
            ], axis = 0).reshape(*field_reduced_shape, field_out_dim, field_out_dim)
            return ndimage.interpolation.zoom(J.transpose(4,3,2,1,0), size = field_shape[::-1], mode = "trilinear").transpose(4,3,2,1,0)

    else:
        J = np.gradient(func(np.asarray(field), **kwargs), axis = -1)
        return J

class RandNetTorch(nn.Module):
    """
    Shallow random neural network used to estimate the delta between a coarse integrator (e.g. Runge-Kutta 1) and the actual solution of a differential equation.
    """

    def __init__(self, field_out_size: int, time_steps: int, mesh_size: Tuple[int, int, int], mesh_sampling_rate: int  | Tuple[int, int, int] = 1, device: torch.device = torch.device("cpu"), max_U_norm: float = 1.0, regularization_lambda: float = 0.1, optimizer_kwargs: Dict[str, Any] = {"lr": 1e-3, "weight_decay": 1e-4}):

        super().__init__()

        self.field_out_size = field_out_size
        self.time_steps = time_steps
        self.mesh_size = mesh_size
        self.mesh_sampling_rate = mesh_sampling_rate if isinstance(mesh_sampling_rate, tuple) else (mesh_sampling_rate, mesh_sampling_rate, mesh_sampling_rate)
        assert len(self.mesh_sampling_rate) == len(mesh_size), f"Mesh sampling rate must be a tuple of {len(mesh_size)} integers, got {mesh_sampling_rate}"
        self.device = device
        self.max_U_norm = max_U_norm
        self.rho = 1.0
        self.max_Z_abs = max(self.max_U_norm*self.rho, 1.0)
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(self.net.parameters(), **optimizer_kwargs)
        self.regularization_lambda = regularization_lambda

        self.input_size = (field_out_size, time_steps, mesh_size[0] // self.mesh_sampling_rate[0], mesh_size[1] // self.mesh_sampling_rate[1], mesh_size[2] // self.mesh_sampling_rate[2])
        # shape = (3, 6L, 100, 360, 180)
        self.input_size_flat = int(np.prod(self.input_size))

        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.input_size_flat, self.input_size_flat)
        )

        self.A = torch.randn(self.input_size_flat, self.input_size_flat, device = self.device)
        self.A = self.A / torch.norm(self.A, dim = 1, keepdim = True)
        self.Z = torch.distributions.Uniform(-self.max_Z_abs, self.max_Z_abs).sample((self.input_size_flat,1)).to(self.device)

    def forward(self, U: torch.Tensor):

        if U.dim() < 5:
            U = U.unsqueeze(0)

        assert U.shape[1:] == self.input_size, f"Input tensor must have shape (*,{self.input_size}), got {U.shape}"
        U_flat = U.flatten(start_dim = 1)
        U_hat = self.net(self.A @ U_flat + self.Z)

        return U_hat

    def _update_parameters(self, U: torch.Tensor, U_hat: Optional[torch.Tensor] = None):

        if U_hat is None:
            U_hat = self.forward(U)

        self.optimizer.zero_grad(set_to_none = True)
        loss = self.loss_fn(U_hat, U) + self.regularization_lambda * self.net[1].weight.norm()
        loss.backward()

        self.optimizer.step()

        return loss.item()


class SphericalCrankNicolson():

    def __init__(self, dt: float, mesh: SphericalMesh, mesh_sampling_rate: int | Tuple[int, int, int] = 1):

        self.dt = dt
        self.mesh = mesh
        self.mesh_sampling_rate = mesh_sampling_rate if isinstance(mesh_sampling_rate, tuple) else (mesh_sampling_rate, mesh_sampling_rate, mesh_sampling_rate)

        self.actual_mesh_size = tuple(self.mesh.shape[i] // self.mesh_sampling_rate[i] for i in range(len(self.mesh.shape)))
        x = self.mesh.points[::self.mesh_sampling_rate[0], ::self.mesh_sampling_rate[1], ::self.mesh_sampling_rate[2], :]

        if mesh.library == "torch":
            self.local_dx = torch.stack([torch.diff(x[...,i], dim = i, prepend = torch.zeros_like(x.select(i, 0)).unsqueeze(i)[...,i]) for i in range(x.shape[-1])], dim = -1)
            self.sin = torch.sin
            self.cos = torch.cos
            self.roll = torch.roll
            self.concat = torch.cat
            self.solve = torch.linalg.lstsq
            self.full = torch.full

        elif mesh.library == "jax":
            self.local_dx = jnp.stack([jnp.diff(x[...,i], axis = i, prepend = jnp.zeros_like(x.select(i, 0)).expand_dims(i)[...,i]) for i in range(x.shape[-1])], axis = -1)
            self.sin = jnp.sin
            self.cos = jnp.cos
            self.roll = jnp.roll
            self.concat = jnp.cat
            self.solve = jnp.linalg.lstsq
            self.full = jnp.full

        else:
            self.local_dx = np.stack([np.diff(x[...,i], axis = i, prepend = np.zeros_like(x.select(i, 0)).expand_dims(i)[...,i]) for i in range(x.shape[-1])], axis = -1)
            self.sin = np.sin
            self.cos = np.cos
            self.roll = np.roll
            self.concat = np.concatenate
            self.solve = np.linalg.lstsq
            self.full = np.full

    def __call__(
        self, 
        U_initial: torch.Tensor | jnp.ndarray | np.ndarray, 
        time_steps: int, 
        velocity_term: float | torch.Tensor | jnp.ndarray | np.ndarray, 
        diffusion_term: float | torch.Tensor | jnp.ndarray | np.ndarray | Callable, 
        advection_term: float | torch.Tensor | jnp.ndarray | np.ndarray | Callable, 
        reaction_term: float |torch.Tensor | jnp.ndarray | np.ndarray | Callable,
        dirichlet_bc: List[float | torch.Tensor | jnp.ndarray | np.ndarray] | Callable
    ):
        if self.local_dx.shape[:2] != U_initial.shape[:2]:
            U_initial = U_initial[::self.mesh_sampling_rate[0], ::self.mesh_sampling_rate[1], ::self.mesh_sampling_rate[2], :]
            assert self.local_dx.shape[:2] == U_initial.shape[:2], f"Local spatial step must have shape {U_initial.shape[:2]}, got {self.local_dx.shape[:2]}"

        size_1, size_2, size_3, field_dim = U_initial.shape
        # In case of spherical coordinates, dimension 1 will be in ]0;1], dimension 2 in [0;2*pi], dimension 3 in ]0;pi[

        if isinstance(dirichlet_bc, Callable):
            boundary_conditions = [tuple(dirichlet_bc(bp) for bp in tup) for tup in self.mesh.get_boundary_points()]
        else:
            boundary_conditions = dirichlet_bc

        output = torch.empty((time_steps+1,size_1-2, size_2-2, size_3-2, field_dim), device = self.mesh.device)

        # Exclude boundary points
        U_curr = U_initial[1:-1,1:-1,1:-1,:]
        # shape = (x-2, y-2, z-2, 3)
        output[0] = U_curr
        U_prev = torch.zeros_like(U_curr) if isinstance(U_initial, torch.Tensor) else jnp.zeros_like(U_curr) if isinstance(U_initial, jnp.ndarray) else np.zeros_like(U_curr)
        # shape = (x-2, y-2, z-2, 3)

        k = self.dt
        X = self.mesh.points[::self.mesh_sampling_rate[0],::self.mesh_sampling_rate[1],::self.mesh_sampling_rate[2]][1:-1,1:-1,1:-1]
        X = X.reshape(*X.shape, 1)
        H = self.local_dx[1:-1,1:-1,1:-1]
        H = H.reshape(*H.shape, 1)
        assert X.shape == H.shape, f"X and H must have the same shape, got {X.shape} and {H.shape}"
        inv_h0 = H[..., [0], :]**(-1)
        inv_h1 = H[..., [1], :]**(-1)
        inv_h2 = H[..., [2], :]**(-1)
        inv_h0[inv_h0 > UB] = UB
        inv_h1[inv_h1 > UB] = UB
        inv_h2[inv_h2 > UB] = UB

        laplacian_coefficients = (
            1,
            X[..., [0],:]**(-2),
            (X[..., [0],:] * (self.sin(X[..., [2],:]) + EPS))**(-2),    
        )
        laplacian_gradient_coefficients = (
            2 * X[..., [0], :]**(-1),
            self.cos(X[..., [1], :]) * (self.sin(X[..., [1], :]) + EPS)**(-1),
            0
        )
        gradient_coefficients = (
            1,
            X[..., [0], :]**(-1),
            X[..., [0], :]**(-1) * (self.sin(X[..., [1], :]) + EPS)**(-1),
        )

        tensor_boundary_conditions = []
        for i,bc in enumerate(boundary_conditions):
            shape_to_fit = [time_steps, size_1-2, size_2-2, size_3-2, field_dim]
            shape_to_fit[i+1] = 1
            tensor_boundary_conditions.append([])
            for j,b in enumerate(bc):
                if isinstance(b, torch.Tensor | jnp.ndarray | np.ndarray):
                    B = fit_tensor_shape(b, tuple(shape_to_fit))

                elif isinstance(b, float):
                    B = self.full(tuple(shape_to_fit), b, device = U_curr.device)[:time_steps]

                elif b is None:
                    B = U_curr.select(i, -j)
                    B = B.reshape(1, *shape_to_fit[1:]).repeat(shape_to_fit[0], 1, 1, 1, 1)

                elif isinstance(b, Callable):
                    B = b(self.mesh.points.select(i, 0 + j*self.mesh.shape[i])).reshape(tuple(shape_to_fit))[:time_steps]
                else:
                    B = b

                if hasattr(B, "reshape"):
                    B = B.reshape(*B.shape, 1)

                # shape = (time_steps, size_1-2, size_2-2, size_3-2, 3, 1) where one of the sizes becomes 1

                tensor_boundary_conditions[i].append(B)
        
        gc.collect()

        for n in trange(time_steps):
            V = get_jacobian_on_field(velocity_term, U_curr)
            D = get_jacobian_on_field(diffusion_term, U_curr)
            A = get_jacobian_on_field(advection_term, U_curr)
            R = get_jacobian_on_field(reaction_term, U_curr)

            next_coefficient = 1 / k**2 + V / k 
            curr_centered_coefficient = torch.zeros_like(U_curr) if isinstance(U_curr, torch.Tensor) else jnp.zeros_like(U_curr) if isinstance(U_curr, jnp.ndarray) else np.zeros_like(U_curr)
            if curr_centered_coefficient.dim() < 5:
                curr_centered_coefficient = curr_centered_coefficient.reshape(*curr_centered_coefficient.shape, 1)
            elif curr_centered_coefficient.dim() > 5:
                curr_centered_coefficient = curr_centered_coefficient.squeeze()

            curr_centered_coefficient = curr_centered_coefficient.repeat(1, 1, 1, 1, field_dim)
            curr_centered_coefficient += 2 / (k**2)
            curr_centered_coefficient += V / k
            curr_centered_coefficient -= inv_h0*laplacian_gradient_coefficients[0]*D
            curr_centered_coefficient -= 2*inv_h0**2*laplacian_coefficients[0]*D
            curr_centered_coefficient -= inv_h1*laplacian_gradient_coefficients[1]*D
            curr_centered_coefficient -= 2*inv_h1**2*laplacian_coefficients[1]*D
            curr_centered_coefficient -= inv_h2*laplacian_gradient_coefficients[2]*D
            curr_centered_coefficient -= 2*inv_h2**2*laplacian_coefficients[2]*D
            curr_centered_coefficient -= inv_h0*gradient_coefficients[0]*A
            curr_centered_coefficient -= inv_h1*gradient_coefficients[1]*A
            curr_centered_coefficient -= inv_h2*gradient_coefficients[2]*A
            curr_centered_coefficient += R

            curr_int_coefficient = inv_h0**2*laplacian_coefficients[0]*D
            curr_ext_coefficient = inv_h0**2*laplacian_coefficients[0]*D
            curr_ext_coefficient += inv_h0*laplacian_gradient_coefficients[0]*D
            curr_ext_coefficient += inv_h0*gradient_coefficients[0]*A
            curr_right_coefficient = inv_h1**2*laplacian_coefficients[1]*D
            curr_right_coefficient += inv_h1*laplacian_gradient_coefficients[1]*D
            curr_right_coefficient += inv_h1*gradient_coefficients[1]*A
            curr_left_coefficient = inv_h1**2*laplacian_coefficients[1]*D
            curr_up_coefficient = inv_h2**2*laplacian_coefficients[2]*D
            curr_up_coefficient += inv_h2*laplacian_gradient_coefficients[2]*D
            curr_up_coefficient += inv_h2*gradient_coefficients[2]*A
            curr_down_coefficient = inv_h2**2*laplacian_coefficients[2]*D

            if U_curr.dim() < 5:
                U_curr = U_curr.reshape(*U_curr.shape, 1)
            elif U_curr.dim() > 5:
                U_curr = U_curr.squeeze()
            
            if U_prev.dim() < 5:
                U_prev = U_prev.reshape(*U_curr.shape)
            elif U_prev.dim() > 5:
                U_prev = U_prev.squeeze()

            int_term = self.concat([
                tensor_boundary_conditions[0][0][n],
                curr_int_coefficient[:-1,:,:] @ U_curr[:-1,:,:]
            ])
            ext_term = self.concat([
                curr_ext_coefficient[1:,:,:] @ U_curr[1:,:,:],
                tensor_boundary_conditions[0][1][n]
            ])
            right_term = self.roll(curr_right_coefficient @ U_curr, shifts = -1, dims = 1)
            left_term = self.roll(curr_left_coefficient @ U_curr, shifts = 1, dims = 1)
            up_term = self.concat([
                curr_up_coefficient[:,:,:-1] @ U_curr[:,:,:-1],
                tensor_boundary_conditions[2][1][n]
            ], axis = 2)
            down_term = self.concat([
                tensor_boundary_conditions[2][0][n],
                curr_down_coefficient[:,:,1:] @ U_curr[:,:,1:]
            ], axis = 2)
            prev_coefficient = -k**(-2)

            explicit_term = curr_centered_coefficient*U_curr + prev_coefficient*U_prev + int_term + ext_term + right_term + left_term + up_term + down_term
            # shape = (size_1-2, size_2-2, size_3-2, 3, 3)
            U_next = torch.sum(next_coefficient.transpose(-1,-2) * explicit_term, dim = -1)
            output[n+1] = U_next
            U_prev = U_curr
            U_curr = U_next

        for i, bc in enumerate(tensor_boundary_conditions):
            # Apply zero-padding on both sides of dimensions 0 through i (inclusive)
            # Padding format for torch.nn.functional.pad goes from last dimension backwards:
            # (pad_left_dim_n, pad_right_dim_n, pad_left_dim_n-1, pad_right_dim_n-1, ...)
            num_dims = len(output.shape)
            pad_tuple = sum([[1, int(dim > 0)] if dim <= i else [0, 0] for dim in range(num_dims)], [])[::-1]
            
            padded_bc0 = torch.nn.functional.pad(bc[0].squeeze(-1), pad_tuple, mode='constant', value=0)
            padded_bc1 = torch.nn.functional.pad(bc[1].squeeze(-1), pad_tuple, mode='constant', value=0)
            
            output = torch.cat([padded_bc0, output, padded_bc1], dim = i+1)

        output = torch.nn.functional.interpolate(
            output.transpose(-1,1), 
            size = self.mesh.shape[-1::-1], 
            mode = "trilinear"
        ).transpose(-1,1)

        if self.mesh.library == "torch":
            return output
        elif self.mesh.library == "jax":
            return jnp.asarray(output)
        else:
            return output.detach().cpu().numpy()


class Parareal:

    def __init__(self, coarse_integrator: Callable, delta_estimator: Callable):
        """
        Initialize Parareal algorithm.
        
        Args:
            coarse_integrator: Coarse integrator function
            delta_estimator: Delta estimator function
        """
        self.coarse_integrator = coarse_integrator
        self.delta_estimator = delta_estimator