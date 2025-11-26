from turtle import ycor
import torch
import torch.nn as nn
import numpy as np
from scipy import ndimage
import jax
import jax.numpy as jnp
from tqdm import trange
import gc
from typing import Tuple, Dict, Any, Optional, Callable, List, Union
from .spherical_mesh import *
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
            nn.Dropout(0.1),
            nn.Linear(self.input_size_flat, self.input_size_flat)
        )

        self.residual = nn.Sequential(
            nn.Linear(self.input_size_flat, self.input_size_flat),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.input_size_flat, self.input_size_flat),
            nn.BatchNorm1d(self.input_size_flat)
        )

        self.A = torch.randn(self.input_size_flat, self.input_size_flat, device = self.device)
        self.A = self.A / torch.norm(self.A, dim = 1, keepdim = True)
        self.Z = torch.distributions.Uniform(-self.max_Z_abs, self.max_Z_abs).sample((self.input_size_flat,1)).to(self.device)

    def forward(self, U: torch.Tensor):

        if U.dim() < 5:
            U = U.unsqueeze(0)

        initial_shape = U.shape

        assert U.shape[1:] == self.input_size, f"Input tensor must have shape (*,{self.input_size}), got {U.shape}"
        U_flat = U.flatten(start_dim = 1)
        U_hat = self.net(U_flat @ self.A + self.Z)
        U_tilde = self.residual(U_flat)

        Y = torch.nn.functional.glu(torch.cat([U_tilde, U_hat], dim = -1)) 

        return Y.reshape(initial_shape)

    def _update_parameters(self, U: torch.Tensor, U_hat: Optional[torch.Tensor] = None):

        if U_hat is None:
            U_hat = self.forward(U)

        self.optimizer.zero_grad(set_to_none = True)
        loss = self.loss_fn(U_hat, U) + self.regularization_lambda * self.net[1].weight.norm()
        loss.backward()

        self.optimizer.step()

        return loss.item()


class BaseIntegrator():

    def __init__(self, dt: float, mesh: GenericMesh, mesh_sampling_rate: int | Tuple[int, int, int] = 1):

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


class SphericalCrankNicolson(BaseIntegrator):

    def __init__(self, dt: float, mesh: SphericalMesh, mesh_sampling_rate: int | Tuple[int, int, int] = 1):

        super().__init__(dt, mesh, mesh_sampling_rate)

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
        inv_h0 = (H[..., [0], :] + EPS)**(-1)
        inv_h1 = (H[..., [1], :] + EPS)**(-1)
        inv_h2 = (H[..., [2], :] + EPS)**(-1)
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



class SphericalRungeKutta8(BaseIntegrator):

    def __init__(self, dt: float, mesh: SphericalMesh, mesh_sampling_rate: int | Tuple[int, int, int] = 1):

        super().__init__(dt, mesh, mesh_sampling_rate)
        
        # RK8 coefficients (8-stage method)
        # Using a standard 8-stage Runge-Kutta method
        self.a = [
            [0],
            [1/4],
            [3/32, 9/32],
            [1932/2197, -7200/2197, 7296/2197],
            [439/216, -8, 3680/513, -845/4104],
            [-8/27, 2, -3544/2565, 1859/4104, -11/40],
            [16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55],
            [25/216, 0, 1408/2565, 2197/4104, -1/5, 0]
        ]
        self.b = [25/216, 0, 1408/2565, 2197/4104, -1/5, 0, 0, 0]  # 8th order weights
        self.c = [0, 1/4, 3/8, 12/13, 1, 1/2, 1, 1]  # Time nodes

    def _compute_rhs(
        self,
        U: torch.Tensor | jnp.ndarray | np.ndarray,
        velocity_term: float | torch.Tensor | jnp.ndarray | np.ndarray,
        diffusion_term: float | torch.Tensor | jnp.ndarray | np.ndarray | Callable,
        advection_term: float | torch.Tensor | jnp.ndarray | np.ndarray | Callable,
        reaction_term: float | torch.Tensor | jnp.ndarray | np.ndarray | Callable,
        X: torch.Tensor | jnp.ndarray | np.ndarray,
        H: torch.Tensor | jnp.ndarray | np.ndarray,
        inv_h0: torch.Tensor | jnp.ndarray | np.ndarray,
        inv_h1: torch.Tensor | jnp.ndarray | np.ndarray,
        inv_h2: torch.Tensor | jnp.ndarray | np.ndarray,
        laplacian_coefficients: Tuple,
        laplacian_gradient_coefficients: Tuple,
        gradient_coefficients: Tuple,
        tensor_boundary_conditions: List,
        n: int
    ):
        """Compute the right-hand side of the PDE: du/dt = f(t, u)"""
        
        V = get_jacobian_on_field(velocity_term, U)
        D = get_jacobian_on_field(diffusion_term, U)
        A = get_jacobian_on_field(advection_term, U)
        R = get_jacobian_on_field(reaction_term, U)
        
        size_1, size_2, size_3, field_dim = U.shape
        
        # Reshape U for matrix operations
        if U.dim() < 5:
            U_reshaped = U.reshape(*U.shape, 1)
        else:
            U_reshaped = U
        
        # Compute centered coefficient
        centered_coefficient = torch.zeros_like(U) if isinstance(U, torch.Tensor) else jnp.zeros_like(U) if isinstance(U, jnp.ndarray) else np.zeros_like(U)
        if centered_coefficient.dim() < 5:
            centered_coefficient = centered_coefficient.reshape(*centered_coefficient.shape, 1)
        centered_coefficient = centered_coefficient.repeat(1, 1, 1, 1, field_dim)
        
        centered_coefficient += V
        centered_coefficient -= inv_h0*laplacian_gradient_coefficients[0]*D
        centered_coefficient -= 2*inv_h0**2*laplacian_coefficients[0]*D
        centered_coefficient -= inv_h1*laplacian_gradient_coefficients[1]*D
        centered_coefficient -= 2*inv_h1**2*laplacian_coefficients[1]*D
        centered_coefficient -= inv_h2*laplacian_gradient_coefficients[2]*D
        centered_coefficient -= 2*inv_h2**2*laplacian_coefficients[2]*D
        centered_coefficient -= inv_h0*gradient_coefficients[0]*A
        centered_coefficient -= inv_h1*gradient_coefficients[1]*A
        centered_coefficient -= inv_h2*gradient_coefficients[2]*A
        centered_coefficient += R
        
        # Compute neighbor coefficients
        int_coefficient = inv_h0**2*laplacian_coefficients[0]*D
        ext_coefficient = inv_h0**2*laplacian_coefficients[0]*D
        ext_coefficient += inv_h0*laplacian_gradient_coefficients[0]*D
        ext_coefficient += inv_h0*gradient_coefficients[0]*A
        right_coefficient = inv_h1**2*laplacian_coefficients[1]*D
        right_coefficient += inv_h1*laplacian_gradient_coefficients[1]*D
        right_coefficient += inv_h1*gradient_coefficients[1]*A
        left_coefficient = inv_h1**2*laplacian_coefficients[1]*D
        up_coefficient = inv_h2**2*laplacian_coefficients[2]*D
        up_coefficient += inv_h2*laplacian_gradient_coefficients[2]*D
        up_coefficient += inv_h2*gradient_coefficients[2]*A
        down_coefficient = inv_h2**2*laplacian_coefficients[2]*D
        
        # Compute neighbor terms
        int_term = self.concat([
            tensor_boundary_conditions[0][0][n],
            int_coefficient[:-1,:,:] @ U_reshaped[:-1,:,:]
        ])
        ext_term = self.concat([
            ext_coefficient[1:,:,:] @ U_reshaped[1:,:,:],
            tensor_boundary_conditions[0][1][n]
        ])
        right_term = self.roll(right_coefficient @ U_reshaped, shifts=-1, dims=1)
        left_term = self.roll(left_coefficient @ U_reshaped, shifts=1, dims=1)
        up_term = self.concat([
            up_coefficient[:,:,:-1] @ U_reshaped[:,:,:-1],
            tensor_boundary_conditions[2][1][n]
        ], axis=2)
        down_term = self.concat([
            tensor_boundary_conditions[2][0][n],
            down_coefficient[:,:,1:] @ U_reshaped[:,:,1:]
        ], axis=2)
        
        # Compute RHS: du/dt = centered_term + neighbor_terms
        rhs = centered_coefficient @ U_reshaped + int_term + ext_term + right_term + left_term + up_term + down_term
        
        # Sum over the last dimension to get the final RHS
        if isinstance(rhs, torch.Tensor):
            rhs = torch.sum(rhs, dim=-1)
        elif isinstance(rhs, jnp.ndarray):
            rhs = jnp.sum(rhs, axis=-1)
        else:
            rhs = np.sum(rhs, axis=-1)
        
        return rhs

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

        output = torch.empty((time_steps+1, size_1-2, size_2-2, size_3-2, field_dim), device=self.mesh.device)

        # Exclude boundary points
        U_curr = U_initial[1:-1,1:-1,1:-1,:]
        # shape = (x-2, y-2, z-2, field_dim)
        output[0] = U_curr

        k = self.dt
        X = self.mesh.points[::self.mesh_sampling_rate[0],::self.mesh_sampling_rate[1],::self.mesh_sampling_rate[2]][1:-1,1:-1,1:-1]
        X = X.reshape(*X.shape, 1)
        H = self.local_dx[1:-1,1:-1,1:-1]
        H = H.reshape(*H.shape, 1)
        assert X.shape == H.shape, f"X and H must have the same shape, got {X.shape} and {H.shape}"
        inv_h0 = (H[..., [0], :] + EPS)**(-1)
        inv_h1 = (H[..., [1], :] + EPS)**(-1)
        inv_h2 = (H[..., [2], :] + EPS)**(-1)
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
                    B = self.full(tuple(shape_to_fit), b, device=U_curr.device if hasattr(U_curr, 'device') else None)[:time_steps]

                elif b is None:
                    B = U_curr.select(i, -j)
                    B = B.reshape(1, *shape_to_fit[1:]).repeat(shape_to_fit[0], 1, 1, 1, 1)

                elif isinstance(b, Callable):
                    B = b(self.mesh.points.select(i, 0 + j*self.mesh.shape[i])).reshape(tuple(shape_to_fit))[:time_steps]
                else:
                    B = b

                if hasattr(B, "reshape"):
                    B = B.reshape(*B.shape, 1)

                tensor_boundary_conditions[i].append(B)
        
        gc.collect()

        # RK8 integration loop
        for n in trange(time_steps):
            # Compute k1
            k1 = self._compute_rhs(
                U_curr, velocity_term, diffusion_term, advection_term, reaction_term,
                X, H, inv_h0, inv_h1, inv_h2,
                laplacian_coefficients, laplacian_gradient_coefficients, gradient_coefficients,
                tensor_boundary_conditions, n
            )
            
            # Compute k2
            U2 = U_curr + k * self.c[1] * k1
            k2 = self._compute_rhs(
                U2, velocity_term, diffusion_term, advection_term, reaction_term,
                X, H, inv_h0, inv_h1, inv_h2,
                laplacian_coefficients, laplacian_gradient_coefficients, gradient_coefficients,
                tensor_boundary_conditions, n
            )
            
            # Compute k3
            U3 = U_curr + k * (self.a[2][0] * k1 + self.a[2][1] * k2)
            k3 = self._compute_rhs(
                U3, velocity_term, diffusion_term, advection_term, reaction_term,
                X, H, inv_h0, inv_h1, inv_h2,
                laplacian_coefficients, laplacian_gradient_coefficients, gradient_coefficients,
                tensor_boundary_conditions, n
            )
            
            # Compute k4
            U4 = U_curr + k * (self.a[3][0] * k1 + self.a[3][1] * k2 + self.a[3][2] * k3)
            k4 = self._compute_rhs(
                U4, velocity_term, diffusion_term, advection_term, reaction_term,
                X, H, inv_h0, inv_h1, inv_h2,
                laplacian_coefficients, laplacian_gradient_coefficients, gradient_coefficients,
                tensor_boundary_conditions, n
            )
            
            # Compute k5
            U5 = U_curr + k * (self.a[4][0] * k1 + self.a[4][1] * k2 + self.a[4][2] * k3 + self.a[4][3] * k4)
            k5 = self._compute_rhs(
                U5, velocity_term, diffusion_term, advection_term, reaction_term,
                X, H, inv_h0, inv_h1, inv_h2,
                laplacian_coefficients, laplacian_gradient_coefficients, gradient_coefficients,
                tensor_boundary_conditions, n
            )
            
            # Compute k6
            U6 = U_curr + k * (self.a[5][0] * k1 + self.a[5][1] * k2 + self.a[5][2] * k3 + self.a[5][3] * k4 + self.a[5][4] * k5)
            k6 = self._compute_rhs(
                U6, velocity_term, diffusion_term, advection_term, reaction_term,
                X, H, inv_h0, inv_h1, inv_h2,
                laplacian_coefficients, laplacian_gradient_coefficients, gradient_coefficients,
                tensor_boundary_conditions, n
            )
            
            # Compute k7
            U7 = U_curr + k * (self.a[6][0] * k1 + self.a[6][1] * k2 + self.a[6][2] * k3 + self.a[6][3] * k4 + self.a[6][4] * k5 + self.a[6][5] * k6)
            k7 = self._compute_rhs(
                U7, velocity_term, diffusion_term, advection_term, reaction_term,
                X, H, inv_h0, inv_h1, inv_h2,
                laplacian_coefficients, laplacian_gradient_coefficients, gradient_coefficients,
                tensor_boundary_conditions, n
            )
            
            # Compute k8
            U8 = U_curr + k * (self.a[7][0] * k1 + self.a[7][1] * k2 + self.a[7][2] * k3 + self.a[7][3] * k4 + self.a[7][4] * k5 + self.a[7][5] * k6 + self.a[7][6] * k7)
            k8 = self._compute_rhs(
                U8, velocity_term, diffusion_term, advection_term, reaction_term,
                X, H, inv_h0, inv_h1, inv_h2,
                laplacian_coefficients, laplacian_gradient_coefficients, gradient_coefficients,
                tensor_boundary_conditions, n
            )
            
            # Combine stages using RK8 weights
            U_next = U_curr + k * (
                self.b[0] * k1 + self.b[1] * k2 + self.b[2] * k3 + self.b[3] * k4 +
                self.b[4] * k5 + self.b[5] * k6 + self.b[6] * k7 + self.b[7] * k8
            )
            
            output[n+1] = U_next
            U_curr = U_next

        # Apply boundary conditions
        for i, bc in enumerate(tensor_boundary_conditions):
            # Apply zero-padding on both sides of dimensions 0 through i (inclusive)
            num_dims = len(output.shape)
            pad_tuple = sum([[1, int(dim > 0)] if dim <= i else [0, 0] for dim in range(num_dims)], [])[::-1]
            
            padded_bc0 = torch.nn.functional.pad(bc[0].squeeze(-1), pad_tuple, mode='constant', value=0)
            padded_bc1 = torch.nn.functional.pad(bc[1].squeeze(-1), pad_tuple, mode='constant', value=0)
            
            output = torch.cat([padded_bc0, output, padded_bc1], dim=i+1)

        output = torch.nn.functional.interpolate(
            output.transpose(-1,1), 
            size=self.mesh.shape[-1::-1], 
            mode="trilinear"
        ).transpose(-1,1)

        if self.mesh.library == "torch":
            return output
        elif self.mesh.library == "jax":
            return jnp.asarray(output)
        else:
            return output.detach().cpu().numpy()


class Parareal(BaseIntegrator):

    def __init__(self, coarse_integrator: Callable, delta_estimator: Callable, dt: float, mesh: GenericMesh, mesh_sampling_rate: int | Tuple[int, int, int] = 1):
        """
        Initialize Parareal algorithm.
        
        Args:
            coarse_integrator: Coarse integrator function
            delta_estimator: Delta estimator function
        """
        super().__init__(dt, mesh, mesh_sampling_rate)

        if isinstance(coarse_integrator, BaseIntegrator):
            self.coarse_integrator = coarse_integrator
        else:
            self.coarse_integrator = coarse_integrator(dt, mesh, mesh_sampling_rate)

        self.delta_estimator = delta_estimator

    def train_delta_estimator(self, U_initial: torch.Tensor | jnp.ndarray | np.ndarray, time_steps: int, velocity_term: float | torch.Tensor | jnp.ndarray | np.ndarray, diffusion_term: float | torch.Tensor | jnp.ndarray | np.ndarray | Callable, advection_term: float | torch.Tensor | jnp.ndarray | np.ndarray | Callable, reaction_term: float |torch.Tensor | jnp.ndarray | np.ndarray | Callable, dirichlet_bc: List[float | torch.Tensor | jnp.ndarray | np.ndarray] | Callable, fine_integrator: Callable, n_epochs: int = 100, optimizer_name: str = "AdamW", optimizer_kwargs: Dict[str, Any] = {"lr": 1e-3, "weight_decay": 1e-4}, patience: int = 10, tol: float = 1e-6):
        """
        Train the delta estimator.
        
        Args:
            U_initial: Initial condition
            time_steps: Number of time steps
            velocity_term: Velocity term
            diffusion_term: Diffusion term
            advection_term: Advection term
            reaction_term: Reaction term
            dirichlet_bc: Dirichlet boundary conditions
            fine_integrator: Fine integrator function
            n_epochs: Number of epochs
            optimizer_name: Name of the optimizer
            optimizer_kwargs: Keyword arguments for the optimizer
        """

        coarse_output = self.coarse_integrator(U_initial, time_steps, velocity_term, diffusion_term, advection_term, reaction_term, dirichlet_bc)
        fine_output = fine_integrator(U_initial, time_steps, velocity_term, diffusion_term, advection_term, reaction_term, dirichlet_bc)
        delta_gt = fine_output - coarse_output

        previous_delta = torch.randn_like(coarse_output)
        previous_delta[0] = torch.zeros_like(previous_delta[0])

        optimizer = getattr(torch.optim, optimizer_name)(self.delta_estimator.parameters(), **optimizer_kwargs)
        latest_losses = []
        noise = torch.zeros_like(previous_delta)

        self.delta_estimator.train()
        for epoch in trange(n_epochs, desc = "Training delta estimator"):
            optimizer.zero_grad()
            output_delta = self.delta_estimator(previous_delta + noise)
            loss = torch.norm(output_delta - delta_gt, p = 2, dim = -1).max().item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.delta_estimator.parameters(), max_norm = 1.0)
            optimizer.step()
            previous_delta = output_delta
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item()}")
            latest_losses.append(loss.item())
            if len(latest_losses) > patience:
                latest_losses.pop(0)
                if all(l < tol for l in latest_losses):
                    noise = torch.randn_like(previous_delta)
                else:
                    noise = torch.zeros_like(previous_delta)

        self.delta_estimator.eval()
        return previous_delta

    def __call__(
        self, 
        U_initial: torch.Tensor | jnp.ndarray | np.ndarray, 
        time_steps: int, 
        velocity_term: float | torch.Tensor | jnp.ndarray | np.ndarray, 
        diffusion_term: float | torch.Tensor | jnp.ndarray | np.ndarray | Callable, 
        advection_term: float | torch.Tensor | jnp.ndarray | np.ndarray | Callable, 
        reaction_term: float |torch.Tensor | jnp.ndarray | np.ndarray | Callable,
        dirichlet_bc: List[float | torch.Tensor | jnp.ndarray | np.ndarray] | Callable,
        tol: float = 1e-6,
        max_iter: int = 100
    ):

        coarse_output = self.coarse_integrator(U_initial, time_steps, velocity_term, diffusion_term, advection_term, reaction_term, dirichlet_bc)

        previous_delta = torch.randn_like(coarse_output)
        previous_delta[0] = torch.zeros_like(previous_delta[0])
        n_iterations = 0
        error = torch.inf

        while error > tol and n_iterations < max_iter:
            output_delta = self.delta_estimator(previous_delta)
            error = torch.norm(output_delta - previous_delta, p = 2, dim = -1).max().item()
            previous_delta = output_delta
            n_iterations += 1

        fine_output = coarse_output + output_delta

        return fine_output



