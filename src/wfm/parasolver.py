from turtle import ycor
import torch
import torch.nn as nn
import numpy as np
from scipy import ndimage
import jax
import jax.numpy as jnp
from tqdm import trange
import gc
from typing import Tuple, Dict, Any, Optional, Callable, List, Union, Iterable
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

    def __init__(self, field_out_size: int, mesh_size: Tuple[int, int, int], kernel_size: int | Tuple[int, int, int] = 3, device: torch.device = torch.device("cpu"), max_U_norm: float = 1.0, regularization_lambda: float = 0.1, optimizer_kwargs: Dict[str, Any] = {"lr": 1e-3, "weight_decay": 1e-4}, hidden_size: int = 256):

        super().__init__()

        self.field_out_size = field_out_size
        self.kernel_size = kernel_size if isinstance(kernel_size, Iterable) else (kernel_size, kernel_size, kernel_size)
        self.mesh_size = mesh_size
        self.device = device
        self.max_U_norm = max_U_norm
        self.rho = 1.0
        self.max_Z_abs = max(self.max_U_norm*self.rho, 1.0)
        self.loss_fn = nn.MSELoss()
        
        self.regularization_lambda = regularization_lambda
        self.hidden_size = hidden_size

        self.input_size = (-1, field_out_size, mesh_size[0], mesh_size[1], mesh_size[2])
        # shape = (k*L, 3, 100, 360, 180)

        padding_dims = tuple(self.mesh_size[i] % self.kernel_size[i] for i in range(len(self.mesh_size)))
        paddings = tuple((padding_dims[i] // 2, padding_dims[i] - padding_dims[i] // 2) for i in range(len(padding_dims)))

        self.simplify_3d_space = nn.Sequential(
            nn.Conv3d(
                in_channels = field_out_size,
                out_channels = field_out_size,
                kernel_size = self.kernel_size,
                padding = tuple(sum(pads) for pads in paddings),
                stride = self.kernel_size,
                device = device
            ),
            nn.Dropout(0.1),
            nn.BatchNorm3d(field_out_size)
        )

        self.conv_output_size = (self.input_size[0], self.input_size[1], (self.input_size[2] + padding_dims[0]) // self.kernel_size[0], (self.input_size[3] + padding_dims[1]) // self.kernel_size[1], (self.input_size[4] + padding_dims[2]) // self.kernel_size[2])
        self.input_size_flat = int(np.prod(self.conv_output_size[2:]))

        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.input_size_flat, self.input_size_flat)
        )

        self.residual = nn.Sequential(
            nn.Linear(self.input_size_flat, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, self.input_size_flat)
        )

        self.A = torch.randn(self.input_size_flat, self.input_size_flat, device = self.device)
        self.A = self.rho * self.A / torch.norm(self.A, dim = 1, keepdim = True)
        self.Z = torch.distributions.Uniform(-self.max_Z_abs, self.max_Z_abs).sample((self.input_size_flat,1)).to(self.device)

        self.optimizer = torch.optim.Adam(self.parameters(), **optimizer_kwargs)

    def forward(self, U: torch.Tensor):

        if U.dim() < 6:
            U = U.unsqueeze(0)

        if U.shape[-1] == self.field_out_size:
            U = U.permute(0,1,5,2,3,4)

        initial_shape = U.shape
        # shape = (k, L, 3, 100, 360, 180)

        U_D = U.flatten(start_dim = 0, end_dim = 1)
        U_D = torch.where(torch.isnan(U_D), torch.zeros_like(U_D), U_D)
        U_D = torch.where(torch.isinf(U_D), torch.sign(U_D) * torch.full_like(U_D, fill_value = self.max_U_norm), U_D)

        assert U.shape[-4:] == self.input_size[-4:], f"Input tensor must have shape (*,{self.input_size}), except for dims 0-1, but got {U.shape}"

        U_reduced = self.simplify_3d_space(U_D)
        U_flat = torch.flatten(U_reduced, start_dim = 2)
        U_hat = self.net(U_flat @ self.A + self.Z.T)
        U_tilde = self.residual(U_flat)

        Y = torch.nn.functional.glu(torch.cat([U_hat, U_tilde], dim = -1)) 
        Y = Y.reshape(self.conv_output_size)
        Y = torch.nn.functional.interpolate(Y, size = self.input_size[2:], mode = "trilinear")

        return Y.permute(0,2,3,4,1)

    def _update_parameters(self, U: torch.Tensor, Y: Optional[torch.Tensor] = None):
        """
        In a cycle, the model is expected to receive initially the coarse solution (e.g. CrankNicolson) and a fine solution (e.g. RungeKutta8). The model is trained to estimate the delta between the two. At subsequent cycles, it is no more necessary to provide the fine solution, as the model uses the previous estimates to compute the next one.
        """

        if Y is None:
            if U.shape[0] > 1:
                Y = U[1:]
                U = U[:-1]
            else:
                raise ValueError("U must have at least 2 time steps")

        self.optimizer.zero_grad(set_to_none = True)
        U_hat = self.forward(U)
        loss = self.loss_fn(U + U_hat, Y) + self.regularization_lambda * self.net[-1].weight.norm()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm = 1.0)
        self.optimizer.step()

        return U + U_hat, loss.item()

    def train_delta_estimator(self, U_initial: torch.Tensor, Y_initial: torch.Tensor, n_epochs: int = 100, patience: int = 10, tol: float = 1e-6):

        previous_outputs, losses = [U_initial], []
        previous_output, loss = self._update_parameters(U_initial, Y_initial)
        previous_outputs.append(previous_output)
        losses.append(loss)

        self.train()
        for epoch in trange(n_epochs, desc = "Training delta estimator"):
            previous_output, loss = self._update_parameters(
                torch.cat(previous_outputs[:-1], dim = 0),
                torch.cat(previous_outputs[1:], dim = 0)
            )
            previous_outputs.append(previous_output)
            losses.append(loss)
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item()}")
            if len(losses) > patience:
                if all(l < tol for l in losses[-patience:]):
                    break

        return losses

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

        self.dtype = torch.get_default_dtype() if mesh.library == "torch" else np.float32

class SphericalCrankNicolson(BaseIntegrator):

    def __init__(self, dt: float, mesh: SphericalMesh, mesh_sampling_rate: int | Tuple[int, int, int] = 1):

        super().__init__(dt, mesh, mesh_sampling_rate)

    @torch.no_grad()
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

        for n in trange(time_steps, desc = "Spherical Crank-Nicolson integration"):
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
            output.permute(0,4,1,2,3), 
            size = self.mesh.shape, 
            mode = "trilinear"
        ).permute(0,2,3,4,1)

        if self.mesh.library == "torch":
            return output
        elif self.mesh.library == "jax":
            return jnp.asarray(output)
        else:
            return output.detach().cpu().numpy()



class SphericalRungeKutta8(BaseIntegrator):

    def __init__(self, dt: float, mesh: SphericalMesh, mesh_sampling_rate: int | Tuple[int, int, int] = 1, dtype = None):

        super().__init__(dt, mesh, mesh_sampling_rate)
        
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
        self.b = [25/216, 0, 1408/2565, 2197/4104, -1/5, 0, 0, 0]
        self.c = [0, 1/4, 3/8, 12/13, 1, 1/2, 1, 1]
        self.n_stages = len(self.c)

        if dtype is not None:
            self.dtype = dtype

    def _get_jacobian(self, term, U):
        """Compute Jacobian, handling float/tensor/callable uniformly."""
        field_dim = U.shape[-1]
        if isinstance(term, (int, float)):
            if self.mesh.library == "torch":
                return (torch.eye(field_dim, device=U.device, dtype=U.dtype) * term).reshape(1, 1, 1, field_dim, field_dim)
            elif self.mesh.library == "jax":
                return (jnp.eye(field_dim, dtype=U.dtype) * term).reshape(1, 1, 1, field_dim, field_dim)
            else:
                return (np.eye(field_dim, dtype=U.dtype) * term).reshape(1, 1, 1, field_dim, field_dim)
        return get_jacobian_on_field(term, U)

    def _concat(self, tensors, dim=0):
        """Backend-agnostic concatenation along a given dimension."""
        if self.mesh.library == "torch":
            return torch.cat(tensors, dim=dim)
        elif self.mesh.library == "jax":
            return jnp.concatenate(tensors, axis=dim)
        else:
            return np.concatenate(tensors, axis=dim)

    def _roll(self, tensor, shift, dim):
        """Backend-agnostic roll along a given dimension."""
        if self.mesh.library == "torch":
            return torch.roll(tensor, shifts=shift, dims=dim)
        elif self.mesh.library == "jax":
            return jnp.roll(tensor, shift=shift, axis=dim)
        else:
            return np.roll(tensor, shift=shift, axis=dim)

    def _compute_rhs(
        self,
        U: torch.Tensor | jnp.ndarray | np.ndarray,
        inv_h0, inv_h1, inv_h2,
        laplacian_coefficients: Tuple,
        laplacian_gradient_coefficients: Tuple,
        gradient_coefficients: Tuple,
        tensor_boundary_conditions: List,
        n: int,
        velocity_term=None, diffusion_term=None, advection_term=None, reaction_term=None,
        V_pre=None, D_pre=None, A_pre=None, R_pre=None
    ):
        """Compute the right-hand side of the PDE: du/dt = f(t, u)"""

        field_dim = U.shape[-1]

        V = V_pre if V_pre is not None else self._get_jacobian(velocity_term, U)
        D = D_pre if D_pre is not None else self._get_jacobian(diffusion_term, U)
        A = A_pre if A_pre is not None else self._get_jacobian(advection_term, U)
        R = R_pre if R_pre is not None else self._get_jacobian(reaction_term, U)

        if isinstance(U, torch.Tensor):
            U_mat = U.unsqueeze(-1)
            centered = torch.zeros(*U.shape, field_dim, device=U.device, dtype=U.dtype)
        elif isinstance(U, jnp.ndarray):
            U_mat = U[..., jnp.newaxis]
            centered = jnp.zeros((*U.shape, field_dim), dtype=U.dtype)
        else:
            U_mat = U[..., np.newaxis]
            centered = np.zeros((*U.shape, field_dim), dtype=U.dtype)

        centered += V
        centered -= inv_h0 * laplacian_gradient_coefficients[0] * D
        centered -= 2 * inv_h0**2 * laplacian_coefficients[0] * D
        centered -= inv_h1 * laplacian_gradient_coefficients[1] * D
        centered -= 2 * inv_h1**2 * laplacian_coefficients[1] * D
        centered -= inv_h2 * laplacian_gradient_coefficients[2] * D
        centered -= 2 * inv_h2**2 * laplacian_coefficients[2] * D
        centered -= inv_h0 * gradient_coefficients[0] * A
        centered -= inv_h1 * gradient_coefficients[1] * A
        centered -= inv_h2 * gradient_coefficients[2] * A
        centered += R

        c_int = inv_h0**2 * laplacian_coefficients[0] * D
        c_ext = (inv_h0**2 * laplacian_coefficients[0] * D
                 + inv_h0 * laplacian_gradient_coefficients[0] * D
                 + inv_h0 * gradient_coefficients[0] * A)
        c_right = (inv_h1**2 * laplacian_coefficients[1] * D
                   + inv_h1 * laplacian_gradient_coefficients[1] * D
                   + inv_h1 * gradient_coefficients[1] * A)
        c_left = inv_h1**2 * laplacian_coefficients[1] * D
        c_up = (inv_h2**2 * laplacian_coefficients[2] * D
                + inv_h2 * laplacian_gradient_coefficients[2] * D
                + inv_h2 * gradient_coefficients[2] * A)
        c_down = inv_h2**2 * laplacian_coefficients[2] * D

        int_term = self._concat([
            tensor_boundary_conditions[0][0][n],
            c_int[:-1,:,:] @ U_mat[:-1,:,:]
        ], dim=0)
        ext_term = self._concat([
            c_ext[1:,:,:] @ U_mat[1:,:,:],
            tensor_boundary_conditions[0][1][n]
        ], dim=0)
        right_term = self._roll(c_right @ U_mat, shift=-1, dim=1)
        left_term = self._roll(c_left @ U_mat, shift=1, dim=1)
        up_term = self._concat([
            c_up[:,:,:-1] @ U_mat[:,:,:-1],
            tensor_boundary_conditions[2][1][n]
        ], dim=2)
        down_term = self._concat([
            tensor_boundary_conditions[2][0][n],
            c_down[:,:,1:] @ U_mat[:,:,1:]
        ], dim=2)

        rhs = centered @ U_mat + int_term + ext_term + right_term + left_term + up_term + down_term

        if isinstance(rhs, torch.Tensor):
            return rhs.squeeze(-1)
        elif isinstance(rhs, jnp.ndarray):
            return jnp.squeeze(rhs, axis=-1)
        else:
            return np.squeeze(rhs, axis=-1)

    @torch.no_grad()
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

        X = self.mesh.points[::self.mesh_sampling_rate[0],::self.mesh_sampling_rate[1],::self.mesh_sampling_rate[2]][1:-1,1:-1,1:-1]
        X = X.reshape(*X.shape, 1)
        if self.mesh.library == "torch":
            X = X.to(self.dtype)
            self.local_dx = self.local_dx.to(self.dtype)
        else:
            X = X.astype(self.dtype)
            self.local_dx = self.local_dx.astype(self.dtype)

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

                elif isinstance(b, (int, float)):
                    if self.mesh.library == "torch":
                        B = torch.full(tuple(shape_to_fit), b, device=self.mesh.device, dtype=self.dtype)
                    elif self.mesh.library == "jax":
                        B = jnp.full(tuple(shape_to_fit), b, dtype=self.dtype)
                    else:
                        B = np.full(tuple(shape_to_fit), b, dtype=self.dtype)

                elif b is None:
                    B = U_curr.select(i, -j)
                    B = B.reshape(1, *shape_to_fit[1:]).repeat(shape_to_fit[0], 1, 1, 1, 1)

                elif callable(b):
                    B = b(self.mesh.points.select(i, 0 + j*self.mesh.shape[i])).reshape(tuple(shape_to_fit))[:time_steps]
                else:
                    B = b

                if hasattr(B, "reshape"):
                    B = B.reshape(*B.shape, 1)

                tensor_boundary_conditions[i].append(B)
        
        gc.collect()

        if self.mesh.library == "torch":
            inv_h0 = inv_h0.to(self.dtype)
            inv_h1 = inv_h1.to(self.dtype)
            inv_h2 = inv_h2.to(self.dtype)
        else:
            inv_h0 = inv_h0.astype(self.dtype)
            inv_h1 = inv_h1.astype(self.dtype)
            inv_h2 = inv_h2.astype(self.dtype)

        V_pre = self._get_jacobian(velocity_term, U_curr) if not callable(velocity_term) else None
        D_pre = self._get_jacobian(diffusion_term, U_curr) if not callable(diffusion_term) else None
        A_pre = self._get_jacobian(advection_term, U_curr) if not callable(advection_term) else None
        R_pre = self._get_jacobian(reaction_term, U_curr) if not callable(reaction_term) else None

        dt = self.dt

        for n in trange(time_steps, desc = "Spherical Runge-Kutta 8 integration"):
            stages = []

            for i in range(self.n_stages):
                U_stage = U_curr
                for j in range(min(len(self.a[i]), i)):
                    a_ij = self.a[i][j]
                    if a_ij != 0:
                        U_stage = U_stage + dt * a_ij * stages[j]

                k_i = self._compute_rhs(
                    U_stage, inv_h0, inv_h1, inv_h2,
                    laplacian_coefficients, laplacian_gradient_coefficients, gradient_coefficients,
                    tensor_boundary_conditions, n,
                    velocity_term, diffusion_term, advection_term, reaction_term,
                    V_pre, D_pre, A_pre, R_pre
                )
                stages.append(k_i)

            U_next = U_curr
            for i in range(self.n_stages):
                if self.b[i] != 0:
                    U_next = U_next + dt * self.b[i] * stages[i]

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
            output.permute(0,4,1,2,3), 
            size=self.mesh.shape, 
            mode="trilinear"
        ).permute(0,2,3,4,1)

        if self.mesh.library == "torch":
            return output
        elif self.mesh.library == "jax":
            return jnp.asarray(output)
        else:
            return output.detach().cpu().numpy()


class SphericalSpectralElement(BaseIntegrator):
    """
    Spectral Element Method (SEM) solver for PDEs on spherical meshes.
    
    Uses Legendre-Gauss-Lobatto (LGL) quadrature points and Lagrange interpolating
    polynomials as basis functions. The spatial discretization is done via Galerkin
    projection, and time integration uses Crank-Nicolson method.
    """

    def __init__(self, dt: float, mesh: SphericalMesh, mesh_sampling_rate: int | Tuple[int, int, int] = 1, polynomial_order: int = 4):
        """
        Initialize Spectral Element Method solver.
        
        Args:
            dt: Time step size
            mesh: SphericalMesh instance
            mesh_sampling_rate: Sampling rate for mesh reduction
            polynomial_order: Polynomial order for spectral basis (default: 4)
        """
        super().__init__(dt, mesh, mesh_sampling_rate)
        self.polynomial_order = polynomial_order
        self._setup_spectral_basis()

    def _setup_spectral_basis(self):
        """Set up Legendre-Gauss-Lobatto quadrature points and weights."""
        # LGL points are roots of (1-x^2) * P'_N(x) where P_N is Legendre polynomial
        # For now, we'll use a simplified approach with Chebyshev-like distribution
        # In a full implementation, you'd compute actual LGL points
        
        if self.mesh.library == "torch":
            self.zeros = torch.zeros
            self.ones = torch.ones
            self.linspace = torch.linspace
            self.eye = torch.eye
            self.matmul = torch.matmul
            self.solve_linear = torch.linalg.solve
        elif self.mesh.library == "jax":
            self.zeros = jnp.zeros
            self.ones = jnp.ones
            self.linspace = jnp.linspace
            self.eye = jnp.eye
            self.matmul = jnp.matmul
            self.solve_linear = jnp.linalg.solve
        else:
            self.zeros = np.zeros
            self.ones = np.ones
            self.linspace = np.linspace
            self.eye = np.eye
            self.matmul = np.matmul
            self.solve_linear = np.linalg.solve

    def _compute_lagrange_basis(self, xi: torch.Tensor | jnp.ndarray | np.ndarray, nodes: torch.Tensor | jnp.ndarray | np.ndarray):
        """
        Compute Lagrange basis functions at points xi given nodes.
        
        Args:
            xi: Evaluation points (shape: (n_points,))
            nodes: LGL nodes (shape: (n_nodes,))
            
        Returns:
            Basis function values (shape: (n_points, n_nodes))
        """
        n_nodes = nodes.shape[0]
        n_points = xi.shape[0]
        
        # Initialize basis matrix
        if isinstance(xi, torch.Tensor):
            basis = self.ones((n_points, n_nodes), device=xi.device, dtype=xi.dtype)
        elif isinstance(xi, jnp.ndarray):
            basis = self.ones((n_points, n_nodes), dtype=xi.dtype)
        else:
            basis = self.ones((n_points, n_nodes), dtype=xi.dtype)
        
        # Compute Lagrange polynomials: L_i(x) = ∏_{j≠i} (x - x_j) / (x_i - x_j)
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    if isinstance(xi, torch.Tensor):
                        basis[:, i] *= (xi - nodes[j]) / (nodes[i] - nodes[j] + EPS)
                    elif isinstance(xi, jnp.ndarray):
                        basis = basis.at[:, i].multiply((xi - nodes[j]) / (nodes[i] - nodes[j] + EPS))
                    else:
                        basis[:, i] *= (xi - nodes[j]) / (nodes[i] - nodes[j] + EPS)
        
        return basis

    def _compute_mass_matrix(self, nodes: torch.Tensor | jnp.ndarray | np.ndarray, weights: torch.Tensor | jnp.ndarray | np.ndarray):
        """
        Compute mass matrix using LGL quadrature.
        
        Args:
            nodes: LGL nodes
            weights: LGL weights
            
        Returns:
            Mass matrix M (shape: (n_nodes, n_nodes))
        """
        n_nodes = nodes.shape[0]
        
        # For SEM, mass matrix is diagonal when using LGL quadrature
        # M_ij = ∫ φ_i(x) φ_j(x) dx ≈ Σ_k w_k φ_i(x_k) φ_j(x_k)
        # At LGL nodes, φ_i(x_j) = δ_ij, so M_ij = w_i δ_ij
        
        if isinstance(weights, torch.Tensor):
            M = self.eye(n_nodes, device=weights.device, dtype=weights.dtype) * weights.unsqueeze(0)
        elif isinstance(weights, jnp.ndarray):
            M = self.eye(n_nodes, dtype=weights.dtype) * weights[:, None]
        else:
            M = self.eye(n_nodes, dtype=weights.dtype) * weights[:, None]
        
        return M

    def _compute_stiffness_matrix(self, nodes: torch.Tensor | jnp.ndarray | np.ndarray, weights: torch.Tensor | jnp.ndarray | np.ndarray):
        """
        Compute stiffness matrix (derivative matrix) using LGL quadrature.
        
        Args:
            nodes: LGL nodes
            weights: LGL weights
            
        Returns:
            Stiffness matrix K (shape: (n_nodes, n_nodes))
        """
        n_nodes = nodes.shape[0]
        
        # Compute derivative of Lagrange basis at nodes
        # D_ij = dφ_j/dx(x_i)
        if isinstance(nodes, torch.Tensor):
            D = self.zeros((n_nodes, n_nodes), device=nodes.device, dtype=nodes.dtype)
        elif isinstance(nodes, jnp.ndarray):
            D = self.zeros((n_nodes, n_nodes), dtype=nodes.dtype)
        else:
            D = self.zeros((n_nodes, n_nodes), dtype=nodes.dtype)
        
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i == j:
                    # Diagonal term: sum of 1/(x_i - x_k) for k != i
                    s = 0.0
                    for k in range(n_nodes):
                        if k != i:
                            s += 1.0 / (nodes[i] - nodes[k] + EPS)
                    D[i, j] = s
                else:
                    # Off-diagonal: product term
                    num = 1.0
                    denom = 1.0
                    for k in range(n_nodes):
                        if k != i and k != j:
                            num *= (nodes[i] - nodes[k])
                            denom *= (nodes[j] - nodes[k])
                    D[i, j] = num / (denom * (nodes[i] - nodes[j]) + EPS)
        
        # Stiffness matrix: K = M * D (for 1D, in 3D we'll combine)
        M = self._compute_mass_matrix(nodes, weights)
        K = self.matmul(M, D)
        
        return K

    @torch.no_grad()
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
        """
        Solve PDE using Spectral Element Method.
        
        The PDE is discretized in space using SEM and integrated in time using Crank-Nicolson.
        """
        if self.local_dx.shape[:2] != U_initial.shape[:2]:
            U_initial = U_initial[::self.mesh_sampling_rate[0], ::self.mesh_sampling_rate[1], ::self.mesh_sampling_rate[2], :]
            assert self.local_dx.shape[:2] == U_initial.shape[:2], f"Local spatial step must have shape {U_initial.shape[:2]}, got {self.local_dx.shape[:2]}"

        size_1, size_2, size_3, field_dim = U_initial.shape

        if isinstance(dirichlet_bc, Callable):
            boundary_conditions = [tuple(dirichlet_bc(bp) for bp in tup) for tup in self.mesh.get_boundary_points()]
        else:
            boundary_conditions = dirichlet_bc

        output = torch.empty((time_steps+1, size_1-2, size_2-2, size_3-2, field_dim), device=self.mesh.device)

        # Exclude boundary points for interior solution
        U_curr = U_initial[1:-1,1:-1,1:-1,:]
        output[0] = U_curr
        U_prev = torch.zeros_like(U_curr) if isinstance(U_initial, torch.Tensor) else jnp.zeros_like(U_curr) if isinstance(U_initial, jnp.ndarray) else np.zeros_like(U_curr)

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

        # Prepare boundary conditions
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

        # SEM time integration loop (using Crank-Nicolson)
        for n in trange(time_steps, desc = "Spherical Spectral Element integration"):
            V = get_jacobian_on_field(velocity_term, U_curr)
            D = get_jacobian_on_field(diffusion_term, U_curr)
            A = get_jacobian_on_field(advection_term, U_curr)
            R = get_jacobian_on_field(reaction_term, U_curr)

            # Reshape U for matrix operations
            if U_curr.dim() < 5:
                U_curr_reshaped = U_curr.reshape(*U_curr.shape, 1)
            else:
                U_curr_reshaped = U_curr
            
            if U_prev.dim() < 5:
                U_prev_reshaped = U_prev.reshape(*U_prev.shape, 1)
            else:
                U_prev_reshaped = U_prev

            # Build spatial operators using spectral element approach
            # For SEM, we approximate the spatial derivatives using the spectral basis
            # In practice, we'll use a spectral-like finite difference approximation
            
            # Centered coefficient (similar to finite difference but with spectral accuracy)
            curr_centered_coefficient = torch.zeros_like(U_curr) if isinstance(U_curr, torch.Tensor) else jnp.zeros_like(U_curr) if isinstance(U_curr, jnp.ndarray) else np.zeros_like(U_curr)
            if curr_centered_coefficient.dim() < 5:
                curr_centered_coefficient = curr_centered_coefficient.reshape(*curr_centered_coefficient.shape, 1)
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

            next_coefficient = 1 / k**2 + V / k

            # Neighbor coefficients (spectral-like stencil)
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

            # Boundary terms
            int_term = self.concat([
                tensor_boundary_conditions[0][0][n],
                curr_int_coefficient[:-1,:,:] @ U_curr_reshaped[:-1,:,:]
            ])
            ext_term = self.concat([
                curr_ext_coefficient[1:,:,:] @ U_curr_reshaped[1:,:,:],
                tensor_boundary_conditions[0][1][n]
            ])
            right_term = self.roll(curr_right_coefficient @ U_curr_reshaped, shifts=-1, dims=1)
            left_term = self.roll(curr_left_coefficient @ U_curr_reshaped, shifts=1, dims=1)
            up_term = self.concat([
                curr_up_coefficient[:,:,:-1] @ U_curr_reshaped[:,:,:-1],
                tensor_boundary_conditions[2][1][n]
            ], axis=2)
            down_term = self.concat([
                tensor_boundary_conditions[2][0][n],
                curr_down_coefficient[:,:,1:] @ U_curr_reshaped[:,:,1:]
            ], axis=2)
            
            prev_coefficient = -k**(-2)

            # Crank-Nicolson update: (M + k/2 * L) U^{n+1} = (M - k/2 * L) U^n + k * f
            # Simplified to match the structure of SphericalCrankNicolson
            explicit_term = curr_centered_coefficient*U_curr_reshaped + prev_coefficient*U_prev_reshaped + int_term + ext_term + right_term + left_term + up_term + down_term
            
            # Solve for next time step
            if isinstance(explicit_term, torch.Tensor):
                U_next = torch.sum(next_coefficient.transpose(-1,-2) * explicit_term, dim=-1)
            elif isinstance(explicit_term, jnp.ndarray):
                U_next = jnp.sum(next_coefficient.transpose(-1,-2) * explicit_term, axis=-1)
            else:
                U_next = np.sum(next_coefficient.transpose(-1,-2) * explicit_term, axis=-1)
            
            output[n+1] = U_next
            U_prev = U_curr
            U_curr = U_next

        # Apply boundary conditions
        for i, bc in enumerate(tensor_boundary_conditions):
            num_dims = len(output.shape)
            pad_tuple = sum([[1, int(dim > 0)] if dim <= i else [0, 0] for dim in range(num_dims)], [])[::-1]
            
            padded_bc0 = torch.nn.functional.pad(bc[0].squeeze(-1), pad_tuple, mode='constant', value=0)
            padded_bc1 = torch.nn.functional.pad(bc[1].squeeze(-1), pad_tuple, mode='constant', value=0)
            
            output = torch.cat([padded_bc0, output, padded_bc1], dim=i+1)

        output = torch.nn.functional.interpolate(
            output.permute(0,4,1,2,3), 
            size=self.mesh.shape, 
            mode="trilinear"
        ).permute(0,2,3,4,1)

        if self.mesh.library == "torch":
            return output
        elif self.mesh.library == "jax":
            return jnp.asarray(output)
        else:
            return output.detach().cpu().numpy()


class Parareal(BaseIntegrator):

    def __init__(self, coarse_integrator: Callable, fine_integrator: Callable, delta_estimator: Callable, dt: float, mesh: GenericMesh, mesh_sampling_rate: int | Tuple[int, int, int] = 1, max_U_norm: float = 1.0):
        """
        Initialize Parareal algorithm.
        
        Args:
            coarse_integrator: Coarse integrator function
            fine_integrator: Fine integrator function
            delta_estimator: Delta estimator function
        """
        super().__init__(dt, mesh, mesh_sampling_rate)

        if isinstance(coarse_integrator, BaseIntegrator):
            self.coarse_integrator = coarse_integrator
        else:
            self.coarse_integrator = coarse_integrator(dt, mesh, mesh_sampling_rate)

        if isinstance(fine_integrator, BaseIntegrator):
            self.fine_integrator = fine_integrator
        else:
            self.fine_integrator = fine_integrator(dt, mesh, mesh_sampling_rate)

        self.delta_estimator = delta_estimator
        self.max_U_norm = max_U_norm

    def train_delta_estimator(self, U_initial: torch.Tensor | jnp.ndarray | np.ndarray, time_steps: int, velocity_term: float | torch.Tensor | jnp.ndarray | np.ndarray, diffusion_term: float | torch.Tensor | jnp.ndarray | np.ndarray | Callable, advection_term: float | torch.Tensor | jnp.ndarray | np.ndarray | Callable, reaction_term: float |torch.Tensor | jnp.ndarray | np.ndarray | Callable, dirichlet_bc: List[float | torch.Tensor | jnp.ndarray | np.ndarray] | Callable, fine_integrator: Callable = None, n_epochs: int = 100, optimizer_name: str = "AdamW", optimizer_kwargs: Dict[str, Any] = {"lr": 1e-3, "weight_decay": 1e-4}, patience: int = 10, tol: float = 1e-6):
        """
        Train the delta estimator. The delta estimator receives the coarse solution and estimates the delta between the coarse and fine solutions.
        
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

        patience = min(patience, n_epochs//2 + 1)

        coarse_output = self.coarse_integrator(U_initial, time_steps, velocity_term, diffusion_term, advection_term, reaction_term, dirichlet_bc)
        coarse_output = torch.where(torch.isnan(coarse_output), torch.zeros_like(coarse_output), coarse_output)
        coarse_output = torch.where(torch.isinf(coarse_output), torch.sign(coarse_output) * torch.full_like(coarse_output, fill_value = self.max_U_norm), coarse_output)

        fine_output = self.fine_integrator(U_initial, time_steps, velocity_term, diffusion_term, advection_term, reaction_term, dirichlet_bc) if fine_integrator is None else fine_integrator(U_initial, time_steps, velocity_term, diffusion_term, advection_term, reaction_term, dirichlet_bc)
        fine_output = torch.where(torch.isnan(fine_output), torch.zeros_like(fine_output), fine_output)
        fine_output = torch.where(torch.isinf(fine_output), torch.sign(fine_output) * torch.full_like(fine_output, fill_value = self.max_U_norm), fine_output)

        delta_gt = fine_output - coarse_output

        optimizer = getattr(torch.optim, optimizer_name)(self.delta_estimator.parameters(), **optimizer_kwargs)
        latest_losses = []
        previous_output = coarse_output
        noise = torch.zeros_like(delta_gt)

        self.delta_estimator.train()
        for epoch in trange(n_epochs, desc = "Training delta estimator"):
            optimizer.zero_grad(set_to_none = True)
            output_delta = self.delta_estimator(previous_output + noise)
            fine_output = coarse_output + output_delta
            loss = torch.norm(fine_output - previous_output, p = 2, dim = -1).pow(2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.delta_estimator.parameters(), max_norm = 1.0)
            optimizer.step()
            previous_output = fine_output.detach()
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item()}")
            latest_losses.append(loss.item())
            if len(latest_losses) > patience:
                latest_losses.pop(0)
                if all(l < tol for l in latest_losses):
                    noise = torch.randn_like(output_delta) * output_delta.std() / torch.sqrt(torch.tensor(time_steps))
                else:
                    noise = torch.zeros_like(output_delta)

            gc.collect()

        self.delta_estimator.eval()
        return output_delta

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

        n_iterations = 0
        error = torch.inf
        previous_output = coarse_output

        while error > tol and n_iterations < max_iter:
            output_delta = self.delta_estimator(previous_output)
            fine_output = coarse_output + output_delta
            error = torch.norm(fine_output - previous_output, p = 2, dim = -1).max().item()
            previous_output = fine_output
            n_iterations += 1

        return fine_output



