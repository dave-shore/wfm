import torch
import torch.nn as nn
import numpy as np
import jax
import jax.numpy as jnp
from typing import Tuple, Dict, Any, Optional, Callable
from .spherical_mesh import SphericalMesh


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


class CrankNicolson:

    def __init__(self, dt: float, mesh: SphericalMesh, mesh_sampling_rate: int | Tuple[int, int, int] = 1, dist_kernel_size: int = 3):

        self.dt = dt
        self.mesh = mesh
        self.mesh_sampling_rate = mesh_sampling_rate if isinstance(mesh_sampling_rate, tuple) else (mesh_sampling_rate, mesh_sampling_rate, mesh_sampling_rate)
        assert len(self.mesh_sampling_rate) == len(self.mesh.shape), f"Mesh sampling rate must be a tuple of {len(self.mesh.shape)} integers, got {mesh_sampling_rate}"

        actual_mesh_size = tuple(self.mesh.shape[i] // self.mesh_sampling_rate[i] for i in range(len(self.mesh.shape)))
        self.dist_kernel_size = dist_kernel_size if isinstance(dist_kernel_size, tuple) else (dist_kernel_size, dist_kernel_size, dist_kernel_size)
        for a,b in zip(actual_mesh_size, self.dist_kernel_size):
            assert a >= b and a % b == 0, f"Dist kernel size {b} must be lower than and a divisor of the actual mesh size {a}"
        self.dist_kernel_size = tuple(d + int(d % 2 == 0) for d in self.dist_kernel_size)

        self.conv_matrix = - 1* np.ones(self.dist_kernel_size)
        self.conv_matrix[self.dist_kernel_size[0] // 2, self.dist_kernel_size[1] // 2, self.dist_kernel_size[2] // 2] = int(np.prod(self.dist_kernel_size)) - 1
        self.conv_matrix /= np.prod(self.dist_kernel_size)

        if mesh.library == "torch":
            self.dist_convolution = nn.Conv3d(3, 3, kernel_size = self.dist_kernel_size, bias = False, padding = tuple(d // 2 for d in self.dist_kernel_size))ù
            self.dist_convolution.weight.data = torch.from_numpy(self.conv_matrix)
            self.permute_op = torch.permute
            self.norm_op = torch.norm
        elif mesh.library == "jax":
            self.dist_convolution = lambda X: jax.lax.conv_general_dilated(
                X, jnp.array(self.conv_matrix), (1,1,1), (1,1,1), padding = tuple(d // 2 for d in self.dist_kernel_size))
            self.permute_op = jnp.permute_dims
            self.norm_op = jnp.norm
        else:
            self.dist_convolution = lambda X: np.convolve(X, self.conv_matrix, mode = "valid")
            self.permute_op = np.permute_dims
            self.norm_op = np.linalg.norm


    def _compute_avg_local_spatial_step(self):

        X = self.mesh.cartesian_coordinates()
        X = X[::self.mesh_sampling_rate[0], ::self.mesh_sampling_rate[1], ::self.mesh_sampling_rate[2], :]
        X = self.permute_op(X, (3,0,1,2))
        avg_local_dists = self.dist_convolution(X)
        # Shape should not change

        return self.norm_op(avg_local_dists, axis = 0)


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