import torch
import torch.nn as nn
from tltorch import TuckerTensor
import numpy as np
from sklearn.linear_model import Ridge
from scipy import ndimage
import jax
import jax.numpy as jnp
from tqdm import trange
import os
import gc
from typing import Tuple, Dict, Any, Optional, Callable, List, Union, Iterable
from .spherical_mesh import *
from .base import *
from .functional import ConvReduction


os.environ['SCIPY_ARRAY_API'] = "1"


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


def _safe_clamp(tensor, max_val=UB):
    """Clamp values and replace NaN/Inf for numerical stability."""
    if isinstance(tensor, torch.Tensor):
        tensor = torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor)
        tensor = torch.where(torch.isinf(tensor), torch.sign(tensor) * max_val, tensor)
        tensor = torch.clamp(tensor, -max_val, max_val)
    elif isinstance(tensor, jnp.ndarray):
        tensor = jnp.where(jnp.isnan(tensor), jnp.zeros_like(tensor), tensor)
        tensor = jnp.where(jnp.isinf(tensor), jnp.sign(tensor) * max_val, tensor)
        tensor = jnp.clip(tensor, -max_val, max_val)
    else:
        tensor = np.where(np.isnan(tensor), np.zeros_like(tensor), tensor)
        tensor = np.where(np.isinf(tensor), np.sign(tensor) * max_val, tensor)
        tensor = np.clip(tensor, -max_val, max_val)
    return tensor


def get_jacobian_on_field(func: torch.Tensor | jnp.ndarray | np.ndarray | Callable, field: torch.Tensor | jnp.ndarray | np.ndarray, interpolation_factor: int = 4, **kwargs):

    field = field.squeeze()

    if not isinstance(func, Callable):
        field_out_dim = field.shape[-1]
        if isinstance(func, (int, float)):
            if isinstance(field, torch.Tensor):
                return (torch.eye(field_out_dim, device=field.device, dtype=field.dtype) * func).reshape(1, 1, 1, field_out_dim, field_out_dim)
            elif isinstance(field, jnp.ndarray):
                return (jnp.eye(field_out_dim, dtype=field.dtype) * func).reshape(1, 1, 1, field_out_dim, field_out_dim)
            else:
                return (np.eye(field_out_dim, dtype=field.dtype) * func).reshape(1, 1, 1, field_out_dim, field_out_dim)
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


class RandNetCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.random_layer = nn.Linear(input_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size, bias = False)

        torch.nn.init.trunc_normal_(self.random_layer.weight)
        torch.nn.init.trunc_normal_(self.random_layer.bias)
        torch.nn.init.trunc_normal_(self.output_layer.weight)

    def forward(self, x: torch.Tensor, hidden_state: torch.Tensor):

        x_random = self.random_layer(x)
        h_random = self.random_layer(hidden_state)

        h_output = self.output_layer(h_random + x_random)

        return h_output


class RandNetRNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.randnet_cell = RandNetCell(input_size, hidden_size, output_size)
        
    def forward(self, x: torch.Tensor, dim = 1):

        x = x.transpose(dim, 0)
        h = torch.zeros_like(x)

        for x_step in x:
            h = self.randnet_cell(x_step.squeeze(), h)

        return h.transpose(dim, 0)


def _reduce_3d_to_1d_tucker(U: torch.Tensor, max_rank: int = 8):

    U = torch.as_tensor(U, dtype = torch.get_default_dtype())
    ranks = [U.shape[0], U.shape[1]] + [max_rank if s > max_rank else s for s in U.shape[2:]]
    core_tensor = TuckerTensor.from_tensor(U, rank = ranks)

    return core_tensor.decomposition[0].flatten(start_dim = 2), ranks, core_tensor.decomposition[1]


def _expand_1d_to_3d_tucker(core_tensor: torch.Tensor, ranks: Tuple[int, int, int], factors: Tuple[torch.Tensor, ...]):

    core_tensor = torch.as_tensor(core_tensor, dtype = factors[0].dtype)
    core_tensor = core_tensor.reshape(*ranks)

    return TuckerTensor(core_tensor, factors, rank = ranks).to_tensor()



class RandNetTorch(nn.Module):
    """
    Shallow random neural network used to estimate the delta between a coarse integrator (e.g. Runge-Kutta 1) and the actual solution of a differential equation.
    """

    def __init__(self, field_out_size: int, mesh_size: Tuple[int, int, int], kernel_size: int = 8, device: torch.device = torch.device("cpu"), max_U_norm: float = 1.0, regularization_lambda: float = 0.01, optimizer_kwargs: Dict[str, Any] = {"lr": 1e-2}, max_dataset_size: int = 1000, use_explicit_ridge: bool = False):

        super().__init__()

        self.field_out_size = field_out_size
        self.kernel_size = kernel_size 
        self.mesh_size = mesh_size
        self.device = device
        self.max_U_norm = max_U_norm
        self.rho = 1.0
        self.max_Z_abs = max(self.max_U_norm*self.rho, 1.0)
        self.loss_fn = nn.MSELoss()

        self.use_explicit_ridge = use_explicit_ridge
        self.regularization_lambda = regularization_lambda

        self.input_size = (-1, field_out_size, mesh_size[0], mesh_size[1], mesh_size[2])
        # shape = (k*L, 3, 100, 360, 180)

        # Need to simplify the 3D space to a 2D space, by convolving with a kernel
        self.simplify_3d_space = ConvReduction(kernel_size = kernel_size, stride = 1, padding = 0, channels = field_out_size, ndim = 3)
        self.backconv_3d_space = self.simplify_3d_space.inverse_op

        self.conv_output_size = [self.input_size[0], self.input_size[1]] + [min(self.kernel_size, s) for s in self.input_size[2:]]
        self.input_size_flat = int(np.prod(self.conv_output_size[2:]))
        # output here should have shape (batch_size, L, 3, 8, 8, 8)

        self.random_layer = nn.Linear(self.input_size_flat, self.input_size_flat)
        for param in self.random_layer.parameters():
            torch.nn.init.trunc_normal_(param)
            param.requires_grad = False

        self.output_layer = Ridge(alpha = regularization_lambda)

        self.interior_dataset = torch.empty(0, self.input_size_flat, device = device, dtype = torch.float32)
        # shape = (-1, 3*8*8*8)
        self.max_dataset_size = max_dataset_size


    def _check_dataset_size(self):
        if self.interior_dataset.shape[0] >= self.max_dataset_size:
            self.interior_dataset = self.interior_dataset[-self.max_dataset_size:]


    def forward(self, U: torch.Tensor):

        if U.dim() < 6:
            U = U.unsqueeze(0)

        if U.shape[-1] == self.field_out_size:
            U = U.permute(0,1,5,2,3,4)

        U_D = U.flatten(start_dim = 0, end_dim = 1)
        U_D = torch.where(torch.isnan(U_D), torch.zeros_like(U_D), U_D)
        U_D = torch.where(torch.isinf(U_D), torch.sign(U_D) * torch.full_like(U_D, fill_value = self.max_U_norm), U_D)

        assert U.shape[-4:] == self.input_size[-4:], f"Input tensor must have shape (*,{self.input_size}), except for dims 0-1, but got {U.shape}"

        U_flat, ranks, factors = self.simplify_3d_space(U_D)
        U_random = self.random_layer(U_flat).flatten(end_dim = 1)
        # U_random.shape = (batch_size*L, 3*8*8*8)

        # Train Ridge Regression model on the dataset collected so far, if any
        if self.interior_dataset.numel() > 0:
            self.output_layer.fit(self.interior_dataset.detach(), torch.zeros_like(self.interior_dataset).detach())
        else:
            self.output_layer.fit(torch.randn_like(U_random).detach(), torch.zeros_like(U_random).detach())

        U_reg = self.output_layer.predict(U_random.detach()).reshape(U_flat.shape)

        U_output = self.backconv_3d_space(U_reg, ranks, factors)
        if U_output.dim() < U.dim():
            U_output = U_output.unsqueeze(0)
        shape_differences = [U.shape[i] - U_output.shape[i] for i in range(len(U_output.shape))][::-1]
        shape_paddings = sum([[delta // 2, sum(divmod(delta, 2))] for delta in shape_differences], [])
        U_output = torch.nn.functional.pad(U_output.reshape(-1, *U_output.shape[-3:]), tuple(shape_paddings[:-6]), mode = "replicate")
        U_output = U_output.reshape(*U.shape)

        self.interior_dataset = torch.cat([self.interior_dataset, U_random], dim = 0)
        self._check_dataset_size()

        return U_output



class RadialBasisFunctionTorch(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.exp(-x**2)


class ELM_base(nn.Module):
    """
    From https://github.com/Parallel-in-Time-Differential-Equations/RandNet-Parareal/blob/main/models.py
    """
    
    def __init__(self, d, seed=47, res_size=500, loss='relu', M=1, R=1, alpha=0.1, degree=1, m=5):

        from sklearn.preprocessing import PolynomialFeatures

        self.d = d
        self.N = res_size
        self.rng = np.random.default_rng(seed)
        self.m = m
        
        radbas = RadialBasisFunctionTorch()
        relu = nn.ReLU()
        tanh = nn.Tanh()
        self.loss = lambda x: torch.stack([radbas(x), relu(x), tanh(x)]).mean(dim = 0)
        self.M = M
        self.R = R
        self.alpha = alpha
        self.mdl = Ridge(alpha=self.alpha)
        self.poly = PolynomialFeatures(degree=degree)
        self.poly.fit(torch.zeros((1, d)))
        self.degree = self.poly.n_output_features_
        
        bias, C = self._init_obj()
        self.bias, self.C = bias, C

        
    def _init_obj(self):
        N, rng = self.N, self.rng
        bias = torch.as_tensor(rng.uniform(-1, 1, (N, 1)))
        C = torch.as_tensor(rng.uniform(-1, 1, (N, self.degree)))
        return bias, C
    
    def _fit(self, x, y, bias, C):

        x = self.poly.fit_transform(x)
        X = self.loss(bias + C @ x.T) # activation
        X = X.T #first col is intercept
        self.mdl.fit(X, y)

    def fit(self, x, y, k):
        self.x = x
        self.y = y
        self.k = k
        
    def predict(self, new_x):
        bias = self.M * self.R * self.bias
        bias = self.bias
        C = self.R * self.C

        s_idx = torch.argsort(scipy.spatial.distance.cdist(new_x, self.x, metric='sqeuclidean')[0,:])
        xm = self.x[s_idx[:self.m], :]
        ym = self.y[s_idx[:self.m], :]
        
        new_X = self.poly.fit_transform(new_x)
        _int = bias + C @ new_X.T
        new_X = self.loss(_int)
        self._fit(xm, ym, bias, C)
        return torch.squeeze(self.mdl.predict(new_X.T))
        
    def forward(self, x, y, new_x, k):
        self.fit(x, y, k)
        return self.predict(new_x)
    


class BaseIntegrator():
    """
    Base class for PDE integrators.

    Canonical tensor dimension convention
    ----------------------------------------
    Field tensors : ``(batch_size, sequence_length, [radial], [longitude], [latitude], field_dim)``
    Mesh / dx     : ``(    1,           1,          [radial], [longitude], [latitude],    1     )``

    ``batch_size``, ``sequence_length`` and ``field_dim`` are automatic
    singletons for mesh and dx variables.  One or two of the three spatial
    dimensions (radial, longitude, latitude) may be absent depending on
    mesh dimensionality; ``canonical_ndim`` adjusts accordingly.
    """

    SPATIAL_DIM_NAMES = ("radial", "longitude", "latitude")

    def __init__(self, dt: float, mesh: GenericMesh, mesh_sampling_rate: int | Tuple[int, int, int] = 1):

        self.dt = dt
        self.mesh = mesh

        self.spatial_ndim = len(self.mesh.shape)
        self.canonical_ndim = 2 + self.spatial_ndim + 1

        self.mesh_sampling_rate = mesh_sampling_rate if isinstance(mesh_sampling_rate, tuple) else (mesh_sampling_rate, mesh_sampling_rate, mesh_sampling_rate)

        self.actual_mesh_size = tuple(self.mesh.shape[i] // self.mesh_sampling_rate[i] for i in range(len(self.mesh.shape)))
        x = self.mesh.points[::self.mesh_sampling_rate[0], ::self.mesh_sampling_rate[1], ::self.mesh_sampling_rate[2], :]

        if mesh.library == "torch":
            local_dx_components = []
            for i in range(x.shape[-1]):
                xi = x[..., i]
                prep_shape = list(xi.shape)
                prep_shape[i] = 1
                prep = torch.zeros(prep_shape, device=xi.device, dtype=xi.dtype)
                local_dx_components.append(torch.diff(xi, dim=i, prepend=prep))
            self.local_dx = torch.stack(local_dx_components, dim=-1)
            self.sin = torch.sin
            self.cos = torch.cos
            self.roll = torch.roll
            self.concat = torch.cat
            self.solve = torch.linalg.lstsq
            self.full = torch.full

        elif mesh.library == "jax":
            local_dx_components = []
            for i in range(x.shape[-1]):
                xi = x[..., i]
                prep_shape = list(xi.shape)
                prep_shape[i] = 1
                prep = jnp.zeros(prep_shape, dtype=xi.dtype)
                local_dx_components.append(jnp.diff(xi, axis=i, prepend=prep))
            self.local_dx = jnp.stack(local_dx_components, axis=-1)
            self.sin = jnp.sin
            self.cos = jnp.cos
            self.roll = jnp.roll
            self.concat = jnp.concatenate
            self.solve = jnp.linalg.lstsq
            self.full = jnp.full

        else:
            local_dx_components = []
            for i in range(x.shape[-1]):
                xi = x[..., i]
                prep_shape = list(xi.shape)
                prep_shape[i] = 1
                prep = np.zeros(prep_shape, dtype=xi.dtype)
                local_dx_components.append(np.diff(xi, axis=i, prepend=prep))
            self.local_dx = np.stack(local_dx_components, axis=-1)
            self.sin = np.sin
            self.cos = np.cos
            self.roll = np.roll
            self.concat = np.concatenate
            self.solve = np.linalg.lstsq
            self.full = np.full

        self.dtype = torch.get_default_dtype() if mesh.library == "torch" else np.float32

        self._canonical_local_dx = self._prepend_unit_dims(self.local_dx, 2)
        self._canonical_mesh_points = self._prepend_unit_dims(x, 2)

    # ------------------------------------------------------------------
    # Dimension-convention helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _prepend_unit_dims(tensor, n):
        """Prepend *n* singleton dimensions to *tensor*."""
        return tensor[(None,) * n]

    def _normalize_field(self, U):
        """Pad a field tensor to the canonical layout.

        Missing leading dimensions (batch, sequence) are inserted as
        singletons.  Returns ``(padded_tensor, n_dims_added)``.
        """
        added = 0
        ndim = U.ndim if hasattr(U, 'ndim') else np.asarray(U).ndim
        while ndim < self.canonical_ndim:
            if isinstance(U, torch.Tensor):
                U = U.unsqueeze(0)
            elif isinstance(U, jnp.ndarray):
                U = jnp.expand_dims(U, axis=0)
            else:
                U = np.expand_dims(U, axis=0)
            added += 1
            ndim += 1
        return U, added

    def _denormalize_field(self, U, n_dims_added):
        """Remove leading singleton dimensions that were added by
        :meth:`_normalize_field`."""
        for _ in range(n_dims_added):
            if U.shape[0] == 1:
                if isinstance(U, torch.Tensor):
                    U = U.squeeze(0)
                elif isinstance(U, jnp.ndarray):
                    U = jnp.squeeze(U, axis=0)
                else:
                    U = np.squeeze(U, axis=0)
            else:
                break
        return U

    @property
    def batch_dim(self):
        """Index of the batch dimension in the canonical layout."""
        return 0

    @property
    def seq_dim(self):
        """Index of the sequence / time dimension in the canonical layout."""
        return 1

    @property
    def field_dim_index(self):
        """Index of the field dimension (always last) in the canonical layout."""
        return 2 + self.spatial_ndim

    @property
    def spatial_dim_indices(self):
        """Indices of the spatial dimensions in the canonical layout."""
        return tuple(range(2, 2 + self.spatial_ndim))

    def _spatial_shape(self, U=None):
        """Return the spatial extents from a canonical tensor or the mesh."""
        if U is not None:
            return U.shape[2:2 + self.spatial_ndim]
        return self.actual_mesh_size

    # ------------------------------------------------------------------
    # Batch dispatch
    # ------------------------------------------------------------------

    @torch.no_grad()
    def __call__(self, U_initial, time_steps, *args, **kwargs):
        """Run the integrator on a (possibly batched) field.

        ``U_initial`` is normalised to the canonical layout
        ``(batch_size, 1, *spatial, field_dim)`` and the per-sample
        integration is dispatched to :meth:`_integrate_single`.  The
        per-sample outputs of shape
        ``(time_steps + 1, *spatial_full, field_dim)`` are stacked along
        a new batch dimension to produce the canonical output shape
        ``(batch_size, time_steps + 1, *spatial_full, field_dim)``.

        If the original input had a smaller rank than the canonical
        layout (e.g. an unbatched ``(*spatial, field_dim)`` tensor), the
        leading singleton dimensions added by :meth:`_normalize_field`
        are stripped from the output before returning.
        """
        U_initial, _n_dims_added = self._normalize_field(U_initial)
        batch_size = U_initial.shape[0]

        sample_outputs = [
            self._integrate_single(U_initial[b, 0], time_steps, *args, **kwargs)
            for b in range(batch_size)
        ]

        output = self._stack_along_batch(sample_outputs)
        output = self._denormalize_field(output, _n_dims_added)
        return self._convert_to_library(output)

    def _integrate_single(self, U_initial, time_steps, *args, **kwargs):
        """Integrate a single (unbatched, unsequenced) sample.

        Subclasses must override this method to accept ``U_initial``
        of shape ``(*spatial, field_dim)`` and return a tensor of shape
        ``(time_steps + 1, *spatial_full, field_dim)``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement _integrate_single"
        )

    def _stack_along_batch(self, sample_outputs):
        """Stack per-sample outputs along a new batch dimension."""
        if self.mesh.library == "torch":
            return torch.stack([
                o if isinstance(o, torch.Tensor) else torch.as_tensor(o)
                for o in sample_outputs
            ], dim=0)
        elif self.mesh.library == "jax":
            return jnp.stack([
                jnp.asarray(o.detach().cpu().numpy()) if isinstance(o, torch.Tensor)
                else jnp.asarray(o)
                for o in sample_outputs
            ], axis=0)
        else:
            return np.stack([
                o.detach().cpu().numpy() if isinstance(o, torch.Tensor)
                else np.asarray(o)
                for o in sample_outputs
            ], axis=0)

    def _convert_to_library(self, output):
        """Convert *output* to the array library declared by ``self.mesh``."""
        if self.mesh.library == "torch":
            if isinstance(output, torch.Tensor):
                return output
            if isinstance(output, jnp.ndarray):
                return torch.as_tensor(np.asarray(output))
            return torch.as_tensor(output)
        elif self.mesh.library == "jax":
            if isinstance(output, jnp.ndarray):
                return output
            if isinstance(output, torch.Tensor):
                return jnp.asarray(output.detach().cpu().numpy())
            return jnp.asarray(output)
        else:
            if isinstance(output, torch.Tensor):
                return output.detach().cpu().numpy()
            return np.asarray(output)


def _build_spherical_coefficients(X, sin_fn, cos_fn):
    """
    Build the correct spherical Laplacian, Laplacian-gradient, and gradient
    coefficient tuples for grid dimensions (dim0=radial, dim1=longitude/phi,
    dim2=colatitude/theta).

    Point coordinates: X[..., 0]=r, X[..., 1]=lon(phi), X[..., 2]=lat(theta/colatitude).
    """
    r = X[..., [0], :]
    theta = X[..., [2], :]  # colatitude
    sin_theta = sin_fn(theta) + EPS
    cos_theta = cos_fn(theta)

    # Spherical Laplacian: d²/dr² + (2/r)d/dr
    #   + (1/r²)d²/dθ² + (cosθ/(r²sinθ))d/dθ
    #   + (1/(r²sin²θ))d²/dφ²
    laplacian_coefficients = (
        1,                          # dim 0 (r): coefficient of d²/dr²
        (r * sin_theta)**(-2),      # dim 1 (φ/lon): 1/(r²sin²θ)
        r**(-2),                    # dim 2 (θ/colat): 1/r²
    )
    laplacian_gradient_coefficients = (
        2 * r**(-1),                        # dim 0 (r): 2/r
        0,                                  # dim 1 (φ/lon): no gradient correction
        cos_theta * (r**2 * sin_theta)**(-1),  # dim 2 (θ/colat): cosθ/(r²sinθ)
    )
    # Gradient operator: d/dr, (1/(r sinθ))d/dφ, (1/r)d/dθ
    gradient_coefficients = (
        1,                          # dim 0 (r)
        (r * sin_theta)**(-1),      # dim 1 (φ/lon): 1/(r sinθ)
        r**(-1),                    # dim 2 (θ/colat): 1/r
    )
    return laplacian_coefficients, laplacian_gradient_coefficients, gradient_coefficients


def _build_inv_h(H):
    """Build clamped inverse spatial steps from H."""
    inv_h0 = (H[..., [0], :] + EPS)**(-1)
    inv_h1 = (H[..., [1], :] + EPS)**(-1)
    inv_h2 = (H[..., [2], :] + EPS)**(-1)
    inv_h0 = torch.clamp(inv_h0, -UB, UB) if isinstance(inv_h0, torch.Tensor) else np.clip(inv_h0, -UB, UB)
    inv_h1 = torch.clamp(inv_h1, -UB, UB) if isinstance(inv_h1, torch.Tensor) else np.clip(inv_h1, -UB, UB)
    inv_h2 = torch.clamp(inv_h2, -UB, UB) if isinstance(inv_h2, torch.Tensor) else np.clip(inv_h2, -UB, UB)
    return inv_h0, inv_h1, inv_h2


def _prepare_boundary_conditions(boundary_conditions, time_steps, size_1, size_2, size_3, field_dim, U_curr, full_fn, mesh, dtype):
    """Prepare boundary condition tensors for all integrators."""
    tensor_boundary_conditions = []
    for i, bc in enumerate(boundary_conditions):
        shape_to_fit = [time_steps, size_1 - 2, size_2 - 2, size_3 - 2, field_dim]
        shape_to_fit[i + 1] = 1
        tensor_boundary_conditions.append([])
        for j, b in enumerate(bc):
            if isinstance(b, (torch.Tensor, jnp.ndarray, np.ndarray)):
                B = fit_tensor_shape(b, tuple(shape_to_fit))
            elif isinstance(b, (int, float)):
                if mesh.library == "torch":
                    B = torch.full(tuple(shape_to_fit), b, device=mesh.device, dtype=dtype)
                elif mesh.library == "jax":
                    B = jnp.full(tuple(shape_to_fit), b, dtype=dtype)
                else:
                    B = np.full(tuple(shape_to_fit), b, dtype=dtype)
            elif b is None:
                B = U_curr.select(i, -j)
                B = B.reshape(1, *shape_to_fit[1:]).repeat(shape_to_fit[0], 1, 1, 1, 1)
            elif callable(b):
                B = b(mesh.points.select(i, 0 + j * mesh.shape[i])).reshape(tuple(shape_to_fit))[:time_steps]
            else:
                B = b

            if hasattr(B, "reshape"):
                B = B.reshape(*B.shape, 1)

            tensor_boundary_conditions[i].append(B)
    return tensor_boundary_conditions


class SphericalCrankNicolson(BaseIntegrator):
    """
    Crank-Nicolson integrator for second-order-in-time PDEs on spherical meshes.

    Solves V·∂u/∂t + ∂²u/∂t² = D·∇²u + A·∇u + R·u  using an implicit
    Crank-Nicolson-style time discretisation with second-order finite
    differences in space.
    """

    def __init__(self, dt: float, mesh: SphericalMesh, mesh_sampling_rate: int | Tuple[int, int, int] = 1):

        super().__init__(dt, mesh, mesh_sampling_rate)

    @torch.no_grad()
    def _integrate_single(
        self,
        U_initial: torch.Tensor | jnp.ndarray | np.ndarray,
        time_steps: int,
        velocity_term: float | torch.Tensor | jnp.ndarray | np.ndarray,
        diffusion_term: float | torch.Tensor | jnp.ndarray | np.ndarray | Callable,
        advection_term: float | torch.Tensor | jnp.ndarray | np.ndarray | Callable,
        reaction_term: float | torch.Tensor | jnp.ndarray | np.ndarray | Callable,
        dirichlet_bc: List[float | torch.Tensor | jnp.ndarray | np.ndarray] | Callable,
    ):
        """Integrate a single (unbatched) sample.

        Expects ``U_initial`` of shape ``(*spatial, field_dim)`` and
        returns a tensor of shape ``(time_steps + 1, *mesh.shape, field_dim)``.
        Batching over the leading dimension is handled by
        :meth:`BaseIntegrator.__call__`.
        """
        if self.local_dx.shape[:2] != U_initial.shape[:2]:
            U_initial = U_initial[::self.mesh_sampling_rate[0], ::self.mesh_sampling_rate[1], ::self.mesh_sampling_rate[2], :]
            assert self.local_dx.shape[:2] == U_initial.shape[:2], f"Local spatial step must have shape {U_initial.shape[:2]}, got {self.local_dx.shape[:2]}"

        size_1, size_2, size_3, field_dim = U_initial.shape

        if isinstance(dirichlet_bc, Callable):
            boundary_conditions = [tuple(dirichlet_bc(bp) for bp in tup) for tup in self.mesh.get_boundary_points()]
        else:
            boundary_conditions = dirichlet_bc

        output = torch.empty((time_steps+1,size_1-2, size_2-2, size_3-2, field_dim), device = self.mesh.device)

        U_curr = U_initial[1:-1,1:-1,1:-1,:]
        output[0] = U_curr
        U_prev = torch.zeros_like(U_curr) if isinstance(U_initial, torch.Tensor) else jnp.zeros_like(U_curr) if isinstance(U_initial, jnp.ndarray) else np.zeros_like(U_curr)

        k = self.dt
        X = self.mesh.points[::self.mesh_sampling_rate[0],::self.mesh_sampling_rate[1],::self.mesh_sampling_rate[2]][1:-1,1:-1,1:-1]
        X = X.reshape(*X.shape, 1)
        H = self.local_dx[1:-1,1:-1,1:-1]
        H = H.reshape(*H.shape, 1)
        assert X.shape == H.shape, f"X and H must have the same shape, got {X.shape} and {H.shape}"

        inv_h0, inv_h1, inv_h2 = _build_inv_h(H)

        laplacian_coefficients, laplacian_gradient_coefficients, gradient_coefficients = \
            _build_spherical_coefficients(X, self.sin, self.cos)

        tensor_boundary_conditions = _prepare_boundary_conditions(
            boundary_conditions, time_steps, size_1, size_2, size_3, field_dim,
            U_curr, self.full, self.mesh, self.dtype,
        )

        gc.collect()

        if isinstance(U_curr, torch.Tensor):
            eye = torch.eye(field_dim, device=U_curr.device, dtype=U_curr.dtype).reshape(1, 1, 1, field_dim, field_dim)
        elif isinstance(U_curr, jnp.ndarray):
            eye = jnp.eye(field_dim, dtype=U_curr.dtype).reshape(1, 1, 1, field_dim, field_dim)
        else:
            eye = np.eye(field_dim, dtype=U_curr.dtype).reshape(1, 1, 1, field_dim, field_dim)

        for n in trange(time_steps, desc = "Spherical Crank-Nicolson integration"):
            V = get_jacobian_on_field(velocity_term, U_curr)
            D = get_jacobian_on_field(diffusion_term, U_curr)
            A = get_jacobian_on_field(advection_term, U_curr)
            R = get_jacobian_on_field(reaction_term, U_curr)

            next_coefficient = (1 / k**2 + EPS) * eye + V / k

            curr_centered_coefficient = (2 / k**2) * eye + V / k
            curr_centered_coefficient = curr_centered_coefficient - inv_h0 * laplacian_gradient_coefficients[0] * D
            curr_centered_coefficient = curr_centered_coefficient - 2 * inv_h0**2 * laplacian_coefficients[0] * D
            curr_centered_coefficient = curr_centered_coefficient - inv_h1 * laplacian_gradient_coefficients[1] * D
            curr_centered_coefficient = curr_centered_coefficient - 2 * inv_h1**2 * laplacian_coefficients[1] * D
            curr_centered_coefficient = curr_centered_coefficient - inv_h2 * laplacian_gradient_coefficients[2] * D
            curr_centered_coefficient = curr_centered_coefficient - 2 * inv_h2**2 * laplacian_coefficients[2] * D
            curr_centered_coefficient = curr_centered_coefficient - inv_h0 * gradient_coefficients[0] * A
            curr_centered_coefficient = curr_centered_coefficient - inv_h1 * gradient_coefficients[1] * A
            curr_centered_coefficient = curr_centered_coefficient - inv_h2 * gradient_coefficients[2] * A
            curr_centered_coefficient = curr_centered_coefficient + R

            curr_int_coefficient = inv_h0**2 * laplacian_coefficients[0] * D
            curr_ext_coefficient = inv_h0**2 * laplacian_coefficients[0] * D
            curr_ext_coefficient = curr_ext_coefficient + inv_h0 * laplacian_gradient_coefficients[0] * D
            curr_ext_coefficient = curr_ext_coefficient + inv_h0 * gradient_coefficients[0] * A
            curr_right_coefficient = inv_h1**2 * laplacian_coefficients[1] * D
            curr_right_coefficient = curr_right_coefficient + inv_h1 * laplacian_gradient_coefficients[1] * D
            curr_right_coefficient = curr_right_coefficient + inv_h1 * gradient_coefficients[1] * A
            curr_left_coefficient = inv_h1**2 * laplacian_coefficients[1] * D
            curr_up_coefficient = inv_h2**2 * laplacian_coefficients[2] * D
            curr_up_coefficient = curr_up_coefficient + inv_h2 * laplacian_gradient_coefficients[2] * D
            curr_up_coefficient = curr_up_coefficient + inv_h2 * gradient_coefficients[2] * A
            curr_down_coefficient = inv_h2**2 * laplacian_coefficients[2] * D

            if U_curr.dim() < 5:
                U_curr = U_curr.reshape(*U_curr.shape, 1)
            elif U_curr.dim() > 5:
                U_curr = U_curr.squeeze()
            
            if U_prev.dim() < 5:
                U_prev = U_prev.reshape(*U_curr.shape)
            elif U_prev.dim() > 5:
                U_prev = U_prev.squeeze()

            int_term = torch.cat([
                tensor_boundary_conditions[0][0][n],
                curr_int_coefficient[:-1,:,:] @ U_curr[:-1,:,:]
            ], dim=0)
            ext_term = torch.cat([
                curr_ext_coefficient[1:,:,:] @ U_curr[1:,:,:],
                tensor_boundary_conditions[0][1][n]
            ], dim=0)
            right_term = torch.roll(curr_right_coefficient @ U_curr, shifts=-1, dims=1)
            left_term = torch.roll(curr_left_coefficient @ U_curr, shifts=1, dims=1)
            up_term = torch.cat([
                curr_up_coefficient[:,:,:-1] @ U_curr[:,:,:-1],
                tensor_boundary_conditions[2][1][n]
            ], dim=2)
            down_term = torch.cat([
                tensor_boundary_conditions[2][0][n],
                curr_down_coefficient[:,:,1:] @ U_curr[:,:,1:]
            ], dim=2)
            prev_coefficient = -k**(-2)

            # Matrix-vector product for centered term, scalar multiply for prev term
            explicit_term = (curr_centered_coefficient @ U_curr) + prev_coefficient * U_prev \
                + int_term + ext_term + right_term + left_term + up_term + down_term

            # Solve the implicit system: next_coefficient @ U_next = explicit_term
            U_next = torch.linalg.solve(next_coefficient.expand_as(curr_centered_coefficient), explicit_term).squeeze(-1)
            U_next = _safe_clamp(U_next)

            output[n+1] = U_next
            U_prev = U_curr.squeeze(-1) if U_curr.dim() > U_next.dim() else U_curr
            U_curr = U_next

        for i, bc in enumerate(tensor_boundary_conditions):
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

        return output



class SphericalRungeKutta8(BaseIntegrator):
    """
    Runge-Kutta-Fehlberg 4(5) integrator for PDEs on spherical meshes.

    Uses the 5th-order accurate RKF45 Butcher tableau (6 stages).
    """

    def __init__(self, dt: float, mesh: SphericalMesh, mesh_sampling_rate: int | Tuple[int, int, int] = 1, dtype = None):

        super().__init__(dt, mesh, mesh_sampling_rate)

        # RKF45 Butcher tableau — 6 stages, 5th-order solution
        self.a = [
            [],
            [1/4],
            [3/32, 9/32],
            [1932/2197, -7200/2197, 7296/2197],
            [439/216, -8, 3680/513, -845/4104],
            [-8/27, 2, -3544/2565, 1859/4104, -11/40],
        ]
        self.b = [16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55]
        self.c = [0, 1/4, 3/8, 12/13, 1, 1/2]
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
    def _integrate_single(
        self,
        U_initial: torch.Tensor | jnp.ndarray | np.ndarray,
        time_steps: int,
        velocity_term: float | torch.Tensor | jnp.ndarray | np.ndarray,
        diffusion_term: float | torch.Tensor | jnp.ndarray | np.ndarray | Callable,
        advection_term: float | torch.Tensor | jnp.ndarray | np.ndarray | Callable,
        reaction_term: float | torch.Tensor | jnp.ndarray | np.ndarray | Callable,
        dirichlet_bc: List[float | torch.Tensor | jnp.ndarray | np.ndarray] | Callable,
    ):
        """Integrate a single (unbatched) sample with RKF45.

        Expects ``U_initial`` of shape ``(*spatial, field_dim)`` and
        returns a tensor of shape ``(time_steps + 1, *mesh.shape, field_dim)``.
        Batching over the leading dimension is handled by
        :meth:`BaseIntegrator.__call__`.
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

        U_curr = U_initial[1:-1,1:-1,1:-1,:]
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

        inv_h0, inv_h1, inv_h2 = _build_inv_h(H)

        laplacian_coefficients, laplacian_gradient_coefficients, gradient_coefficients = \
            _build_spherical_coefficients(X, self.sin, self.cos)

        tensor_boundary_conditions = _prepare_boundary_conditions(
            boundary_conditions, time_steps, size_1, size_2, size_3, field_dim,
            U_curr, self.full, self.mesh, self.dtype,
        )

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

        for n in trange(time_steps, desc = "Spherical Runge-Kutta integration"):
            stages = []

            for i in range(self.n_stages):
                U_stage = U_curr
                for j in range(len(self.a[i])):
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
                k_i = _safe_clamp(k_i)
                stages.append(k_i)

            U_next = U_curr
            for i in range(self.n_stages):
                if self.b[i] != 0:
                    U_next = U_next + dt * self.b[i] * stages[i]

            U_next = _safe_clamp(U_next)
            output[n+1] = U_next
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

        return output


class SphericalSpectralElement(BaseIntegrator):
    """
    Spectral Element Method (SEM) solver for PDEs on spherical meshes.
    
    Uses Legendre-Gauss-Lobatto (LGL) quadrature points and Lagrange
    interpolating polynomials as basis functions.  The spatial
    discretisation uses spectral derivative operators within elements,
    and time integration uses an implicit Crank-Nicolson scheme.
    """

    def __init__(self, dt: float, mesh: SphericalMesh, mesh_sampling_rate: int | Tuple[int, int, int] = 1, polynomial_order: int = 4):
        super().__init__(dt, mesh, mesh_sampling_rate)
        self.polynomial_order = polynomial_order
        self._setup_spectral_basis()

    # -----------------------------------------------------------------
    # Spectral basis setup
    # -----------------------------------------------------------------
    def _setup_spectral_basis(self):
        """Compute LGL nodes, weights, differentiation matrix and spectral
        filter on the reference element [-1, 1]."""
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

        N = self.polynomial_order
        nodes_np, weights_np = self._compute_lgl_nodes_and_weights(N)
        D_np = self._compute_differentiation_matrix(nodes_np)

        if self.mesh.library == "torch":
            dev = self.mesh.device
            dt_t = torch.get_default_dtype()
            self.lgl_nodes = torch.tensor(nodes_np, dtype=dt_t, device=dev)
            self.lgl_weights = torch.tensor(weights_np, dtype=dt_t, device=dev)
            self.D_ref = torch.tensor(D_np, dtype=dt_t, device=dev)
            self.D2_ref = self.D_ref @ self.D_ref
        elif self.mesh.library == "jax":
            self.lgl_nodes = jnp.array(nodes_np)
            self.lgl_weights = jnp.array(weights_np)
            self.D_ref = jnp.array(D_np)
            self.D2_ref = self.D_ref @ self.D_ref
        else:
            self.lgl_nodes = nodes_np
            self.lgl_weights = weights_np
            self.D_ref = D_np
            self.D2_ref = D_np @ D_np

    @staticmethod
    def _compute_lgl_nodes_and_weights(N):
        """Compute *N+1* Legendre-Gauss-Lobatto nodes and weights on [-1, 1]."""
        if N == 0:
            return np.array([0.0]), np.array([2.0])
        if N == 1:
            return np.array([-1.0, 1.0]), np.array([1.0, 1.0])

        x = -np.cos(np.pi * np.arange(N + 1) / N)

        for _ in range(200):
            P_prev = np.ones(N + 1)
            P_curr = x.copy()
            for m in range(2, N + 1):
                P_next = ((2 * m - 1) * x * P_curr - (m - 1) * P_prev) / m
                P_prev = P_curr
                P_curr = P_next

            denom = 1.0 - x**2
            denom[0] = 1.0
            denom[-1] = 1.0
            dP = N * (P_prev - x * P_curr) / (denom + 1e-30)
            d2P = (2 * x * dP - N * (N + 1) * P_curr) / (denom + 1e-30)

            delta = np.zeros(N + 1)
            delta[1:-1] = dP[1:-1] / (d2P[1:-1] + 1e-30)
            x -= delta
            if np.max(np.abs(delta)) < 1e-15:
                break

        x[0] = -1.0
        x[-1] = 1.0

        P_prev = np.ones(N + 1)
        P_curr = x.copy()
        for m in range(2, N + 1):
            P_next = ((2 * m - 1) * x * P_curr - (m - 1) * P_prev) / m
            P_prev = P_curr
            P_curr = P_next
        weights = 2.0 / (N * (N + 1) * P_curr**2 + 1e-30)
        return x, weights

    @staticmethod
    def _compute_differentiation_matrix(nodes):
        """Barycentric differentiation matrix on arbitrary 1-D nodes."""
        N = len(nodes)
        w = np.ones(N)
        for j in range(N):
            for k in range(N):
                if k != j:
                    w[j] *= (nodes[j] - nodes[k])
        w = 1.0 / (w + 1e-30 * np.sign(w + 1e-30))
        D = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                if i != j:
                    D[i, j] = (w[j] / (w[i] + 1e-30)) / (nodes[i] - nodes[j] + 1e-30)
                    D[i, i] -= D[i, j]
        return D

    def _compute_lagrange_basis(self, xi, nodes):
        n_nodes = nodes.shape[0]
        n_points = xi.shape[0]
        if isinstance(xi, torch.Tensor):
            basis = self.ones((n_points, n_nodes), device=xi.device, dtype=xi.dtype)
        else:
            basis = self.ones((n_points, n_nodes), dtype=getattr(xi, "dtype", np.float64))
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

    def _compute_mass_matrix(self, nodes, weights):
        n_nodes = nodes.shape[0]
        if isinstance(weights, torch.Tensor):
            return self.eye(n_nodes, device=weights.device, dtype=weights.dtype) * weights.unsqueeze(0)
        return self.eye(n_nodes, dtype=getattr(weights, "dtype", np.float64)) * weights[:, None]

    def _compute_stiffness_matrix(self, nodes, weights):
        n_nodes = nodes.shape[0]
        if isinstance(nodes, torch.Tensor):
            D_mat = self.zeros((n_nodes, n_nodes), device=nodes.device, dtype=nodes.dtype)
        else:
            D_mat = self.zeros((n_nodes, n_nodes), dtype=getattr(nodes, "dtype", np.float64))
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i == j:
                    s = 0.0
                    for k in range(n_nodes):
                        if k != i:
                            s += 1.0 / (nodes[i] - nodes[k] + EPS)
                    D_mat[i, j] = s
                else:
                    num = 1.0
                    denom = 1.0
                    for k in range(n_nodes):
                        if k != i and k != j:
                            num *= (nodes[i] - nodes[k])
                            denom *= (nodes[j] - nodes[k])
                    D_mat[i, j] = num / (denom * (nodes[i] - nodes[j]) + EPS)
        M = self._compute_mass_matrix(nodes, weights)
        return self.matmul(M, D_mat)

    # -----------------------------------------------------------------
    # Spectral derivative helpers
    # -----------------------------------------------------------------
    def _apply_spectral_deriv(self, U, dim, order=1):
        """Apply the element-wise spectral derivative along *dim*.

        The interior grid along *dim* is partitioned into elements of
        ``polynomial_order`` sub-intervals (``polynomial_order + 1``
        nodes).  Inside each element the pre-computed LGL
        differentiation matrix is used; scaling is applied based on the
        element length in physical space.  This keeps the operation
        O(N · p²) per direction.
        """
        p = self.polynomial_order
        N = U.shape[dim]
        D_ref = self.D_ref if order == 1 else self.D2_ref

        # Move target dim to position 0 for easy slicing
        perm = list(range(U.ndim))
        perm[0], perm[dim] = perm[dim], perm[0]
        Ut = U.permute(*perm) if isinstance(U, torch.Tensor) else np.transpose(U, perm)
        flat_shape = (N, -1)
        trailing = Ut.shape[1:]
        if isinstance(Ut, torch.Tensor):
            Uf = Ut.reshape(flat_shape[0], int(np.prod(trailing)))
            result = torch.zeros_like(Uf)
        else:
            Uf = Ut.reshape(flat_shape[0], int(np.prod(trailing)))
            result = np.zeros_like(Uf)

        elem_size = p + 1
        starts = list(range(0, N - p, p))
        if starts and starts[-1] + elem_size < N:
            starts.append(N - elem_size)

        weight = torch.zeros(N, 1, device=Uf.device, dtype=Uf.dtype) if isinstance(Uf, torch.Tensor) else np.zeros((N, 1))

        for s in starts:
            e = min(s + elem_size, N)
            actual = e - s
            scale = 2.0 / max(float(actual - 1), 1.0)
            D_loc = D_ref[:actual, :actual] * (scale ** order)
            if isinstance(Uf, torch.Tensor):
                chunk = Uf[s:e]
                result[s:e] += D_loc @ chunk
            else:
                chunk = Uf[s:e]
                result[s:e] += D_loc @ chunk
            weight[s:e] += 1.0

        weight = weight.clamp(min=1.0) if isinstance(weight, torch.Tensor) else np.clip(weight, 1.0, None)
        result = result / weight

        if isinstance(result, torch.Tensor):
            result = result.reshape(Ut.shape)
            inv_perm = [0] * len(perm)
            for i, p_i in enumerate(perm):
                inv_perm[p_i] = i
            result = result.permute(*inv_perm)
        else:
            result = result.reshape(Ut.shape)
            inv_perm = [0] * len(perm)
            for i, p_i in enumerate(perm):
                inv_perm[p_i] = i
            result = np.transpose(result, inv_perm)
        return result

    def _spectral_filter(self, U, strength=36.0, cutoff_frac=0.67):
        """Apply exponential spectral filter along all spatial dims to damp
        unresolved high-frequency modes (σ-filtering)."""
        for dim in range(min(3, U.ndim - 1)):
            N = U.shape[dim]
            if N < 4:
                continue
            cutoff = int(N * cutoff_frac)
            if isinstance(U, torch.Tensor):
                k = torch.arange(N, device=U.device, dtype=U.dtype)
                sigma = torch.ones(N, device=U.device, dtype=U.dtype)
                mask = k > cutoff
                eta = (k[mask] - cutoff).float() / max(N // 2 - cutoff, 1)
                sigma[mask] = torch.exp(-strength * eta ** strength)
                shape = [1] * U.ndim
                shape[dim] = N
                sigma = sigma.reshape(shape)
                U_hat = torch.fft.fft(U.float(), dim=dim)
                U = torch.fft.ifft(U_hat * sigma, dim=dim).real.to(U.dtype)
            else:
                k = np.arange(N)
                sigma = np.ones(N)
                mask = k > cutoff
                eta = (k[mask] - cutoff).astype(float) / max(N // 2 - cutoff, 1)
                sigma[mask] = np.exp(-strength * eta ** strength)
                shape = [1] * U.ndim
                shape[dim] = N
                sigma = sigma.reshape(shape)
                U_hat = np.fft.fft(U, axis=dim)
                U = np.fft.ifft(U_hat * sigma, axis=dim).real
        return U

    # -----------------------------------------------------------------
    # Main solver
    # -----------------------------------------------------------------
    @torch.no_grad()
    def _integrate_single(
        self,
        U_initial: torch.Tensor | jnp.ndarray | np.ndarray,
        time_steps: int,
        velocity_term: float | torch.Tensor | jnp.ndarray | np.ndarray,
        diffusion_term: float | torch.Tensor | jnp.ndarray | np.ndarray | Callable,
        advection_term: float | torch.Tensor | jnp.ndarray | np.ndarray | Callable,
        reaction_term: float | torch.Tensor | jnp.ndarray | np.ndarray | Callable,
        dirichlet_bc: List[float | torch.Tensor | jnp.ndarray | np.ndarray] | Callable,
    ):
        """Solve PDE using Spectral Element spatial discretisation with
        Crank-Nicolson time integration on a single (unbatched) sample.

        Expects ``U_initial`` of shape ``(*spatial, field_dim)`` and
        returns a tensor of shape ``(time_steps + 1, *mesh.shape, field_dim)``.
        Batching over the leading dimension is handled by
        :meth:`BaseIntegrator.__call__`.
        """
        if self.local_dx.shape[:2] != U_initial.shape[:2]:
            U_initial = U_initial[::self.mesh_sampling_rate[0], ::self.mesh_sampling_rate[1], ::self.mesh_sampling_rate[2], :]
            assert self.local_dx.shape[:2] == U_initial.shape[:2]

        size_1, size_2, size_3, field_dim = U_initial.shape

        if isinstance(dirichlet_bc, Callable):
            boundary_conditions = [tuple(dirichlet_bc(bp) for bp in tup) for tup in self.mesh.get_boundary_points()]
        else:
            boundary_conditions = dirichlet_bc

        output = torch.empty((time_steps+1, size_1-2, size_2-2, size_3-2, field_dim), device=self.mesh.device)

        U_curr = U_initial[1:-1,1:-1,1:-1,:]
        output[0] = U_curr
        U_prev = torch.zeros_like(U_curr) if isinstance(U_initial, torch.Tensor) else jnp.zeros_like(U_curr) if isinstance(U_initial, jnp.ndarray) else np.zeros_like(U_curr)

        k = self.dt
        X = self.mesh.points[::self.mesh_sampling_rate[0],::self.mesh_sampling_rate[1],::self.mesh_sampling_rate[2]][1:-1,1:-1,1:-1]
        X = X.reshape(*X.shape, 1)
        H = self.local_dx[1:-1,1:-1,1:-1]
        H = H.reshape(*H.shape, 1)
        assert X.shape == H.shape

        inv_h0, inv_h1, inv_h2 = _build_inv_h(H)

        laplacian_coefficients, laplacian_gradient_coefficients, gradient_coefficients = \
            _build_spherical_coefficients(X, self.sin, self.cos)

        tensor_boundary_conditions = _prepare_boundary_conditions(
            boundary_conditions, time_steps, size_1, size_2, size_3, field_dim,
            U_curr, self.full, self.mesh, self.dtype,
        )

        gc.collect()

        if isinstance(U_curr, torch.Tensor):
            eye = torch.eye(field_dim, device=U_curr.device, dtype=U_curr.dtype).reshape(1, 1, 1, field_dim, field_dim)
        elif isinstance(U_curr, jnp.ndarray):
            eye = jnp.eye(field_dim, dtype=U_curr.dtype).reshape(1, 1, 1, field_dim, field_dim)
        else:
            eye = np.eye(field_dim, dtype=U_curr.dtype).reshape(1, 1, 1, field_dim, field_dim)

        for n in trange(time_steps, desc = "Spherical Spectral Element integration"):
            V = get_jacobian_on_field(velocity_term, U_curr)
            D_coeff = get_jacobian_on_field(diffusion_term, U_curr)
            A_coeff = get_jacobian_on_field(advection_term, U_curr)
            R_coeff = get_jacobian_on_field(reaction_term, U_curr)

            # ---- Spectral spatial operators ----
            # Second derivatives via spectral element derivative
            d2U_dr   = self._apply_spectral_deriv(U_curr, dim=0, order=2)
            d2U_dlon = self._apply_spectral_deriv(U_curr, dim=1, order=2)
            d2U_dlat = self._apply_spectral_deriv(U_curr, dim=2, order=2)
            # First derivatives via spectral element derivative
            dU_dr   = self._apply_spectral_deriv(U_curr, dim=0, order=1)
            dU_dlon = self._apply_spectral_deriv(U_curr, dim=1, order=1)
            dU_dlat = self._apply_spectral_deriv(U_curr, dim=2, order=1)

            # Expand for matrix multiply
            def _to_mat(t):
                return t.unsqueeze(-1) if isinstance(t, torch.Tensor) else t[..., np.newaxis]

            # Diffusion: D · (laplacian)
            lap_term = (
                laplacian_coefficients[0] * (D_coeff @ _to_mat(d2U_dr))
                + laplacian_gradient_coefficients[0] * (D_coeff @ _to_mat(dU_dr))
                + laplacian_coefficients[1] * (D_coeff @ _to_mat(d2U_dlon))
                # dim1 (lon) has no gradient correction (laplacian_gradient_coefficients[1] == 0)
                + laplacian_coefficients[2] * (D_coeff @ _to_mat(d2U_dlat))
                + laplacian_gradient_coefficients[2] * (D_coeff @ _to_mat(dU_dlat))
            )
            # Advection: A · (gradient)
            adv_term = (
                gradient_coefficients[0] * (A_coeff @ _to_mat(dU_dr))
                + gradient_coefficients[1] * (A_coeff @ _to_mat(dU_dlon))
                + gradient_coefficients[2] * (A_coeff @ _to_mat(dU_dlat))
            )
            # Reaction
            react_term = R_coeff @ _to_mat(U_curr)

            # Assemble RHS  (spatial operator applied to U_curr)
            spatial_rhs = lap_term + adv_term + react_term  # shape (..., fd, 1)

            # ---- Crank-Nicolson time stepping ----
            # (1/k² + V/k) U^{n+1} = (2/k²) U^n + (V/k) U^n - (1/k²) U^{n-1} + L[U^n]
            next_coefficient = (1 / k**2 + EPS) * eye + V / k

            U_curr_mat = _to_mat(U_curr)
            U_prev_mat = _to_mat(U_prev)

            rhs = (2 / k**2) * U_curr_mat + (V / k) @ U_curr_mat - (1 / k**2) * U_prev_mat + spatial_rhs

            U_next = torch.linalg.solve(
                next_coefficient.expand(*U_curr.shape[:-1], field_dim, field_dim),
                rhs.squeeze(-1),
            )

            # Spectral filter to damp unresolved modes
            U_next = self._spectral_filter(U_next)
            U_next = _safe_clamp(U_next)

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

        return output


class Parareal(BaseIntegrator):

    def __init__(self, coarse_integrator: Callable, fine_integrator: Callable | None = None, delta_estimator: Callable | None = None, dt: float = 0.1, mesh: GenericMesh | None = None, mesh_sampling_rate: int | Tuple[int, int, int] = 1, max_U_norm: float = 1.0):
        """
        Initialize Parareal algorithm.
        
        Args:
            coarse_integrator: Coarse integrator function
            fine_integrator: Fine integrator function
            delta_estimator: Delta estimator function
            dt: Time step
            mesh: Mesh
            mesh_sampling_rate: Mesh sampling rate
            max_U_norm: Maximum norm of the field
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

    def _check_integration(self, U: torch.Tensor | jnp.ndarray | np.ndarray, time_steps: int, velocity_term: float | torch.Tensor | jnp.ndarray | np.ndarray, diffusion_term: float | torch.Tensor | jnp.ndarray | np.ndarray | Callable, advection_term: float | torch.Tensor | jnp.ndarray | np.ndarray | Callable, reaction_term: float |torch.Tensor | jnp.ndarray | np.ndarray | Callable, dirichlet_bc: List[float | torch.Tensor | jnp.ndarray | np.ndarray] | Callable):

        if self.fine_integrator is None:
            raise ValueError("Fine integrator must be set for integration check")

        F = self.fine_integrator(U, time_steps, velocity_term, diffusion_term, advection_term, reaction_term, dirichlet_bc)
        f_hat = self.__call__(U, time_steps, velocity_term, diffusion_term, advection_term, reaction_term, dirichlet_bc)
        diff = torch.norm(F - f_hat, p = 2, dim = -1)

        return (diff.min().item(), diff.mean().item(), diff.max().item())

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
        U_initial, _n_dims_added = self._normalize_field(U_initial)
        # U_initial.shape = (batch_size, time_steps+1, radial, longitude, latitude, field_dim)

        coarse_output = self.coarse_integrator(U_initial, time_steps, velocity_term, diffusion_term, advection_term, reaction_term, dirichlet_bc)
        # coarse_output.shape = (batch_size, time_steps+1, radial, longitude, latitude, field_dim)

        k = 0
        error = torch.inf if isinstance(coarse_output, torch.Tensor) else float("inf")
        previous_output = coarse_output[:,:-1]

        while error > tol and k < max_iter:
            print(f"Parareal Iteration {k}, error: {error}")
            output_delta = self.delta_estimator(previous_output)
            # output_delta.shape = (batch_size, time_steps+1, radial, longitude, latitude, field_dim)
            if isinstance(coarse_output, torch.Tensor):
                output_delta = torch.cat([
                    torch.zeros_like(previous_output[:,0:1]),
                    output_delta.permute(0,1,3,4,5,2)
                ], dim=1)
            elif isinstance(coarse_output, jnp.ndarray):
                output_delta = jnp.concatenate([
                    jnp.zeros_like(previous_output[:,0:1]),
                    output_delta.permute(0,1,3,4,5,2)
                ], axis=1)
            else:
                output_delta = np.concatenate([
                    np.zeros_like(previous_output[:,0:1]),
                    output_delta.permute(0,1,3,4,5,2)
                ], axis=1)
            
            fine_output = coarse_output + output_delta
            fine_output_single_steps = fine_output.flatten(end_dim = 1).unsqueeze(1)
            refined_output = self.coarse_integrator(fine_output_single_steps, 1, velocity_term, diffusion_term, advection_term, reaction_term, dirichlet_bc)
            refined_output = refined_output.unflatten(0, (-1, time_steps+1))

            # Track convergence (compare against previous iteration's
            # solution, dropping the final time step so the shapes match
            # ``previous_output``)
            if isinstance(fine_output, torch.Tensor):
                error = torch.norm(fine_output[:, :-1] - previous_output, p=2).item()
            elif isinstance(fine_output, jnp.ndarray):
                error = float(jnp.linalg.norm(fine_output[:, :-1] - previous_output))
            else:
                error = float(np.linalg.norm(fine_output[:, :-1] - previous_output))

            previous_output = fine_output[:, :-1]
            k += 1

        fine_output = self._denormalize_field(fine_output, _n_dims_added)
        return fine_output
