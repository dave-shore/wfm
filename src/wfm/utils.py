import numpy as np
import torch
from typing import Iterable

def batch_generator(X: Iterable, batch_size: int):
    """
    Generate batches of data from an iterable.
    """
    for i in range(0, len(X), batch_size):
        yield X[i:i+batch_size]

def ordinal_position_in_axis(matrix, axis=1, maximize = False):
    """
    Count using argsort - more memory efficient for large arrays.
    """
    # Get sorted indices for each row
    sorted_indices = np.argsort(matrix, axis=axis)
    if maximize:
        sorted_indices = sorted_indices[::-1]
        
    # For each element, find its position in sorted array
    # The count of larger elements = (row_length - position - 1)
    result = np.zeros_like(matrix, dtype=int)
    
    for i in range(matrix.shape[axis]):
        # Find the position of each original element in sorted array
        reverse_indices = np.argsort(sorted_indices[i], axis=axis)
        result[i] = reverse_indices + 1
    
    return result


class LazyTensor_np:
    """Wrapper for numpy arrays to provide LazyTensor-like interface"""
    
    def __init__(self, data):
        self.data = np.asarray(data)
    
    def sum(self, axis=None, keepdims=False):
        """Sum along specified axis, returns numpy array"""
        return np.sum(self.data, axis=axis, keepdims=keepdims)
    
    def argKmin(self, K, axis=-1):
        """Return indices of K smallest values along axis, returns numpy array"""
        if axis == -1:
            axis = len(self.data.shape) - 1
        # Get K smallest values and their indices
        sorted_indices = np.argsort(self.data, axis=axis)
        if K >= self.data.shape[axis]:
            return sorted_indices
        else:
            # Take first K indices
            if axis == 0:
                return sorted_indices[:K]
            elif axis == 1:
                return sorted_indices[:, :K]
            else:
                # For higher dimensions, use advanced indexing
                slices = [slice(None)] * len(self.data.shape)
                slices[axis] = slice(K)
                return sorted_indices[tuple(slices)]
    
    def argmin(self, axis=None):
        """Return indices of minimum values, returns numpy array"""
        return np.argmin(self.data, axis=axis)
    
    def argKmax(self, K, axis=-1):
        """Return indices of K largest values along axis, returns numpy array"""
        if axis == -1:
            axis = len(self.data.shape) - 1
        # Get K largest values and their indices
        sorted_indices = np.argsort(self.data, axis=axis)
        if K >= self.data.shape[axis]:
            return sorted_indices
        else:
            # Take last K indices (largest values)
            if axis == 0:
                return sorted_indices[-K:]
            elif axis == 1:
                return sorted_indices[:, -K:]
            else:
                # For higher dimensions, use advanced indexing
                slices = [slice(None)] * len(self.data.shape)
                slices[axis] = slice(-K, None)
                return sorted_indices[tuple(slices)]
    
    def argmax(self, axis=None):
        """Return indices of maximum values, returns numpy array"""
        return np.argmax(self.data, axis=axis)
    
    def __getitem__(self, key):
        """Support indexing operations"""
        return LazyTensor_np(self.data[key])
    
    def __setitem__(self, key, value):
        """Support assignment operations"""
        self.data[key] = value
    
    def __matmul__(self, other):
        """Support matrix multiplication operations"""
        return LazyTensor_np(self.data @ other.data)
    
    def __mul__(self, other):
        """Support multiplication operations"""
        if isinstance(other, LazyTensor_np):
            return LazyTensor_np(self.data * other.data)
        else:
            return LazyTensor_np(self.data * other)
    
    def __rmul__(self, other):
        """Support right multiplication operations"""
        return LazyTensor_np(other * self.data)
    
    def __add__(self, other):
        """Support addition operations"""
        if isinstance(other, LazyTensor_np):
            return LazyTensor_np(self.data + other.data)
        else:
            return LazyTensor_np(self.data + other)
    
    def __radd__(self, other):
        """Support right addition operations"""
        return LazyTensor_np(other + self.data)
    
    def __sub__(self, other):
        """Support subtraction operations"""
        if isinstance(other, LazyTensor_np):
            return LazyTensor_np(self.data - other.data)
        else:
            return LazyTensor_np(self.data - other)
    
    def __rsub__(self, other):
        """Support right subtraction operations"""
        return LazyTensor_np(other - self.data)
    
    def __pow__(self, other):
        """Support power operations"""
        return LazyTensor_np(self.data ** other)
    
    def __truediv__(self, other):
        """Support division operations"""
        if isinstance(other, LazyTensor_np):
            return LazyTensor_np(self.data / other.data)
        else:
            return LazyTensor_np(self.data / other)
    
    def __rtruediv__(self, other):
        """Support right division operations"""
        return LazyTensor_np(other / self.data)
    
    @property
    def shape(self):
        """Return shape of the underlying array"""
        return self.data.shape
    
    @property
    def dtype(self):
        """Return dtype of the underlying array"""
        return self.data.dtype

    @property
    def T(self):
        """Return transpose of the underlying array"""
        return LazyTensor_np(self.data.T)


class LazyTensor_torch:
    """Wrapper for torch tensors to provide LazyTensor-like interface"""
    
    def __init__(self, data):
        self.data = torch.as_tensor(data)
    
    def sum(self, axis=None, keepdims=False):
        """Sum along specified axis, returns numpy array"""
        result = torch.sum(self.data, dim=axis, keepdim=keepdims)
        return result.detach().cpu().numpy()
    
    def argKmin(self, K, axis=-1):
        """Return indices of K smallest values along axis, returns numpy array"""
        if axis == -1:
            axis = len(self.data.shape) - 1
        # Get K smallest values and their indices
        sorted_indices = torch.argsort(self.data, dim=axis)
        if K >= self.data.shape[axis]:
            return sorted_indices.detach().cpu().numpy()
        else:
            # Take first K indices
            if axis == 0:
                return sorted_indices[:K].detach().cpu().numpy()
            elif axis == 1:
                return sorted_indices[:, :K].detach().cpu().numpy()
            else:
                # For higher dimensions, use advanced indexing
                slices = [slice(None)] * len(self.data.shape)
                slices[axis] = slice(K)
                return sorted_indices[tuple(slices)].detach().cpu().numpy()
    
    def argmin(self, axis=None):
        """Return indices of minimum values, returns numpy array"""
        result = torch.argmin(self.data, dim=axis)
        return result.detach().cpu().numpy()
    
    def argKmax(self, K, axis=-1):
        """Return indices of K largest values along axis, returns numpy array"""
        if axis == -1:
            axis = len(self.data.shape) - 1
        # Get K largest values and their indices
        sorted_indices = torch.argsort(self.data, dim=axis)
        if K >= self.data.shape[axis]:
            return sorted_indices.detach().cpu().numpy()
        else:
            # Take last K indices (largest values)
            if axis == 0:
                return sorted_indices[-K:].detach().cpu().numpy()
            elif axis == 1:
                return sorted_indices[:, -K:].detach().cpu().numpy()
            else:
                # For higher dimensions, use advanced indexing
                slices = [slice(None)] * len(self.data.shape)
                slices[axis] = slice(-K, None)
                return sorted_indices[tuple(slices)].detach().cpu().numpy()
    
    def argmax(self, axis=None):
        """Return indices of maximum values, returns numpy array"""
        result = torch.argmax(self.data, dim=axis)
        return result.detach().cpu().numpy()
    
    def __getitem__(self, key):
        """Support indexing operations"""
        return LazyTensor_torch(self.data[key])
    
    def __setitem__(self, key, value):
        """Support assignment operations"""
        self.data[key] = value
    
    def __mul__(self, other):
        """Support multiplication operations"""
        if isinstance(other, LazyTensor_torch):
            return LazyTensor_torch(self.data * other.data)
        else:
            return LazyTensor_torch(self.data * other)

    def __matmul__(self, other):
        """Support matrix multiplication operations"""
        return LazyTensor_torch(self.data @ other.data)
    
    def __rmul__(self, other):
        """Support right multiplication operations"""
        return LazyTensor_torch(other * self.data)
    
    def __add__(self, other):
        """Support addition operations"""
        if isinstance(other, LazyTensor_torch):
            return LazyTensor_torch(self.data + other.data)
        else:
            return LazyTensor_torch(self.data + other)
    
    def __radd__(self, other):
        """Support right addition operations"""
        return LazyTensor_torch(other + self.data)
    
    def __sub__(self, other):
        """Support subtraction operations"""
        if isinstance(other, LazyTensor_torch):
            return LazyTensor_torch(self.data - other.data)
        else:
            return LazyTensor_torch(self.data - other)
    
    def __rsub__(self, other):
        """Support right subtraction operations"""
        return LazyTensor_torch(other - self.data)
    
    def __pow__(self, other):
        """Support power operations"""
        return LazyTensor_torch(self.data ** other)
    
    def __truediv__(self, other):
        """Support division operations"""
        if isinstance(other, LazyTensor_torch):
            return LazyTensor_torch(self.data / other.data)
        else:
            return LazyTensor_torch(self.data / other)
    
    def __rtruediv__(self, other):
        """Support right division operations"""
        return LazyTensor_torch(other / self.data)
    
    @property
    def shape(self):
        """Return shape of the underlying tensor"""
        return self.data.shape
    
    @property
    def dtype(self):
        """Return dtype of the underlying tensor"""
        return self.data.dtype

    @property
    def T(self):
        """Return transpose of the underlying tensor"""
        return LazyTensor_torch(self.data.T)
