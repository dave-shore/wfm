"""
Wave Foundation Model - A PyTorch-based foundation model with spherical mesh, 
differential equations, and enhanced multi-modal input processing.

This package provides:
- Spherical mesh generation and coordinate transformations
- Diffusion-advection-reaction differential equation solvers
- Advanced multi-modal input processing with binary tokenization
- Peripheral view summarization for tabular and graph data
"""

__version__ = "0.2.0"
__author__ = "Wave Foundation Model Team"

__all__ = [
    "base",
    "functional",
    "input_processor",
    "pde",
    "spherical_mesh"
]
