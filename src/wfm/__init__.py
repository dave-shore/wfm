"""
Wave Foundation Model (WFM) Package

A PyTorch-based foundation model with spherical mesh, differential equations, and multi-modal input processing.
"""

__version__ = "0.1.0"
__author__ = "Davide Riva"

# Core components
from .spherical_mesh import SphericalMesh, SphericalMeshBuilder
from .diffusion_advection_reaction import (
    DiffusionAdvectionReaction, 
    SphericalDiscretizer,
    ReactionFunctions,
    VelocityFields
)
from .input_processor import (
    InputProcessor,
    ImageProcessor,
    TextProcessor,
    NumericalProcessor,
    AudioProcessor,
    VideoProcessor,
    GraphProcessor,
    BatchProcessor,
    InputValidator
)

__all__ = [
    # Spherical mesh
    "SphericalMesh",
    "SphericalMeshBuilder",
    
    # Differential equations
    "DiffusionAdvectionReaction",
    "SphericalDiscretizer", 
    "ReactionFunctions",
    "VelocityFields",
    
    # Input processing
    "InputProcessor",
    "ImageProcessor",
    "TextProcessor",
    "NumericalProcessor",
    "AudioProcessor",
    "VideoProcessor",
    "GraphProcessor",
    "BatchProcessor",
    "InputValidator",
]
