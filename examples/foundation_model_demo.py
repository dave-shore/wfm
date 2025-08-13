#!/usr/bin/env python3
"""
Foundation Model Demo

This script demonstrates the usage of all three core components of the WFM package:
1. Spherical mesh generation
2. Differential equation solving
3. Multi-modal input processing with enhanced features
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add the src directory to the path to import wfm
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wfm import (
    SphericalMesh, SphericalMeshBuilder,
    DiffusionAdvectionReaction, SphericalDiscretizer,
    InputProcessor, ImageProcessor, AudioProcessor, VideoProcessor
)


def demo_spherical_mesh():
    """Demonstrate spherical mesh generation."""
    print("=== Spherical Mesh Demo ===")
    
    # Create different types of meshes
    uniform_mesh = SphericalMeshBuilder.create_uniform_mesh(
        radius=1.0, n_lat=20, n_lon=40
    )
    print(f"Uniform mesh: {len(uniform_mesh.points)} points")
    
    adaptive_mesh = SphericalMeshBuilder.create_adaptive_mesh(
        radius=1.0, base_n_lat=15, base_n_lon=30, refinement_levels=2
    )
    print(f"Adaptive mesh: {len(adaptive_mesh.points)} points")
    
    # Test coordinate transformations
    test_point = torch.tensor([1.0, 0.0, 0.0])
    spherical_coords = uniform_mesh.transform_coordinates(test_point)
    print(f"Point {test_point} in spherical coordinates: {spherical_coords}")
    
    return uniform_mesh, adaptive_mesh


def demo_differential_equations():
    """Demonstrate differential equation solving."""
    print("\n=== Differential Equations Demo ===")
    
    # Create a simple diffusion-advection-reaction equation
    dar = DiffusionAdvectionReaction(
        diffusion_coeff=0.1,
        velocity_field=lambda x, y, z: torch.stack([x, y, z], dim=-1),
        reaction_function=lambda u: -0.5 * u
    )
    
    # Create a mesh for discretization
    mesh = SphericalMeshBuilder.create_uniform_mesh(radius=1.0, n_lat=10, n_lon=20)
    
    # Create discretizer
    discretizer = SphericalDiscretizer(mesh)
    
    # Set initial conditions
    initial_conditions = torch.ones(len(mesh.points)) * 0.5
    
    # Solve the equation
    solution = discretizer.discretize_equation(
        dar, initial_conditions, time_steps=10, dt=0.01
    )
    
    print(f"Solved equation over {len(mesh.points)} mesh points")
    print(f"Solution shape: {solution.shape}")
    
    return dar, discretizer, solution


def demo_input_processing():
    """Demonstrate enhanced input processing capabilities."""
    print("\n=== Enhanced Input Processing Demo ===")
    
    # Initialize processor
    processor = InputProcessor(target_shape=(12, 12, 3))
    
    # Test image processing with focal/peripheral distinction
    print("Testing image processing...")
    test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    image_tensor = processor.process_input(test_image, 'image')
    print(f"Image tensor shape: {image_tensor.shape}")
    print(f"Data type column shape: {image_tensor[0, :, 0, :].shape}")
    
    # Test audio processing with MFCC features
    print("Testing audio processing...")
    # Create synthetic audio data
    sample_rate = 22050
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    synthetic_audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    audio_tensor = processor.process_input(synthetic_audio, 'audio')
    print(f"Audio tensor shape: {audio_tensor.shape}")
    
    # Test video processing
    print("Testing video processing...")
    # Create synthetic video frames
    video_frames = np.random.randint(0, 255, (5, 100, 100, 3), dtype=np.uint8)
    video_tensor = processor.process_input(video_frames, 'video')
    print(f"Video tensor shape: {video_tensor.shape}")
    
    # Test text processing
    print("Testing text processing...")
    text_tensor = processor.process_input("Hello, World!", 'text')
    print(f"Text tensor shape: {text_tensor.shape}")
    
    # Test numerical processing
    print("Testing numerical processing...")
    numerical_tensor = processor.process_input([1.0, 2.0, 3.0, 4.0], 'numerical')
    print(f"Numerical tensor shape: {numerical_tensor.shape}")
    
    return processor, image_tensor, audio_tensor, video_tensor


def demo_integration():
    """Demonstrate integration of all components."""
    print("\n=== Integration Demo ===")
    
    # Create mesh
    mesh = SphericalMeshBuilder.create_uniform_mesh(radius=1.0, n_lat=15, n_lon=30)
    
    # Create input processor
    processor = InputProcessor(target_shape=(12, 12, 3))
    
    # Process mixed input
    mixed_input = [
        np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8),  # Image
        "Sample text input",  # Text
        [1.0, 2.0, 3.0, 4.0, 5.0],  # Numerical
        np.sin(np.linspace(0, 2*np.pi, 1000))  # Audio
    ]
    
    mixed_tensor = processor.process_input(mixed_input, 'mixed')
    print(f"Mixed input tensor shape: {mixed_tensor.shape}")
    
    # Create a simple differential equation
    dar = DiffusionAdvectionReaction(
        diffusion_coeff=0.05,
        velocity_field=lambda x, y, z: torch.zeros_like(torch.stack([x, y, z], dim=-1)),
        reaction_function=lambda u: -0.1 * u
    )
    
    # Solve on the mesh
    discretizer = SphericalDiscretizer(mesh)
    initial_conditions = torch.ones(len(mesh.points))
    solution = discretizer.discretize_equation(
        dar, initial_conditions, time_steps=5, dt=0.02
    )
    
    print(f"Integrated solution shape: {solution.shape}")
    
    return mesh, processor, mixed_tensor, solution


def visualize_results(mesh, image_tensor, audio_tensor, video_tensor, solution):
    """Visualize the results of the demonstrations."""
    print("\n=== Visualization ===")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('WFM Foundation Model Demo Results', fontsize=16)
    
    # Plot mesh points
    points = mesh.points.cpu().numpy()
    axes[0, 0].scatter(points[:, 0], points[:, 1], s=1, alpha=0.6)
    axes[0, 0].set_title('Spherical Mesh (XY projection)')
    axes[0, 0].set_aspect('equal')
    
    # Plot image tensor (first frame, without data type column)
    img_data = image_tensor[0, :, 1:, :].cpu().numpy()  # Skip first column
    axes[0, 1].imshow(img_data)
    axes[0, 1].set_title('Processed Image (12x12x3)')
    
    # Plot audio tensor (first frame, without data type column)
    audio_data = audio_tensor[0, :, 1:, :].cpu().numpy()  # Skip first column
    axes[0, 2].imshow(audio_data)
    axes[0, 2].set_title('Audio MFCC Features (12x12x3)')
    
    # Plot video tensor (first frame, without data type column)
    video_data = video_tensor[0, :, 1:, :].cpu().numpy()  # Skip first column
    axes[1, 0].imshow(video_data)
    axes[1, 0].set_title('Video Frame (12x12x3)')
    
    # Plot solution evolution
    solution_np = solution.cpu().numpy()
    axes[1, 1].plot(solution_np.T)
    axes[1, 1].set_title('Solution Evolution')
    axes[1, 1].set_xlabel('Time Step')
    axes[1, 1].set_ylabel('Value')
    
    # Plot data type column for image
    data_type_col = image_tensor[0, :, 0, :].cpu().numpy()
    axes[1, 2].imshow(data_type_col, cmap='viridis')
    axes[1, 2].set_title('Data Type Column (One-hot encoded)')
    
    plt.tight_layout()
    plt.show()


def main():
    """Run all demonstrations."""
    print("Wave Foundation Model (WFM) Demo")
    print("=" * 50)
    
    try:
        # Run demonstrations
        mesh, adaptive_mesh = demo_spherical_mesh()
        dar, discretizer, solution = demo_differential_equations()
        processor, image_tensor, audio_tensor, video_tensor = demo_input_processing()
        integrated_mesh, integrated_processor, mixed_tensor, integrated_solution = demo_integration()
        
        # Visualize results
        visualize_results(mesh, image_tensor, audio_tensor, video_tensor, solution)
        
        print("\n=== Demo Completed Successfully! ===")
        print("All components are working correctly.")
        
    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
