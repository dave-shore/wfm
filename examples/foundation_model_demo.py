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
import pandas as pd
import networkx as nx

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
    
    # Test text processing with binary tokenization
    print("\n--- Text Processing with Binary Tokenization ---")
    sample_text = "The quick brown fox jumps over the lazy dog. This is a sample text for demonstration."
    text_tensor = processor.process_input(sample_text)
    print(f"Text tensor shape: {text_tensor.shape}")
    print(f"Data type column (first 3 rows): {text_tensor[0, :3, 0, 0]}")
    
    # Test numerical processing
    print("\n--- Numerical Processing ---")
    sample_numbers = np.random.randn(50)
    numerical_tensor = processor.process_input(sample_numbers)
    print(f"Numerical tensor shape: {numerical_tensor.shape}")
    print(f"Data type column (first 3 rows): {numerical_tensor[0, :3, 0, 0]}")
    
    # Test tabular processing
    print("\n--- Tabular Processing with Peripheral Views ---")
    sample_data = {
        'feature1': [1.2, 3.4, 5.6, 7.8],
        'feature2': [2.1, 4.3, 6.5, 8.7],
        'feature3': [0.9, 1.8, 2.7, 3.6]
    }
    df = pd.DataFrame(sample_data)
    tabular_tensor = processor.process_input(df)
    print(f"Tabular tensor shape: {tabular_tensor.shape}")
    print(f"Data type column (first 3 rows): {tabular_tensor[0, :3, 0, 0]}")
    
    # Test graph processing
    print("\n--- Graph Processing with Neighbor Summaries ---")
    G = nx.Graph()
    G.add_nodes_from([0, 1, 2, 3])
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (0, 3)])
    
    # Add some node attributes
    nx.set_node_attributes(G, {0: {'weight': 1.0}, 1: {'weight': 2.0}, 2: {'weight': 1.5}, 3: {'weight': 0.8}})
    
    graph_tensor = processor.process_input(G)
    print(f"Graph tensor shape: {graph_tensor.shape}")
    print(f"Data type column (first 3 rows): {graph_tensor[0, :3, 0, 0]}")
    
    # Test image processing
    print("\n--- Image Processing ---")
    # Create a synthetic image
    synthetic_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    image_tensor = processor.process_input(synthetic_image)
    print(f"Image tensor shape: {image_tensor.shape}")
    print(f"Data type column (first 3 rows): {image_tensor[0, :3, 0, 0]}")
    
    # Test audio processing
    print("\n--- Audio Processing ---")
    # Create synthetic audio data
    synthetic_audio = np.random.randn(16000)  # 1 second at 16kHz
    audio_tensor = processor.process_input(synthetic_audio)
    print(f"Audio tensor shape: {audio_tensor.shape}")
    print(f"Data type column (first 3 rows): {audio_tensor[0, :3, 0, 0]}")
    
    return {
        'text': text_tensor,
        'numerical': numerical_tensor,
        'tabular': tabular_tensor,
        'graph': graph_tensor,
        'image': image_tensor,
        'audio': audio_tensor
    }


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


def visualize_results(mesh, dar_equation, input_tensors, solution_evolution):
    """Visualize all results including new tensor structures."""
    print("\n=== Visualization ===")
    
    # Create subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Spherical mesh
    ax1 = fig.add_subplot(3, 4, 1, projection='3d')
    points = mesh.cartesian_coordinates
    ax1.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 2], cmap='viridis', s=1)
    ax1.set_title('Spherical Mesh')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # 2. Processed image with data type column
    ax2 = fig.add_subplot(3, 4, 2)
    image_tensor = input_tensors['image'][0]  # Remove sequence dimension
    # Show the main image content (without data type column)
    main_image = image_tensor[:, 1:, :]  # Skip data type column
    ax2.imshow(main_image.mean(dim=-1).cpu().numpy(), cmap='viridis')
    ax2.set_title('Processed Image (Main Content)')
    ax2.axis('off')
    
    # 3. Data type column visualization
    ax3 = fig.add_subplot(3, 4, 3)
    data_type_col = image_tensor[:, 0, :]  # Data type column
    ax3.imshow(data_type_col.cpu().numpy(), cmap='viridis')
    ax3.set_title('Data Type Column')
    ax3.axis('off')
    
    # 4. Audio MFCC features
    ax4 = fig.add_subplot(3, 4, 4)
    audio_tensor = input_tensors['audio'][0]
    main_audio = audio_tensor[:, 1:, :]  # Skip data type column
    ax4.imshow(main_audio.mean(dim=-1).cpu().numpy(), cmap='viridis')
    ax4.set_title('Audio MFCC Features')
    ax4.axis('off')
    
    # 5. Text binary representation
    ax5 = fig.add_subplot(3, 4, 5)
    text_tensor = input_tensors['text'][0]
    main_text = text_tensor[:, 1:, :]  # Skip data type column
    ax5.imshow(main_text.mean(dim=-1).cpu().numpy(), cmap='viridis')
    ax5.set_title('Text Binary Representation')
    ax5.axis('off')
    
    # 6. Tabular data with peripheral views
    ax6 = fig.add_subplot(3, 4, 6)
    tabular_tensor = input_tensors['tabular'][0]
    # Show the structure: [left_peripheral, main_content, right_peripheral]
    ax6.imshow(tabular_tensor.mean(dim=-1).cpu().numpy(), cmap='viridis')
    ax6.set_title('Tabular: [Left|Main|Right] Views')
    ax6.axis('off')
    
    # 7. Graph data with neighbor summaries
    ax7 = fig.add_subplot(3, 4, 7)
    graph_tensor = input_tensors['graph'][0]
    # Show the structure: [neighbor_summaries, main_content, node_attributes]
    ax7.imshow(graph_tensor.mean(dim=-1).cpu().numpy(), cmap='viridis')
    ax7.set_title('Graph: [Neighbors|Main|Attributes]')
    ax7.axis('off')
    
    # 8. Solution evolution
    ax8 = fig.add_subplot(3, 4, 8)
    if solution_evolution is not None:
        # Plot solution at different time steps
        time_steps = [0, len(solution_evolution)//4, len(solution_evolution)//2, len(solution_evolution)-1]
        for i, t in enumerate(time_steps):
            if t < len(solution_evolution):
                solution = solution_evolution[t]
                # Take a slice through the solution
                slice_data = solution[0, :, solution.shape[2]//2, 0].cpu().numpy()
                ax8.plot(slice_data, label=f't={t}', alpha=0.7)
        ax8.set_title('Solution Evolution')
        ax8.legend()
        ax8.set_xlabel('Latitude Index')
        ax8.set_ylabel('Solution Value')
    
    # 9. Tensor structure comparison
    ax9 = fig.add_subplot(3, 4, 9)
    tensor_shapes = {
        'Image': input_tensors['image'].shape,
        'Audio': input_tensors['audio'].shape,
        'Text': input_tensors['text'].shape,
        'Numerical': input_tensors['numerical'].shape,
        'Tabular': input_tensors['tabular'].shape,
        'Graph': input_tensors['graph'].shape
    }
    
    # Create a simple visualization of tensor shapes
    y_pos = np.arange(len(tensor_shapes))
    widths = [tensor_shapes[name][2] for name in tensor_shapes.keys()]  # Width dimension
    
    ax9.barh(y_pos, widths)
    ax9.set_yticks(y_pos)
    ax9.set_yticklabels(tensor_shapes.keys())
    ax9.set_xlabel('Tensor Width (including peripheral views)')
    ax9.set_title('Tensor Structure Comparison')
    
    # 10. Binary tokenization visualization
    ax10 = fig.add_subplot(3, 4, 10)
    # Show binary patterns in text processing
    text_tensor = input_tensors['text'][0]
    main_text = text_tensor[:, 1:, :]  # Skip data type column
    # Convert to binary visualization
    binary_viz = (main_text > 0.5).float()
    ax10.imshow(binary_viz.mean(dim=-1).cpu().numpy(), cmap='binary')
    ax10.set_title('Binary Tokenization')
    ax10.axis('off')
    
    # 11. Peripheral view analysis
    ax11 = fig.add_subplot(3, 4, 11)
    # Compare peripheral views across different data types
    peripheral_data = []
    labels = []
    
    for name, tensor in input_tensors.items():
        if tensor.shape[2] > 13:  # Has peripheral views
            left_peripheral = tensor[0, :, 0, :].mean(dim=-1).cpu().numpy()
            peripheral_data.append(left_peripheral)
            labels.append(f'{name} (L)')
            
            right_peripheral = tensor[0, :, -1, :].mean(dim=-1).cpu().numpy()
            peripheral_data.append(right_peripheral)
            labels.append(f'{name} (R)')
    
    if peripheral_data:
        peripheral_array = np.array(peripheral_data)
        ax11.imshow(peripheral_array, cmap='viridis', aspect='auto')
        ax11.set_yticks(range(len(labels)))
        ax11.set_yticklabels(labels)
        ax11.set_title('Peripheral Views Comparison')
        ax11.set_xlabel('Height Index')
    
    # 12. Data type encoding analysis
    ax12 = fig.add_subplot(3, 4, 12)
    # Show data type columns for all processors
    data_type_cols = []
    data_type_labels = []
    
    for name, tensor in input_tensors.items():
        data_type_col = tensor[0, :, 0, :].cpu().numpy()
        data_type_cols.append(data_type_col)
        data_type_labels.append(name)
    
    data_type_array = np.array(data_type_cols)
    ax12.imshow(data_type_array, cmap='viridis', aspect='auto')
    ax12.set_yticks(range(len(data_type_labels)))
    ax12.set_yticklabels(data_type_labels)
    ax12.set_title('Data Type Encoding')
    ax12.set_xlabel('Height Index')
    ax12.set_ylabel('Data Type')
    
    plt.tight_layout()
    plt.show()
    
    print("Visualization completed! Check the plots for:")
    print("- Spherical mesh structure")
    print("- Processed data tensors with data type columns")
    print("- Peripheral view summarization")
    print("- Binary tokenization patterns")
    print("- Tensor structure comparison")
    print("- Data type encoding analysis")


def main():
    """Run all demonstrations."""
    print("Wave Foundation Model (WFM) Demo")
    print("=" * 50)
    
    try:
        # Run demonstrations
        mesh, adaptive_mesh = demo_spherical_mesh()
        dar, discretizer, solution = demo_differential_equations()
        input_tensors = demo_input_processing()
        integrated_mesh, integrated_processor, mixed_tensor, integrated_solution = demo_integration()
        
        # Visualize results
        visualize_results(mesh, dar, input_tensors, solution)
        
        print("\n=== Demo Completed Successfully! ===")
        print("All components are working correctly.")
        
    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
