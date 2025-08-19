# Wave Foundation Model (WFM)

A PyTorch-based foundation model with spherical mesh, differential equations, and **enhanced multi-modal input processing** with binary tokenization and peripheral view summarization.

## Key Features

- **Spherical Mesh Generation**: 3D spherical mesh with coordinate transformations and collision avoidance
- **Differential Equation Solver**: Diffusion-advection-reaction equations on spherical meshes
- **Enhanced Multi-Modal Input Processing**: 
  - **Binary Tokenization**: SentencePiece-like tokenizer with hierarchical clustering and binary encoding
  - **Peripheral View Summarization**: Row/column summaries for tabular data, neighbor/node summaries for graphs
  - **Advanced Audio Processing**: MFCC features with stereo support
  - **Video Processing**: Parallel image and audio stream processing
- **Unified Tensor Representation**: All data types converted to `(12, 13, 3)` tensors with data type encoding

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd wfm

# Install dependencies
poetry install

# Or install with pip
pip install -e .
```

## Quick Start

### 1. Spherical Mesh Generation

```python
from wfm import SphericalMesh, SphericalMeshBuilder

# Create a uniform spherical mesh
mesh = SphericalMesh(radius=1.0, n_lat=64, n_lon=128)

# Or use predefined mesh types
adaptive_mesh = SphericalMeshBuilder.create_adaptive_mesh()
high_res_mesh = SphericalMeshBuilder.create_high_resolution_mesh()

# Transform coordinates
cartesian_points = mesh.cartesian_coordinates
spherical_points = mesh.spherical_coordinates
```

### 2. Differential Equations

```python
from wfm import DiffusionAdvectionReaction, SphericalDiscretizer

# Define a DAR equation
dar = DiffusionAdvectionReaction(
    diffusion_coeff=1.0,
    velocity_field=lambda x: torch.tensor([0.1, 0.0, 0.0]),
    reaction_function=lambda x: -0.1 * x
)

# Discretize and solve
discretizer = SphericalDiscretizer(mesh, time_step=0.01)
solution = discretizer.discretize_equation(dar, initial_condition, time_steps=100)
```

### 3. Enhanced Input Processing

```python
from wfm import InputProcessor, BinaryTokenizer

# Initialize processor with binary tokenization
processor = InputProcessor(target_shape=(12, 12, 3))

# Process different data types
text_tensor = processor.process_input("Hello, world!")  # Binary tokenization
tabular_tensor = processor.process_input(pd.DataFrame(...))  # Peripheral views
graph_tensor = processor.process_input(nx.Graph(...))  # Neighbor summaries
image_tensor = processor.process_input(image_array)  # RGB processing
audio_tensor = processor.process_input(audio_array)  # MFCC features
```

## Architecture

### Core Components

1. **SphericalMesh**: Generates and manages 3D spherical meshes
2. **DiffusionAdvectionReaction**: Defines and solves PDEs on spherical domains
3. **InputProcessor**: Main orchestrator for multi-modal data processing

### Specialized Processors

- **ImageProcessor**: RGB pixel processing with focal/peripheral view concept
- **AudioProcessor**: MFCC feature extraction with stereo support
- **VideoProcessor**: Parallel frame and audio stream processing
- **TextProcessor**: Binary tokenization with hierarchical clustering
- **NumericalProcessor**: Text-based representation with statistical summaries
- **TabularProcessor**: Text-like encoding with row/column peripheral summaries
- **GraphProcessor**: Tabular conversion with neighbor/node attribute summaries

### Binary Tokenization

The `BinaryTokenizer` mimics SentencePiece with:
- **Hierarchical Clustering**: Organizes tokens using Ward clustering on embeddings
- **Binary Encoding**: Tokens encoded as binary vectors stored as base-10 integers
- **L² Dimension**: Binary dimension equals focus patch size (12² = 144)
- **Proximity Preservation**: Similar tokens get similar binary codes

### Peripheral View Summarization

- **Tabular Data**: 
  - Left view: Row summary statistics (mean, std, min, max)
  - Right view: Column summary statistics
- **Graph Data**:
  - Left view: Neighbor summaries (count, degree stats)
  - Right view: Node attribute summaries

## Data Type Encoding

Each processed tensor includes a `(12, 1, 3)` data type column:
- **Image/Text/Numerical**: `[1, 0, 0]` (text-like types)
- **Audio**: `[0, 1, 0]` (audio type)
- **Video**: `[0, 0, 1]` (video type)

## Tensor Structure

The unified tensor structure is `(sequence_length, height, width+1, channels)` where:
- `height = 12`: Standard height dimension
- `width+1`: Includes data type column + peripheral views + main content
- `channels = 3`: RGB/feature channels

For data with peripheral views:
```
[Data Type | Left Peripheral | Main Content | Right Peripheral]
[   (1)    |      (1)       |     (11)     |      (1)      ]
```

## Advanced Usage

### Specialized Input Processing

#### Text with Binary Tokenization

```python
from wfm import BinaryTokenizer

# Initialize with a real multilingual tokenizer
tokenizer = BinaryTokenizer(
    base_tokenizer=your_pretrained_tokenizer,
    vocab_size=8192,
    binary_dim=144  # 12²
)

# Process text
binary_codes = tokenizer.encode("Sample text")
text_matrix = tokenizer.get_binary_matrix("Sample text")
```

#### Tabular Data with Peripheral Views

```python
from wfm import TabularProcessor

processor = TabularProcessor(target_shape=(12, 12, 3))
df = pd.DataFrame({
    'feature1': [1.2, 3.4, 5.6],
    'feature2': [2.1, 4.3, 6.5]
})

# Creates tensor with row/column summaries in peripheral views
tensor = processor.process(df)
```

#### Graph Data with Neighbor Summaries

```python
from wfm import GraphProcessor

processor = GraphProcessor(target_shape=(12, 12, 3))
G = nx.Graph()
G.add_edges_from([(0, 1), (1, 2), (2, 0)])

# Creates tensor with neighbor and node attribute summaries
tensor = processor.process(G)
```

## Examples

Run the comprehensive demo:

```bash
python examples/foundation_model_demo.py
```

This demonstrates:
- Spherical mesh generation
- DAR equation solving
- All input processors with new features
- Visualization of tensor structures and peripheral views

## Performance Considerations

- **Binary Tokenization**: Efficient integer-based storage vs. one-hot encoding
- **Peripheral Views**: Compact summarization reduces memory footprint
- **Hierarchical Clustering**: Pre-computed token organization for fast lookup
- **Batch Processing**: Support for processing multiple inputs simultaneously

## Extending the Package

### Adding New Data Types

1. Create a new processor class inheriting from base processors
2. Implement the `process()` method
3. Add peripheral view summarization if needed
4. Update `InputProcessor._is_*()` methods

### Custom Binary Tokenization

```python
class CustomTokenizer(BinaryTokenizer):
    def _create_binary_code(self, cluster_id, token_pos, num_clusters):
        # Implement custom binary encoding logic
        pass
```

## Testing

```bash
# Run tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_input_processor.py -v
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests
5. Submit a pull request

## Citation

```bibtex
@software{wfm2024,
  title={Wave Foundation Model: A PyTorch-based foundation model with spherical mesh and enhanced multi-modal processing},
  author={Wave Foundation Model Team},
  year={2024},
  url={https://github.com/your-repo/wfm}
}
```

## Roadmap

- [ ] Integration with real multilingual tokenizers (HuggingFace, SentencePiece)
- [ ] Advanced peripheral view algorithms (attention-based summarization)
- [ ] Performance benchmarking suite
- [ ] Additional visualization tools
- [ ] GPU acceleration for large-scale processing
- [ ] Pre-trained models and embeddings
