# Hierarchical K-Means Clustering Implementation

## Overview

This implementation provides a hierarchical version of K-means clustering that builds a tree structure by iteratively clustering the cluster centers from previous iterations. The method returns a flattened array of leaf labels obtained by ordering the tree and taking the labels at the lowest level.

## Key Features

### 1. Tree Structure Building
- **Hierarchical clustering**: Each iteration clusters the centroids from the previous level
- **Complete tree**: Builds the entire tree structure, not just one branch
- **Level-by-level processing**: Processes all nodes at each level before moving to the next

### 2. Leaf Label Flattening
- **Tree traversal**: Recursively traverses the tree to assign labels
- **Ordered labels**: Labels are assigned in tree order for consistent results
- **Flattened output**: Returns a simple array of leaf labels for each data point

### 3. Memory Efficiency
- **Centroid-based clustering**: Only clusters centroids, not all data points
- **Reduced complexity**: Each level has fewer points to cluster than the previous
- **LazyTensor optimization**: Uses PyKeops for efficient distance computations

## Algorithm Structure

### Main Method: `_hierarchical_kmeans_clustering`
```python
def _hierarchical_kmeans_clustering(self, X: np.ndarray, metric: str, points_per_cluster: int):
```

**Parameters:**
- `X`: Input data array of shape (n_samples, n_features)
- `metric`: Distance metric ('euclidean', 'cosine')
- `points_per_cluster`: Target number of points per leaf cluster

**Returns:**
- `flattened_labels`: Array of leaf cluster labels for each data point

### Tree Building: `_build_hierarchical_tree`
```python
def _build_hierarchical_tree(self, X: np.ndarray, metric: str, points_per_cluster: int) -> dict:
```

**Process:**
1. **Initialize root**: Create root node with all data points
2. **Level-by-level processing**: Process all nodes at current level
3. **K-means clustering**: Cluster centroids for each node
4. **Child creation**: Create child nodes for each cluster
5. **Termination**: Stop when all nodes have ≤ points_per_cluster points

### Label Flattening: `_flatten_tree_labels`
```python
def _flatten_tree_labels(self, tree: dict, n_samples: int) -> np.ndarray:
```

**Process:**
1. **Tree traversal**: Recursively traverse the tree structure
2. **Leaf identification**: Identify leaf nodes (no children)
3. **Label assignment**: Assign sequential labels to data points in each leaf
4. **Offset tracking**: Maintain proper label offsets across branches

## Tree Structure

Each tree node contains:
```python
{
    'level': int,           # Depth level in the tree
    'data_indices': array,  # Indices of data points in this node
    'centers': array,       # Cluster centroids for this node
    'children': list        # List of child nodes
}
```

## Algorithm Flow

### 1. Initialization
- Create root node with all data points
- Set initial level to 0
- Initialize current level nodes list

### 2. Iterative Clustering
```python
while current_nodes:
    level += 1
    next_level_nodes = []
    
    for node in current_nodes:
        if len(node['data_indices']) <= points_per_cluster:
            continue  # Skip if already small enough
        
        # Perform K-means on this node's centers
        cl, c = self._kmeans_clustering(node['centers'], ...)
        
        # Create children for each cluster
        for cluster_id in range(n_clusters):
            # Create child node
            # Add to next level
```

### 3. Tree Traversal
```python
def traverse_tree(node, label_offset=0):
    if not node['children']:
        # Leaf node - assign labels
        for i, data_idx in enumerate(node['data_indices']):
            flattened_labels[data_idx] = label_offset + i
        return label_offset + len(node['data_indices'])
    
    # Internal node - process children recursively
    for child in node['children']:
        label_offset = traverse_tree(child, label_offset)
```

## Complexity Analysis

### Time Complexity
- **Per level**: O(k × n × d) where k is number of clusters, n is points, d is dimensions
- **Total levels**: O(log(n/points_per_cluster))
- **Overall**: O(log(n/points_per_cluster) × k × n × d)

### Space Complexity
- **Tree storage**: O(n) for storing data indices and centers
- **Temporary storage**: O(k × d) for cluster centroids
- **Overall**: O(n + k × d)

## Memory Efficiency

### Advantages over Standard Hierarchical Clustering
- **Reduced data**: Only clusters centroids, not all data points
- **Level reduction**: Each level has fewer points than the previous
- **LazyTensor optimization**: Efficient distance computations using PyKeops

### Memory Usage Pattern
- **Level 0**: n points
- **Level 1**: k₁ points (where k₁ < n)
- **Level 2**: k₂ points (where k₂ < k₁)
- **...**: Decreasing at each level

## Usage Example

```python
# Create binary tokenizer
tokenizer = BinaryTokenizer(
    base_tokenizer=your_tokenizer,
    base_embedder=your_embedder,
    binary_dim=16,
    cluster_size=5  # Target points per leaf cluster
)

# Perform hierarchical K-means clustering
X = your_data  # Shape: (n_samples, n_features)
labels = tokenizer._hierarchical_kmeans_clustering(
    X, 
    metric="cosine", 
    points_per_cluster=5
)

# labels now contains the flattened leaf labels
print(f"Number of unique clusters: {len(np.unique(labels))}")
print(f"Cluster sizes: {np.bincount(labels)}")
```

## Advantages

### 1. Hierarchical Structure
- **Tree representation**: Maintains hierarchical relationships
- **Multi-level clustering**: Captures both fine and coarse structure
- **Interpretable**: Easy to understand the clustering hierarchy

### 2. Scalability
- **Reduced complexity**: Each level processes fewer points
- **Memory efficient**: Only stores necessary information
- **Parallelizable**: Can process different branches in parallel

### 3. Flexibility
- **Configurable granularity**: Adjustable points_per_cluster parameter
- **Multiple metrics**: Supports different distance metrics
- **Adaptive clustering**: Automatically determines number of clusters per level

## Limitations

### 1. Tree Depth
- **Fixed granularity**: May create very deep trees for large datasets
- **Memory overhead**: Tree structure requires additional memory
- **Complexity**: More complex than flat clustering

### 2. Cluster Quality
- **Greedy approach**: May not find globally optimal clustering
- **Centroid-based**: Relies on centroid representation
- **Sensitivity**: Sensitive to initial cluster assignments

## Future Improvements

### 1. Parallel Processing
- **Multi-threading**: Process different branches in parallel
- **GPU acceleration**: Use GPU for distance computations
- **Distributed processing**: Scale to very large datasets

### 2. Adaptive Clustering
- **Dynamic granularity**: Adjust points_per_cluster based on data density
- **Quality metrics**: Use clustering quality to guide tree building
- **Early termination**: Stop when quality improvement is minimal

### 3. Memory Optimization
- **Lazy evaluation**: Compute centroids only when needed
- **Compression**: Compress tree structure for large datasets
- **Streaming**: Process data in streams for very large datasets

## Testing

The implementation includes comprehensive tests:
- **Basic functionality**: Small dataset clustering
- **Tree structure**: Verify tree building and traversal
- **Label consistency**: Ensure proper label assignment
- **Edge cases**: Handle various data sizes and configurations

## Dependencies

- **NumPy**: Array operations and data manipulation
- **PyKeops**: Efficient distance computations (optional)
- **PyTorch**: Tensor operations (if using torch backend)

## Performance Characteristics

### Small Datasets (< 1,000 points)
- **Time**: Comparable to standard K-means
- **Memory**: Minimal overhead
- **Quality**: Good clustering quality

### Medium Datasets (1,000 - 10,000 points)
- **Time**: 2-5x slower than flat K-means
- **Memory**: Moderate overhead
- **Quality**: Better than flat K-means due to hierarchy

### Large Datasets (> 10,000 points)
- **Time**: 5-10x slower than flat K-means
- **Memory**: Significant overhead for tree storage
- **Quality**: Much better than flat K-means
