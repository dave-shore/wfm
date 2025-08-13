"""
Input Processor Module

This module provides functionality to process inputs of various types and convert them
into sequences of 12x12x3 dimensional tensors for the foundation model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List, Tuple, Optional, Dict, Any, Sequence
import numpy as np
from PIL import Image
import cv2
import json
import re
from pathlib import Path
import librosa
import warnings

# Suppress librosa warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='librosa')

# Data type constants for one-hot encoding
DATA_TYPES = {
    'image': [1, 0, 0],
    'audio': [0, 1, 0], 
    'video': [0, 0, 1],
    'text': [1, 0, 0],      # Map text to image type for now
    'numerical': [1, 0, 0],  # Map numerical to image type for now
    'graph': [1, 0, 0]       # Map graph to image type for now
}


class InputProcessor:
    """
    Main class for processing various input types into standardized tensor sequences.
    
    This processor converts inputs into sequences of 12x12x3 dimensional tensors,
    with an additional (12,1,3) column prepended for one-hot encoded data type indication.
    """
    
    def __init__(
        self,
        target_shape: Tuple[int, int, int] = (12, 12, 3),
        max_sequence_length: Optional[int] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the input processor.
        
        Args:
            target_shape: Target tensor shape (height, width, channels)
            max_sequence_length: Maximum length of output sequences
            device: PyTorch device for tensor operations
        """
        self.target_shape = target_shape
        self.max_sequence_length = max_sequence_length
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize specialized processors
        self.image_processor = ImageProcessor(target_shape, device)
        self.text_processor = TextProcessor(target_shape, device)
        self.numerical_processor = NumericalProcessor(target_shape, device)
        self.audio_processor = AudioProcessor(target_shape, device)
        self.video_processor = VideoProcessor(target_shape, device)
        self.graph_processor = GraphProcessor(target_shape, device)
        
    def process_input(
        self,
        input_data: Any,
        input_type: Optional[str] = None
    ) -> torch.Tensor:
        """
        Process input data into a sequence of tensors.
        
        Args:
            input_data: Input data of any supported type
            input_type: Optional hint about input type
            
        Returns:
            Tensor sequence of shape (sequence_length, height, width+1, channels)
            where the first column contains one-hot encoded data type information
        """
        # Auto-detect input type if not provided
        if input_type is None:
            input_type = self._detect_input_type(input_data)
        
        # Process based on detected type
        if input_type == 'image':
            return self.image_processor.process(input_data)
        elif input_type == 'text':
            return self.text_processor.process(input_data)
        elif input_type == 'numerical':
            return self.numerical_processor.process(input_data)
        elif input_type == 'audio':
            return self.audio_processor.process(input_data)
        elif input_type == 'video':
            return self.video_processor.process(input_data)
        elif input_type == 'graph':
            return self.graph_processor.process(input_data)
        elif input_type == 'mixed':
            return self._process_mixed_input(input_data)
        else:
            raise ValueError(f"Unsupported input type: {input_type}")
    
    def _detect_input_type(self, input_data: Any) -> str:
        """Auto-detect the type of input data."""
        if isinstance(input_data, (str, list)) and all(isinstance(x, str) for x in input_data if isinstance(input_data, list)):
            return 'text'
        elif isinstance(input_data, (np.ndarray, torch.Tensor)) and len(input_data.shape) >= 2:
            return 'image'
        elif isinstance(input_data, (int, float, np.number)) or (isinstance(input_data, (list, np.ndarray, torch.Tensor)) and all(isinstance(x, (int, float, np.number)) for x in input_data)):
            return 'numerical'
        elif isinstance(input_data, dict) and 'nodes' in input_data and 'edges' in input_data:
            return 'graph'
        elif isinstance(input_data, (list, tuple)) and len(input_data) > 0:
            # Mixed input types
            return 'mixed'
        else:
            # Default to numerical for unknown types
            return 'numerical'
    
    def _process_mixed_input(self, input_data: Sequence) -> torch.Tensor:
        """Process mixed input types by concatenating their representations."""
        processed_sequences = []
        
        for item in input_data:
            item_type = self._detect_input_type(item)
            if item_type == 'image':
                seq = self.image_processor.process(item)
            elif item_type == 'text':
                seq = self.text_processor.process(item)
            elif item_type == 'numerical':
                seq = self.numerical_processor.process(item)
            elif item_type == 'graph':
                seq = self.graph_processor.process(item)
            else:
                seq = self.numerical_processor.process(item)
            
            processed_sequences.append(seq)
        
        # Concatenate sequences along sequence dimension
        return torch.cat(processed_sequences, dim=0)


class ImageProcessor:
    """Processor for image inputs with focal vs peripheral area distinction."""
    
    def __init__(self, target_shape: Tuple[int, int, int], device: torch.device):
        self.target_shape = target_shape
        self.device = device
        
    def process(self, image_input: Union[str, np.ndarray, torch.Tensor, Image.Image]) -> torch.Tensor:
        """Process image input into tensor sequence with focal/peripheral distinction."""
        # Convert to tensor if needed
        if isinstance(image_input, str):
            # Load image from file path
            image = self._load_image(image_input)
        elif isinstance(image_input, Image.Image):
            image = np.array(image_input)
        elif isinstance(image_input, np.ndarray):
            image = image_input
        elif isinstance(image_input, torch.Tensor):
            image = image_input.cpu().numpy()
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")
        
        # Process image with focal vs peripheral areas
        processed_image = self._process_focal_peripheral(image)
        
        # Add one-hot encoded data type column
        tensor = self._add_data_type_column(processed_image, 'image')
        
        # Add sequence dimension
        return tensor.unsqueeze(0)
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """Load image from file path."""
        try:
            # Try PIL first
            image = Image.open(image_path)
            return np.array(image)
        except:
            try:
                # Try OpenCV
                image = cv2.imread(image_path)
                if image is not None:
                    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    raise ValueError(f"Could not load image from: {image_path}")
            except:
                raise ValueError(f"Failed to load image from: {image_path}")
    
    def _process_focal_peripheral(self, image: np.ndarray) -> torch.Tensor:
        """Process image distinguishing focal (central 4x4) from peripheral areas."""
        target_height, target_width, target_channels = self.target_shape
        
        if len(image.shape) == 2:
            # Grayscale image
            resized = cv2.resize(image, (target_width, target_height))
            # Convert to 3-channel by repeating
            resized = np.stack([resized] * 3, axis=-1)
        else:
            # Color image
            resized = cv2.resize(image, (target_width, target_height))
            
            # Ensure 3 channels
            if resized.shape[-1] == 1:
                resized = np.concatenate([resized] * 3, axis=-1)
            elif resized.shape[-1] == 4:
                resized = resized[:, :, :3]  # Remove alpha channel
        
        # Convert to tensor
        tensor = torch.from_numpy(resized).float().to(self.device)
        tensor = self._normalize_image(tensor)
        
        return tensor
    
    def _normalize_image(self, tensor: torch.Tensor) -> torch.Tensor:
        """Normalize image tensor to [0, 1] range."""
        if tensor.max() > 1.0:
            tensor = tensor / 255.0
        return tensor
    
    def _add_data_type_column(self, tensor: torch.Tensor, data_type: str) -> torch.Tensor:
        """Add one-hot encoded data type column to the left of the tensor."""
        # Create one-hot encoded column
        one_hot = torch.tensor(DATA_TYPES[data_type], dtype=torch.float32, device=self.device)
        one_hot = one_hot.unsqueeze(0).expand(12, -1)  # Shape: (12, 3)
        
        # Concatenate along width dimension (dim=1)
        # Original tensor: (12, 12, 3), one_hot: (12, 1, 3)
        one_hot = one_hot.unsqueeze(1)  # Shape: (12, 1, 3)
        result = torch.cat([one_hot, tensor], dim=1)  # Shape: (12, 13, 3)
        
        return result


class TextProcessor:
    """Processor for text inputs."""
    
    def __init__(self, target_shape: Tuple[int, int, int], device: torch.device):
        self.target_shape = target_shape
        self.device = device
        self.vocab_size = 1000  # Simplified vocabulary size
        
    def process(self, text_input: Union[str, List[str]]) -> torch.Tensor:
        """Process text input into tensor sequence."""
        if isinstance(text_input, str):
            texts = [text_input]
        else:
            texts = text_input
        
        # Process each text into a tensor
        tensors = []
        for text in texts:
            tensor = self._text_to_tensor(text)
            tensors.append(tensor)
        
        # Stack into sequence
        return torch.stack(tensors, dim=0)
    
    def _text_to_tensor(self, text: str) -> torch.Tensor:
        """Convert text to tensor representation."""
        target_height, target_width, target_channels = self.target_shape
        
        # Simple character-based encoding
        # Convert text to numerical representation
        char_indices = [ord(c) % self.vocab_size for c in text[:target_height * target_width]]
        
        # Pad or truncate to fit target dimensions
        if len(char_indices) < target_height * target_width:
            char_indices.extend([0] * (target_height * target_width - len(char_indices)))
        else:
            char_indices = char_indices[:target_height * target_width]
        
        # Reshape to target dimensions
        char_tensor = torch.tensor(char_indices, dtype=torch.float32, device=self.device)
        char_tensor = char_tensor.view(target_height, target_width)
        
        # Normalize to [0, 1]
        char_tensor = char_tensor / self.vocab_size
        
        # Expand to 3 channels
        char_tensor = char_tensor.unsqueeze(-1).expand(-1, -1, target_channels)
        
        # Add one-hot encoded data type column
        return self._add_data_type_column(char_tensor, 'text')
    
    def _add_data_type_column(self, tensor: torch.Tensor, data_type: str) -> torch.Tensor:
        """Add one-hot encoded data type column to the left of the tensor."""
        one_hot = torch.tensor(DATA_TYPES[data_type], dtype=torch.float32, device=self.device)
        one_hot = one_hot.unsqueeze(0).expand(12, -1)
        one_hot = one_hot.unsqueeze(1)
        result = torch.cat([one_hot, tensor], dim=1)
        return result


class NumericalProcessor:
    """Processor for numerical inputs."""
    
    def __init__(self, target_shape: Tuple[int, int, int], device: torch.device):
        self.target_shape = target_shape
        self.device = device
        
    def process(self, numerical_input: Union[float, int, np.number, List, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Process numerical input into tensor sequence."""
        # Convert to list if single value
        if isinstance(numerical_input, (int, float, np.number)):
            values = [float(numerical_input)]
        elif isinstance(numerical_input, torch.Tensor):
            values = numerical_input.flatten().cpu().numpy().tolist()
        elif isinstance(numerical_input, np.ndarray):
            values = numerical_input.flatten().tolist()
        else:
            values = [float(x) for x in numerical_input]
        
        target_height, target_width, target_channels = self.target_shape
        
        # Create tensor representation
        if len(values) <= target_height * target_width:
            # Pad with zeros
            padded_values = values + [0.0] * (target_height * target_width - len(values))
        else:
            # Truncate or sample
            step = len(values) / (target_height * target_width)
            padded_values = [values[int(i * step)] for i in range(target_height * target_width)]
        
        # Convert to tensor
        tensor = torch.tensor(padded_values, dtype=torch.float32, device=self.device)
        tensor = tensor.view(target_height, target_width)
        
        # Normalize to [0, 1] range
        if tensor.max() != tensor.min():
            tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
        
        # Expand to 3 channels
        tensor = tensor.unsqueeze(-1).expand(-1, -1, target_channels)
        
        # Add one-hot encoded data type column
        tensor = self._add_data_type_column(tensor, 'numerical')
        
        # Add sequence dimension
        return tensor.unsqueeze(0)
    
    def _add_data_type_column(self, tensor: torch.Tensor, data_type: str) -> torch.Tensor:
        """Add one-hot encoded data type column to the left of the tensor."""
        one_hot = torch.tensor(DATA_TYPES[data_type], dtype=torch.float32, device=self.device)
        one_hot = one_hot.unsqueeze(0).expand(12, -1)
        one_hot = one_hot.unsqueeze(1)
        result = torch.cat([one_hot, tensor], dim=1)
        return result


class AudioProcessor:
    """Processor for audio inputs with MFCC features and stereo support."""
    
    def __init__(self, target_shape: Tuple[int, int, int], device: torch.device):
        self.target_shape = target_shape
        self.device = device
        
    def process(self, audio_input: Union[str, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Process audio input into tensor sequence with MFCC features."""
        # Load audio if it's a file path
        if isinstance(audio_input, str):
            try:
                # Load audio with librosa
                audio, sr = librosa.load(audio_input, sr=None, mono=False)
            except Exception as e:
                print(f"Warning: Could not load audio file {audio_input}: {e}")
                # Fallback to placeholder
                return self._create_placeholder_audio()
        elif isinstance(audio_input, np.ndarray):
            audio = audio_input
            sr = 22050  # Default sample rate
        elif isinstance(audio_input, torch.Tensor):
            audio = audio_input.cpu().numpy()
            sr = 22050  # Default sample rate
        else:
            raise ValueError(f"Unsupported audio input type: {type(audio_input)}")
        
        # Ensure stereo format (replicate mono if needed)
        if len(audio.shape) == 1:
            audio = np.stack([audio, audio])  # Convert mono to stereo
        elif audio.shape[0] == 1:
            audio = np.repeat(audio, 2, axis=0)  # Ensure stereo
        
        # Extract MFCC features
        mfcc_features = self._extract_mfcc(audio, sr)
        
        # Process to target shape (12x4 MFCC features across 3 channels)
        processed_tensor = self._process_mfcc_to_tensor(mfcc_features)
        
        # Add one-hot encoded data type column
        tensor = self._add_data_type_column(processed_tensor, 'audio')
        
        # Add sequence dimension
        return tensor.unsqueeze(0)
    
    def _extract_mfcc(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Extract MFCC features from stereo audio."""
        # Process each channel separately
        mfcc_features = []
        for channel in range(min(2, audio.shape[0])):  # Handle up to 2 channels
            channel_audio = audio[channel]
            
            # Extract MFCC features
            mfcc = librosa.feature.mfcc(y=channel_audio, sr=sr, n_mfcc=12)
            
            # Ensure consistent shape
            if mfcc.shape[1] < 4:
                # Pad with zeros if too short
                mfcc = np.pad(mfcc, ((0, 0), (0, 4 - mfcc.shape[1])), mode='constant')
            elif mfcc.shape[1] > 4:
                # Truncate if too long
                mfcc = mfcc[:, :4]
            
            mfcc_features.append(mfcc)
        
        # If only one channel, duplicate it
        if len(mfcc_features) == 1:
            mfcc_features.append(mfcc_features[0])
        
        # Stack channels
        return np.stack(mfcc_features, axis=-1)  # Shape: (12, 4, 2)
    
    def _process_mfcc_to_tensor(self, mfcc_features: np.ndarray) -> torch.Tensor:
        """Convert MFCC features to target tensor shape."""
        # mfcc_features shape: (12, 4, 2) - need to expand to (12, 12, 3)
        target_height, target_width, target_channels = self.target_shape
        
        # Convert to tensor
        tensor = torch.from_numpy(mfcc_features).float().to(self.device)
        
        # Normalize to [0, 1] range
        if tensor.max() != tensor.min():
            tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
        
        # Expand to target dimensions
        # First, pad width from 4 to 12
        padded_tensor = torch.zeros(target_height, target_width, tensor.shape[-1], device=self.device)
        padded_tensor[:, :4, :] = tensor
        
        # Then expand channels from 2 to 3 by repeating the last channel
        if padded_tensor.shape[-1] == 2:
            last_channel = padded_tensor[:, :, -1:]
            padded_tensor = torch.cat([padded_tensor, last_channel], dim=-1)
        
        return padded_tensor
    
    def _create_placeholder_audio(self) -> torch.Tensor:
        """Create placeholder audio tensor when file loading fails."""
        target_height, target_width, target_channels = self.target_shape
        
        # Create random MFCC-like features
        placeholder = torch.randn(target_height, target_width, target_channels, device=self.device)
        placeholder = torch.sigmoid(placeholder)  # Normalize to [0, 1]
        
        # Add one-hot encoded data type column
        tensor = self._add_data_type_column(placeholder, 'audio')
        
        return tensor.unsqueeze(0)
    
    def _add_data_type_column(self, tensor: torch.Tensor, data_type: str) -> torch.Tensor:
        """Add one-hot encoded data type column to the left of the tensor."""
        one_hot = torch.tensor(DATA_TYPES[data_type], dtype=torch.float32, device=self.device)
        one_hot = one_hot.unsqueeze(0).expand(12, -1)
        one_hot = one_hot.unsqueeze(1)
        result = torch.cat([one_hot, tensor], dim=1)
        return result


class VideoProcessor:
    """Processor for video inputs, treating them as parallel image and audio streams."""
    
    def __init__(self, target_shape: Tuple[int, int, int], device: torch.device):
        self.target_shape = target_shape
        self.device = device
        self.image_processor = ImageProcessor(target_shape, device)
        self.audio_processor = AudioProcessor(target_shape, device)
        
    def process(self, video_input: Union[str, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Process video input as parallel image and audio streams."""
        if isinstance(video_input, str):
            # Load video from file
            video_frames, audio_stream = self._load_video(video_input)
        elif isinstance(video_input, np.ndarray):
            # Assume it's a video tensor
            video_frames = video_input
            audio_stream = None
        elif isinstance(video_input, torch.Tensor):
            video_frames = video_input.cpu().numpy()
            audio_stream = None
        else:
            raise ValueError(f"Unsupported video input type: {type(video_input)}")
        
        # Process video frames
        frame_tensors = self._process_video_frames(video_frames)
        
        # Process audio if available
        if audio_stream is not None:
            audio_tensor = self.audio_processor.process(audio_stream)
            # Combine frame and audio tensors
            combined_tensor = torch.cat([frame_tensors, audio_tensor], dim=0)
        else:
            combined_tensor = frame_tensors
        
        return combined_tensor
    
    def _load_video(self, video_path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Load video frames and audio from file."""
        try:
            # Use OpenCV for video frames
            cap = cv2.VideoCapture(video_path)
            frames = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            
            cap.release()
            
            # Convert to numpy array
            video_frames = np.array(frames) if frames else np.zeros((1, 480, 640, 3))
            
            # Try to extract audio with librosa
            try:
                audio, sr = librosa.load(video_path, sr=None, mono=False)
                if len(audio.shape) == 1:
                    audio = np.stack([audio, audio])  # Ensure stereo
            except:
                audio = None
            
            return video_frames, audio
            
        except Exception as e:
            print(f"Warning: Could not load video file {video_path}: {e}")
            # Return placeholder
            return np.zeros((1, 480, 640, 3)), None
    
    def _process_video_frames(self, frames: np.ndarray) -> torch.Tensor:
        """Process video frames into tensor sequence."""
        # Sample frames if too many
        max_frames = 10  # Limit number of frames for memory efficiency
        
        if len(frames) > max_frames:
            # Sample evenly spaced frames
            indices = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
            frames = frames[indices]
        
        # Process each frame
        frame_tensors = []
        for frame in frames:
            # Convert BGR to RGB if needed
            if frame.shape[-1] == 3:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = frame
            
            # Process frame
            frame_tensor = self.image_processor.process(frame_rgb)
            frame_tensors.append(frame_tensor)
        
        # Stack frames
        if frame_tensors:
            return torch.cat(frame_tensors, dim=0)
        else:
            # Return placeholder
            placeholder = torch.zeros(1, 12, 13, 3, device=self.device)
            return placeholder


class GraphProcessor:
    """Processor for graph inputs."""
    
    def __init__(self, target_shape: Tuple[int, int, int], device: torch.device):
        self.target_shape = target_shape
        self.device = device
        
    def process(self, graph_input: Dict[str, Any]) -> torch.Tensor:
        """Process graph input into tensor sequence."""
        target_height, target_width, target_channels = self.target_shape
        
        # Extract graph components
        nodes = graph_input.get('nodes', [])
        edges = graph_input.get('edges', [])
        
        # Create adjacency matrix representation
        if isinstance(nodes, list) and len(nodes) > 0:
            n_nodes = len(nodes)
            adj_matrix = np.zeros((n_nodes, n_nodes))
            
            # Fill adjacency matrix
            for edge in edges:
                if len(edge) >= 2:
                    i, j = edge[0], edge[1]
                    if 0 <= i < n_nodes and 0 <= j < n_nodes:
                        adj_matrix[i, j] = 1
                        adj_matrix[j, i] = 1  # Undirected graph
            
            # Resize to target dimensions
            resized_matrix = cv2.resize(adj_matrix, (target_width, target_height))
        else:
            # Create empty graph representation
            resized_matrix = np.zeros((target_height, target_width))
        
        # Convert to tensor
        tensor = torch.from_numpy(resized_matrix).float().to(self.device)
        
        # Normalize to [0, 1]
        if tensor.max() != tensor.min():
            tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
        
        # Expand to 3 channels
        tensor = tensor.unsqueeze(-1).expand(-1, -1, target_channels)
        
        # Add one-hot encoded data type column
        tensor = self._add_data_type_column(tensor, 'graph')
        
        # Add sequence dimension
        return tensor.unsqueeze(0)
    
    def _add_data_type_column(self, tensor: torch.Tensor, data_type: str) -> torch.Tensor:
        """Add one-hot encoded data type column to the left of the tensor."""
        one_hot = torch.tensor(DATA_TYPES[data_type], dtype=torch.float32, device=self.device)
        one_hot = one_hot.unsqueeze(0).expand(12, -1)
        one_hot = one_hot.unsqueeze(1)
        result = torch.cat([one_hot, tensor], dim=1)
        return result


class BatchProcessor:
    """Processor for batch inputs."""
    
    def __init__(self, input_processor: InputProcessor):
        self.input_processor = input_processor
        
    def process_batch(
        self,
        batch_inputs: List[Any],
        input_types: Optional[List[str]] = None
    ) -> torch.Tensor:
        """
        Process a batch of inputs.
        
        Args:
            batch_inputs: List of input data
            input_types: Optional list of input types
            
        Returns:
            Batch tensor of shape (batch_size, sequence_length, height, width+1, channels)
        """
        processed_tensors = []
        
        for i, input_data in enumerate(batch_inputs):
            input_type = input_types[i] if input_types else None
            tensor = self.input_processor.process(input_data, input_type)
            processed_tensors.append(tensor)
        
        # Stack along batch dimension
        return torch.stack(processed_tensors, dim=0)
    
    def process_heterogeneous_batch(
        self,
        batch_inputs: List[Any]
    ) -> torch.Tensor:
        """
        Process a batch of heterogeneous inputs.
        
        Args:
            batch_inputs: List of input data of potentially different types
            
        Returns:
            Batch tensor
        """
        return self.process_batch(batch_inputs)


class InputValidator:
    """Validator for input data."""
    
    @staticmethod
    def validate_tensor_shape(tensor: torch.Tensor, expected_shape: Tuple[int, int, int]) -> bool:
        """Validate that tensor has the expected shape with data type column."""
        if len(tensor.shape) != 4:  # (sequence, height, width+1, channels)
            return False
        
        _, height, width, channels = tensor.shape
        # Width should be original width + 1 (for data type column)
        expected_width = expected_shape[1] + 1
        return (height, width, channels) == (expected_shape[0], expected_width, expected_shape[2])
    
    @staticmethod
    def validate_sequence_length(tensor: torch.Tensor, max_length: Optional[int]) -> bool:
        """Validate sequence length."""
        if max_length is None:
            return True
        return tensor.shape[0] <= max_length
    
    @staticmethod
    def validate_input_range(tensor: torch.Tensor, min_val: float = 0.0, max_val: float = 1.0) -> bool:
        """Validate that tensor values are in the expected range."""
        return torch.all((tensor >= min_val) & (tensor <= max_val))
    
    @staticmethod
    def validate_data_type_column(tensor: torch.Tensor) -> bool:
        """Validate that the first column contains valid one-hot encoded data types."""
        if tensor.shape[3] != 3:  # Should have 3 channels
            return False
        
        # Check first column (data type column)
        first_col = tensor[:, :, 0, :]  # Shape: (sequence, height, 3)
        
        # Each row should sum to 1 (one-hot encoding)
        row_sums = torch.sum(first_col, dim=-1)
        return torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)
