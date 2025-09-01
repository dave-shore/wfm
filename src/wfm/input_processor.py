"""
Input Processor Module

This module provides functionality to process inputs of various types and convert them
into sequences of 12x12x3 dimensional tensors for the foundation model.
"""

import torch
from transformers import AutoTokenizer, AutoModel
from typing import Union, List, Tuple, Optional, Dict, Any, Sequence
import numpy as np
from PIL import Image
import cv2
import librosa
import warnings
from sklearn.cluster import AgglomerativeClustering
import polars as pl
import os

# Suppress librosa warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='librosa')

# Data type constants for one-hot encoding
DATA_TYPES = [
    "text",
    "image",
    "audio",
    "video",
    "num",
    "table",
    "code"
]


class InputProcessor:
    """
    Main input processor that handles various data types and converts them to
    sequences of tensors with data type encoding and peripheral view summarization.
    
    The output tensor has shape (sequence_length, height, width+1, channels) where:
    - width+1 includes the data type column and peripheral views
    - Left peripheral view: neighbor/row summaries
    - Main content: processed data
    - Right peripheral view: node attributes/column summaries
    """
    
    def __init__(self, target_shape: Tuple[int, int, int] = (4, 4, 3), base_model: str = "utter-project/EuroLLM-1.7B-Instruct", code_model: str = "google/codegemma-2b", device: Optional[torch.device | str] = None):
        """
        Initialize the input processor.
        
        Args:
            target_shape: Target tensor shape (height, width, channels)
            device: PyTorch device (defaults to CPU)
        """
        self.target_shape = target_shape
        self.device = device or torch.device('cpu')

        # Initialize text-based processors with shared tokenizer
        self.text_binary_tokenizer = BinaryTokenizer(
            base_tokenizer=base_model, 
            base_embedder=base_model, 
            binary_dim=16  # 4**2 for focus patch size
        )
        self.text_processor = TextProcessor(target_shape, self.device, self.text_binary_tokenizer)
        
        # Initialize specialized processors
        self.image_processor = ImageProcessor(target_shape, self.device)
        self.audio_processor = AudioProcessor(target_shape, self.device)
        self.video_processor = VideoProcessor(target_shape, self.device)
        
        self.numerical_processor = NumericalProcessor(target_shape, self.device, self.text_binary_tokenizer)
        self.tabular_processor = TabularProcessor(target_shape, self.device, self.text_binary_tokenizer)

        self.code_binary_tokenizer = BinaryTokenizer(
            base_tokenizer=code_model, 
            base_embedder=code_model, 
            binary_dim=16  # 4**2 for focus patch size
        )
        self.code_processor = TextProcessor(target_shape, self.device, self.code_binary_tokenizer)
    
    def process_input(self, input_data: Any, is_ordered: bool = True) -> torch.Tensor:
        """
        Process input data to tensor sequence.
        
        Args:
            input_data: Input data of various types
            is_ordered: Whether the input data is ordered
            
        Returns:
            Tensor of shape (sequence_length, height, width+1, channels)
        """

        if is_ordered:
            # Determine input type and process accordingly
            if self._is_text(input_data):
                return self.text_processor.process(input_data)
            elif self._is_audio(input_data):
                return self.audio_processor.process(input_data)
            elif self._is_video(input_data):
                return self.video_processor.process(input_data)
            elif self._is_code(input_data):
                return self.code_processor.process(input_data)
            else:
                raise ValueError(f"Unsupported input type: {type(input_data)}")
        else:
            if self._is_image(input_data):
                return self.image_processor.process(input_data)
            elif self._is_numerical(input_data):
                return self.numerical_processor.process(input_data)
            elif self._is_tabular(input_data):
                return self.tabular_processor.process(input_data)
            else:
                raise ValueError(f"Unsupported input type: {type(input_data)}")
    
    def _is_image(self, data: Any) -> bool:
        """Check if data is image-like."""
        file_case = isinstance(data, str) and os.path.exists(data) and data.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
        data_case = isinstance(data, (np.ndarray, torch.Tensor)) and 3 <= len(data.shape) <= 4
        return file_case or data_case
    
    def _is_audio(self, data: Any) -> bool:
        """Check if data is audio-like."""
        file_case = isinstance(data, str) and os.path.exists(data) and data.lower().endswith(('.wav', '.mp3', '.flac', '.ogg'))
        data_case = isinstance(data, (np.ndarray, torch.Tensor)) and len(data.shape) <= 2
        return file_case or data_case
    
    def _is_video(self, data: Any) -> bool:
        """Check if data is video-like."""
        file_case = isinstance(data, str) and os.path.exists(data) and data.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
        data_case = isinstance(data, (np.ndarray, torch.Tensor)) and 4 <= len(data.shape) <=5
        return file_case or data_case
    
    def _is_text(self, data: Any) -> bool:
        """Check if data is text-like."""
        file_case = isinstance(data, str) and os.path.exists(data) and data.lower().endswith(('.txt', '.doc', '.docx', '.pdf'))
        data_case = isinstance(data, str) and not any([self._is_audio(data), self._is_video(data), self._is_code(data)])
        return file_case or data_case

    def _is_numerical(self, data: Any) -> bool:
        """Check if data is numerical-like."""
        file_case = isinstance(data, str) and os.path.exists(data) and data.lower().endswith((".dat"))
        data_case = isinstance(data, (float, int, np.number)) or (isinstance(data, (np.ndarray, torch.Tensor)) and len(data.shape) <= 1)
        return file_case or data_case
    
    def _is_tabular(self, data: Any) -> bool:
        """Check if data is tabular-like."""
        file_case = isinstance(data, str) and os.path.exists(data) and data.lower().endswith((".csv", ".tsv", ".xlsx", ".xls", ".parquet", ".json"))
        data_case = isinstance(data, (pl.DataFrame, np.ndarray)) and len(data.shape) == 2
        return file_case or data_case
    
    def _is_code(self, data: Any) -> bool:
        """Check if data is code-like."""
        file_case = isinstance(data, str) and os.path.exists(data) and data.lower().endswith(('.py', '.html', '.css', '.js', '.java', '.c', '.cpp', '.h', '.hpp', '.bat', '.sh', '.ps1', '.sql', '.xml', '.yaml', '.toml'))
        data_case = isinstance(data, str) and not any([self._is_audio(data), self._is_video(data), self._is_text(data)])
        return file_case or data_case


class BinaryTokenizer:
    """
    A binary tokenizer that mimics SentencePiece with hierarchical clustering.
    Tokens are encoded as binary vectors stored as base-10 integers.
    """
    
    def __init__(self, base_tokenizer, base_embedder, binary_dim: int = 16):
        """
        Initialize the binary tokenizer.
        
        Args:
            base_tokenizer: A pre-trained multilingual tokenizer (e.g., from transformers)
            base_embedder: A pre-trained multilingual embedding layer (e.g., from transformers)
            binary_dim: Dimension of binary vectors (should be L² where L is focus patch size)
        """

        if not np.sqrt(binary_dim).is_integer():
            raise ValueError("Binary dimension must be a perfect square")

        self.base_tokenizer = AutoTokenizer.from_pretrained(base_tokenizer) if isinstance(base_tokenizer, str) else base_tokenizer
        self.base_embedder = AutoModel.from_pretrained(base_embedder) if isinstance(base_embedder, str) else base_embedder
        self.vocab_size = 8**binary_dim
        self.binary_dim = binary_dim
        self.L = int(np.sqrt(binary_dim))
        self.token_to_binary = {}
        self.binary_to_token = {}
        self.token_embeddings = None
        self.cluster_labels = None
        self._build_binary_vocabulary()
    
    def _build_binary_vocabulary(self):
        """Build binary vocabulary using hierarchical clustering on base tokenizer embeddings."""
        # Get embeddings from base tokenizer
        if hasattr(self.base_tokenizer, 'get_vocab'):
            vocab = list(self.base_tokenizer.get_vocab().keys())
        else:
            vocab = list(self.base_tokenizer.vocab.keys())

        extended_vocab = vocab + [f"<{category}>" for category in DATA_TYPES] + [f"</{category}>" for category in DATA_TYPES] + list(self.base_tokenizer.special_tokens_dict.keys())
        self.extended_vocab = extended_vocab[-1:-self.vocab_size-1:-1]
        
        # Get embeddings
        W = self.base_embedder.weight.data.numpy()
        if W.shape[0] < W.shape[1]:
            W = W.T
        embeddings = W[-1:-len(vocab)-1:-1]
        embeddings = np.concatenate([
            np.random.randn(len(self.extended_vocab) - len(vocab), W.shape[1]),
            embeddings
        ], axis = 0)
        
        # Perform hierarchical clustering
        clustering = AgglomerativeClustering(min(self.vocab_size, len(self.extended_vocab)))
        cluster_labels = clustering.fit_predict(embeddings)
        
        # Assign binary codes based on clustering
        self._assign_binary_codes(self.extended_vocab, cluster_labels)
    
    def _assign_binary_codes(self, vocab: List[str], cluster_labels: np.ndarray):
        """Assign binary codes to tokens based on clustering."""
        
        for i, (token, cluster_id) in enumerate(zip(vocab, cluster_labels)):
            self.token_to_binary[token] = cluster_id
            self.binary_to_token[cluster_id] = token
    
    def encode(self, texts: str | List[str]) -> List[int]:
        """Encode text to list of binary integer codes."""

        if isinstance(texts, str):
            texts = [texts]

        # Tokenize with base tokenizer
        if hasattr(self.base_tokenizer, 'encode'):
            tokenized_texts = self.base_tokenizer.encode(texts)
        else:
            tokenized_texts = self.base_tokenizer.tokenize(texts)

        max_seq_length = max(len(encoding) for encoding in tokenized_texts)
        
        # Convert to binary codes
        binary_codes = []
        for encoding in tokenized_texts:
            binary_codes.append([])
            for token in encoding:
                if token not in self.token_to_binary:
                    self.token_to_binary[token] = max(self.token_to_binary.values()) + 1
                    self.binary_to_token[max(self.token_to_binary.values()) + 1] = token

                binary_codes[-1].append(
                    np.fromstring(np.binary_repr(self.token_to_binary[token], self.binary_dim*3, signed = False), dtype = "S1").astype(np.int8)
                )
            binary_codes[-1] = np.pad(binary_codes[-1], (0, max_seq_length - len(encoding)), mode = 'constant')

        binary_codes = np.array(binary_codes, dtype = np.int8).reshape(-1, max_seq_length, self.L, self.L, 3)
        # shape: (batch_size, sequence_length, L, L, 3)
        
        return binary_codes
    
    def decode(self, binary_codes: List[List[int | np.ndarray]]) -> str:
        """Decode binary integer codes back to text."""
        decoded_texts = []
        for encoding in binary_codes:
            text = ""
            for code in encoding:
                if isinstance(code, int):
                    token_id = code
                else:
                    token_id = code.reshape(-1).dot(2**np.arange(self.binary_dim*3))

                new_token = self.binary_to_token.get(token_id, '<???>')
                if new_token.startswith("Ġ") or new_token.startswith("#"):
                    text += new_token[1:]
                else:
                    text += " " + new_token

            decoded_texts.append(text.strip())
        
        return decoded_texts
    

class TextProcessor:
    """
    Processes text data using binary tokenization.
    """
    
    def __init__(self, target_shape: Tuple[int, int, int], device: torch.device, tokenizer: Optional[BinaryTokenizer] = None):
        """
        Initialize text processor.
        
        Args:
            target_shape: Target tensor shape (height, width, channels)
            device: PyTorch device
            tokenizer: Optional pre-initialized binary tokenizer
        """
        self.target_height, self.target_width, self.target_channels = target_shape
        self.device = device
        
        # Initialize default tokenizer if none provided
        if tokenizer is None:
            # Create a mock tokenizer for demonstration
            # In practice, you'd load a real multilingual tokenizer
            self.tokenizer = self._create_mock_tokenizer()
        else:
            self.tokenizer = tokenizer
    
    def _create_mock_tokenizer(self) -> BinaryTokenizer:
        """Create a mock tokenizer for demonstration purposes."""
        class MockTokenizer:
            def __init__(self):
                self.vocab = {f"token_{i}": i for i in range(1000)}
            
            def get_vocab(self):
                return self.vocab
        
        return BinaryTokenizer(MockTokenizer())
    
    def process(self, text_input: Union[str, List[str]]) -> torch.Tensor:
        """
        Process text input to target tensor shape.
        
        Args:
            text_input: Text string or list of strings
            
        Returns:
            Tensor of shape (target_height, target_width+1, target_channels)
        """
        if isinstance(text_input, str):
            text_input = [text_input]
        
        # Get binary matrix from tokenizer
        binary_matrix = self.tokenizer.get_binary_matrix(' '.join(text_input))
        
        # Reshape to target dimensions
        tensor = self._reshape_to_target(binary_matrix)
        
        # Add data type column
        tensor = self._add_data_type_column(tensor, 'text')
        
        return tensor.to(self.device)
    
    def _reshape_to_target(self, binary_matrix: torch.Tensor) -> torch.Tensor:
        """Reshape binary matrix to target dimensions."""
        # Flatten binary matrix
        flat_binary = binary_matrix.flatten()
        
        # Pad or truncate to fit target dimensions
        target_size = self.target_height * self.target_width * self.target_channels
        
        if len(flat_binary) < target_size:
            # Pad with zeros
            padding = torch.zeros(target_size - len(flat_binary))
            flat_binary = torch.cat([flat_binary, padding])
        else:
            # Truncate
            flat_binary = flat_binary[:target_size]
        
        # Reshape to target dimensions
        tensor = flat_binary.reshape(self.target_height, self.target_width, self.target_channels)
        
        return tensor
    
    def _add_data_type_column(self, tensor: torch.Tensor, data_type: str) -> torch.Tensor:
        """Add data type column to tensor."""
        # Create data type column (12, 1, 3)
        data_type_col = torch.zeros(self.target_height, 1, self.target_channels)
        
        # Set one-hot encoding for text
        data_type_col[:, 0, 0] = 1.0  # Text type
        
        # Concatenate with original tensor
        result = torch.cat([data_type_col, tensor], dim=1)
        
        return result


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


class NumericalProcessor:
    """
    Processes numerical data using binary tokenization.
    """
    
    def __init__(self, target_shape: Tuple[int, int, int], device: torch.device, tokenizer: Optional[BinaryTokenizer] = None):
        """
        Initialize numerical processor.
        
        Args:
            target_shape: Target tensor shape (height, width, channels)
            device: PyTorch device
            tokenizer: Optional binary tokenizer for text processing
        """
        self.target_height, self.target_width, self.target_channels = target_shape
        self.device = device
        
        # Initialize text processor for text-like encoding
        if tokenizer is None:
            self.text_processor = TextProcessor(target_shape, device)
        else:
            self.text_processor = TextProcessor(target_shape, device, tokenizer)
    
    def process(self, numerical_input: Union[np.ndarray, List[float], torch.Tensor]) -> torch.Tensor:
        """
        Process numerical input to target tensor shape.
        
        Args:
            numerical_input: Numerical data as array, list, or tensor
            
        Returns:
            Tensor of shape (target_height, target_width+1, target_channels)
        """
        # Convert to text representation
        text_representation = self._numerical_to_text(numerical_input)
        
        # Process using text processor
        tensor = self.text_processor.process(text_representation)
        
        return tensor
    
    def _numerical_to_text(self, numerical_input: Union[np.ndarray, List[float], torch.Tensor]) -> str:
        """Convert numerical data to text representation."""
        if isinstance(numerical_input, torch.Tensor):
            numerical_input = numerical_input.cpu().numpy()
        elif isinstance(numerical_input, list):
            numerical_input = np.array(numerical_input)
        
        # Flatten and convert to text
        flat_data = numerical_input.flatten()
        text_lines = []
        
        # Add summary statistics
        if len(flat_data) > 0:
            text_lines.append(f"mean {np.mean(flat_data):.6f}")
            text_lines.append(f"std {np.std(flat_data):.6f}")
            text_lines.append(f"min {np.min(flat_data):.6f}")
            text_lines.append(f"max {np.max(flat_data):.6f}")
        
        # Add individual values
        for val in flat_data[:100]:  # Limit to first 100 values
            text_lines.append(f"{val:.6f}")
        
        return '\n'.join(text_lines)


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


class TabularProcessor:
    """
    Processes tabular data with text-like encoding and peripheral view summarization.
    Left peripheral view contains row summaries, right peripheral view contains column summaries.
    """
    
    def __init__(self, target_shape: Tuple[int, int, int], device: torch.device, tokenizer: Optional[BinaryTokenizer] = None):
        """
        Initialize tabular processor.
        
        Args:
            target_shape: Target tensor shape (height, width, channels)
            device: PyTorch device
            tokenizer: Optional binary tokenizer for text processing
        """
        self.target_height, self.target_width, self.target_channels = target_shape
        self.device = device
        
        # Initialize text processor for text-like encoding
        if tokenizer is None:
            self.text_processor = TextProcessor(target_shape, device)
        else:
            self.text_processor = TextProcessor(target_shape, device, tokenizer)
    
    def process(self, tabular_input: Union[pd.DataFrame, np.ndarray, List[List]]) -> torch.Tensor:
        """
        Process tabular input to target tensor shape.
        
        Args:
            tabular_input: Tabular data as DataFrame, numpy array, or list of lists
            
        Returns:
            Tensor of shape (target_height, target_width+1, target_channels)
        """
        # Convert to DataFrame if needed
        if isinstance(tabular_input, np.ndarray):
            df = pd.DataFrame(tabular_input)
        elif isinstance(tabular_input, list):
            df = pd.DataFrame(tabular_input)
        else:
            df = tabular_input
        
        # Process as text-like data
        text_representation = self._tabular_to_text(df)
        tensor = self.text_processor.process(text_representation)
        
        # Add peripheral view summarization
        tensor = self._add_peripheral_summaries(tensor, df)
        
        return tensor
    
    def _tabular_to_text(self, df: pd.DataFrame) -> str:
        """Convert tabular data to text representation."""
        # Convert DataFrame to text format
        text_lines = []
        
        # Add column headers
        text_lines.append(' '.join(str(col) for col in df.columns))
        
        # Add data rows
        for _, row in df.iterrows():
            text_lines.append(' '.join(str(val) for val in row))
        
        return '\n'.join(text_lines)
    
    def _add_peripheral_summaries(self, tensor: torch.Tensor, df: pd.DataFrame) -> torch.Tensor:
        """Add peripheral view summarization to tensor."""
        # Extract the main tensor (without data type column)
        main_tensor = tensor[:, 1:, :]
        
        # Calculate row and column summaries
        row_summaries = self._calculate_row_summaries(df)
        col_summaries = self._calculate_column_summaries(df)
        
        # Reshape summaries to fit peripheral views
        row_summary_tensor = self._reshape_summaries_to_peripheral(row_summaries, 'row')
        col_summary_tensor = self._reshape_summaries_to_peripheral(col_summaries, 'column')
        
        # Combine main tensor with peripheral views
        result = torch.cat([row_summary_tensor, main_tensor, col_summary_tensor], dim=1)
        
        # Add data type column back
        data_type_col = tensor[:, 0:1, :]
        result = torch.cat([data_type_col, result], dim=1)
        
        return result
    
    def _calculate_row_summaries(self, df: pd.DataFrame) -> np.ndarray:
        """Calculate summary statistics for each row."""
        summaries = []
        for _, row in df.iterrows():
            # Calculate basic statistics for the row
            numeric_vals = pd.to_numeric(row, errors='coerce').dropna()
            if len(numeric_vals) > 0:
                summary = [
                    numeric_vals.mean(),
                    numeric_vals.std(),
                    numeric_vals.min(),
                    numeric_vals.max()
                ]
            else:
                summary = [0.0, 0.0, 0.0, 0.0]
            summaries.append(summary)
        
        return np.array(summaries)
    
    def _calculate_column_summaries(self, df: pd.DataFrame) -> np.ndarray:
        """Calculate summary statistics for each column."""
        summaries = []
        for col in df.columns:
            numeric_vals = pd.to_numeric(df[col], errors='coerce').dropna()
            if len(numeric_vals) > 0:
                summary = [
                    numeric_vals.mean(),
                    numeric_vals.std(),
                    numeric_vals.min(),
                    numeric_vals.max()
                ]
            else:
                summary = [0.0, 0.0, 0.0, 0.0]
            summaries.append(summary)
        
        return np.array(summaries)
    
    def _reshape_summaries_to_peripheral(self, summaries: np.ndarray, summary_type: str) -> torch.Tensor:
        """Reshape summaries to fit peripheral view dimensions."""
        if summary_type == 'row':
            # Left peripheral view: (12, 1, 3)
            peripheral_width = 1
        else:
            # Right peripheral view: (12, 1, 3)
            peripheral_width = 1
        
        # Pad or truncate summaries to fit peripheral dimensions
        target_size = self.target_height * peripheral_width * self.target_channels
        
        flat_summaries = summaries.flatten()
        if len(flat_summaries) < target_size:
            # Pad with zeros
            padding = np.zeros(target_size - len(flat_summaries))
            flat_summaries = np.concatenate([flat_summaries, padding])
        else:
            # Truncate
            flat_summaries = flat_summaries[:target_size]
        
        # Reshape to peripheral dimensions
        peripheral_tensor = torch.tensor(flat_summaries, dtype=torch.float32).reshape(
            self.target_height, peripheral_width, self.target_channels
        )
        
        return peripheral_tensor

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
    """Validates input tensors and data types."""
    
    @staticmethod
    def validate_tensor_shape(tensor: torch.Tensor, expected_height: int = 12, expected_channels: int = 3) -> bool:
        """
        Validate tensor shape including data type column and peripheral views.
        
        Args:
            tensor: Tensor to validate
            expected_height: Expected height dimension
            expected_channels: Expected number of channels
            
        Returns:
            True if valid, False otherwise
        """
        if tensor.dim() != 4:
            print(f"Expected 4D tensor, got {tensor.dim()}D")
            return False
        
        sequence_length, height, width, channels = tensor.shape
        
        if height != expected_height:
            print(f"Expected height {expected_height}, got {height}")
            return False
        
        if channels != expected_channels:
            print(f"Expected channels {expected_channels}, got {channels}")
            return False
        
        # Width should be at least 13 (1 data type + 12 main content)
        # For data with peripheral views, width will be larger
        if width < 13:
            print(f"Expected width >= 13, got {width}")
            return False
        
        return True
    
    @staticmethod
    def validate_data_type_column(tensor: torch.Tensor) -> bool:
        """
        Validate the data type column in the tensor.
        
        Args:
            tensor: Tensor to validate
            
        Returns:
            True if valid, False otherwise
        """
        if tensor.dim() != 4:
            return False
        
        # Extract data type column (first column)
        data_type_col = tensor[:, :, 0, :]
        
        # Check that each row has exactly one data type indicator
        # Data type should be one-hot encoded across the 3 channels
        for seq_idx in range(data_type_col.shape[0]):
            for row_idx in range(data_type_col.shape[1]):
                row_data = data_type_col[seq_idx, row_idx, :]
                
                # Check if it's a valid one-hot encoding
                if not torch.allclose(row_data.sum(), torch.tensor(1.0), atol=1e-6):
                    return False
                
                # Check if values are binary (0 or 1)
                if not torch.all((row_data == 0) | (row_data == 1)):
                    return False
        
        return True
    
    @staticmethod
    def validate_peripheral_views(tensor: torch.Tensor) -> bool:
        """
        Validate peripheral view structure in the tensor.
        
        Args:
            tensor: Tensor to validate
            
        Returns:
            True if valid, False otherwise
        """
        if tensor.dim() != 4:
            return False
        
        sequence_length, height, width, channels = tensor.shape
        
        # If width > 13, we have peripheral views
        if width > 13:
            # Check that peripheral views have reasonable dimensions
            # Left peripheral: width 1
            # Main content: width 12
            # Right peripheral: width 1
            # Total: 1 + 12 + 1 = 14
            
            if width == 14:
                # Standard peripheral view structure
                return True
            elif width > 14:
                # Extended peripheral view structure
                return True
            else:
                print(f"Invalid peripheral view structure: width {width}")
                return False
        
        return True
    
    @staticmethod
    def validate_tensor_content(tensor: torch.Tensor) -> bool:
        """
        Validate the content of the tensor.
        
        Args:
            tensor: Tensor to validate
            
        Returns:
            True if valid, False otherwise
        """
        if tensor.dim() != 4:
            return False
        
        # Check for NaN or infinite values
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            print("Tensor contains NaN or infinite values")
            return False
        
        # Check value ranges (should be reasonable for normalized data)
        tensor_min = tensor.min().item()
        tensor_max = tensor.max().item()
        
        # Allow for some flexibility in value ranges
        if tensor_min < -10 or tensor_max > 10:
            print(f"Tensor values out of expected range: [{tensor_min}, {tensor_max}]")
            return False
        
        return True
    
    @staticmethod
    def validate_complete_tensor(tensor: torch.Tensor) -> bool:
        """
        Perform complete validation of a tensor.
        
        Args:
            tensor: Tensor to validate
            
        Returns:
            True if all validations pass, False otherwise
        """
        print(f"Validating tensor with shape: {tensor.shape}")
        
        # Basic shape validation
        if not InputValidator.validate_tensor_shape(tensor):
            print("❌ Shape validation failed")
            return False
        
        # Data type column validation
        if not InputValidator.validate_data_type_column(tensor):
            print("❌ Data type column validation failed")
            return False
        
        # Peripheral view validation
        if not InputValidator.validate_peripheral_views(tensor):
            print("❌ Peripheral view validation failed")
            return False
        
        # Content validation
        if not InputValidator.validate_tensor_content(tensor):
            print("❌ Content validation failed")
            return False
        
        print("✅ All validations passed")
        return True
