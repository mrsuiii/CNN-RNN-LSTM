import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, classification_report

from Value import Value

class Conv2D:
    """Convolutional layer implementation from scratch"""
    def __init__(self, filters: int, kernel_size: int, stride: int = 1, padding: str = 'valid'):
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weights = None
        self.biases = None
        
    def set_weights(self, weights: np.ndarray, biases: np.ndarray):
        """Set weights from trained Keras model"""
        self.weights = weights  # Shape: (kernel_h, kernel_w, input_channels, output_channels)
        self.biases = biases    # Shape: (output_channels,)
        
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through convolution layer"""
        batch_size, height, width, channels = x.shape
        kernel_h, kernel_w, in_channels, out_channels = self.weights.shape
        
        # Calculate output dimensions
        if self.padding == 'same':
            out_h = height // self.stride
            out_w = width // self.stride
            pad_h = max(0, (out_h - 1) * self.stride + kernel_h - height)
            pad_w = max(0, (out_w - 1) * self.stride + kernel_w - width)
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            x_padded = np.pad(x, ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant')
        else:
            out_h = (height - kernel_h) // self.stride + 1
            out_w = (width - kernel_w) // self.stride + 1
            x_padded = x
            
        output = np.zeros((batch_size, out_h, out_w, out_channels))
        
        # Perform convolution
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * self.stride
                h_end = h_start + kernel_h
                w_start = j * self.stride
                w_end = w_start + kernel_w
                
                x_slice = x_padded[:, h_start:h_end, w_start:w_end, :]
                
                for k in range(out_channels):
                    output[:, i, j, k] = np.sum(x_slice * self.weights[:, :, :, k], axis=(1, 2, 3)) + self.biases[k]
                    
        return output

class MaxPool2D:
    """Max pooling layer implementation from scratch"""
    def __init__(self, pool_size: int = 2, stride: int = 2):
        self.pool_size = pool_size
        self.stride = stride
        
    def __call__(self, x: np.ndarray) -> np.ndarray:
        batch_size, height, width, channels = x.shape
        out_h = (height - self.pool_size) // self.stride + 1
        out_w = (width - self.pool_size) // self.stride + 1
        
        output = np.zeros((batch_size, out_h, out_w, channels))
        
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * self.stride
                h_end = h_start + self.pool_size
                w_start = j * self.stride
                w_end = w_start + self.pool_size
                
                output[:, i, j, :] = np.max(x[:, h_start:h_end, w_start:w_end, :], axis=(1, 2))
                
        return output

class AvgPool2D:
    """Average pooling layer implementation from scratch"""
    def __init__(self, pool_size: int = 2, stride: int = 2):
        self.pool_size = pool_size
        self.stride = stride
        
    def __call__(self, x: np.ndarray) -> np.ndarray:
        batch_size, height, width, channels = x.shape
        out_h = (height - self.pool_size) // self.stride + 1
        out_w = (width - self.pool_size) // self.stride + 1
        
        output = np.zeros((batch_size, out_h, out_w, channels))
        
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * self.stride
                h_end = h_start + self.pool_size
                w_start = j * self.stride
                w_end = w_start + self.pool_size
                
                output[:, i, j, :] = np.mean(x[:, h_start:h_end, w_start:w_end, :], axis=(1, 2))
                
        return output

class Flatten:
    """Flatten layer implementation"""
    def __call__(self, x: np.ndarray) -> np.ndarray:
        batch_size = x.shape[0]
        return x.reshape(batch_size, -1)

class ReLU:
    """ReLU activation function"""
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

class CNNFromScratch:
    """CNN implementation from scratch using the modular approach"""
    def __init__(self):
        self.layers = []
        
    def add_layer(self, layer):
        self.layers.append(layer)
        
    def load_keras_weights(self, keras_model):
        """Load weights from trained Keras model"""
        keras_layers = [layer for layer in keras_model.layers if len(layer.get_weights()) > 0]
        
        # Separate different layer types
        scratch_conv_layers = [layer for layer in self.layers if isinstance(layer, Conv2D)]
        scratch_dense_layers = [layer for layer in self.layers if hasattr(layer, 'set_weights') and not isinstance(layer, Conv2D)]
        
        conv_idx = 0
        dense_idx = 0
        
        for keras_layer in keras_layers:
            weights = keras_layer.get_weights()
            
            if isinstance(keras_layer, tf.keras.layers.Conv2D):
                if conv_idx < len(scratch_conv_layers):
                    scratch_conv_layers[conv_idx].set_weights(weights[0], weights[1])
                    conv_idx += 1
                    
            elif isinstance(keras_layer, tf.keras.layers.Dense):
                if dense_idx < len(scratch_dense_layers):
                    scratch_dense_layers[dense_idx].set_weights(weights[0], weights[1])
                    dense_idx += 1
                    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Forward propagation through the network"""
        current_input = x
        is_value_object = False  # Track whether we're working with Value objects
        
        for i, layer in enumerate(self.layers):
            if isinstance(layer, Conv2D):
                # Conv2D layers work with numpy arrays
                if is_value_object:
                    # Convert Value back to numpy array
                    current_input = current_input.data
                    is_value_object = False
                
                current_input = layer(current_input)
                # Apply ReLU after convolution
                current_input = np.maximum(0, current_input)
                
            elif hasattr(layer, '__call__') and hasattr(layer, 'W'):
                # expects Value objects
                if not is_value_object:
                    # Convert numpy array to Value for Dense layer computation
                    current_input = Value(current_input)
                    is_value_object = True
                
                current_input = layer(current_input)
                # current_input is now a Value object
                
            else:
                # Regular layers (pooling, flatten, etc.) work with numpy arrays
                if is_value_object:
                    # Convert Value back to numpy array
                    current_input = current_input.data
                    is_value_object = False
                
                current_input = layer(current_input)
        
        # Ensure to return a numpy array
        if is_value_object:
            return current_input.data
        else:
            return current_input
    
    def calculate_flattened_size(self, input_shape, conv_layers, kernel_sizes, pooling_type):
        """Calculate the flattened size after conv and pooling layers"""
        height, width, channels = input_shape
        
        for filters, kernel_size in zip(conv_layers, kernel_sizes):
            # Conv layer with 'same' padding doesn't change spatial dimensions
            # Pooling layer reduces by factor of 2
            height = height // 2
            width = width // 2
            channels = filters
            
        return height * width * channels
