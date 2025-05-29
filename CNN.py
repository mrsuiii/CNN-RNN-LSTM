import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, classification_report
import pickle
import os
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass

# Import Value class from previous assignment (assumed to be available)
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
                # This is a Dense layer from Layer.py - it expects Value objects
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
        
        # Ensure we return a numpy array
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

@dataclass
class ExperimentConfig:
    name: str
    conv_layers: List[int]  # Number of filters per conv layer
    kernel_sizes: List[int]  # Kernel size per conv layer
    pooling_type: str  # 'max' or 'avg'

class CIFAR10Experimenter:
    def __init__(self):
        self.results = {}
        self.models = {}
        self.histories = {}
        
    def load_and_prepare_data(self):
        """Load CIFAR-10 and create train/val/test splits"""
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
        
        # Normalize pixel values
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        # Flatten labels
        y_train = y_train.flatten()
        y_test = y_test.flatten()
        
        # Split training data into train/validation (4:1 ratio)
        train_size = int(0.8 * len(x_train))  # 40k train, 10k validation
        
        # Shuffle before splitting
        indices = np.random.permutation(len(x_train))
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        x_train_final = x_train[train_indices]
        y_train_final = y_train[train_indices]
        x_val = x_train[val_indices]
        y_val = y_train[val_indices]
        
        print(f"Training set: {x_train_final.shape[0]} samples")
        print(f"Validation set: {x_val.shape[0]} samples")
        print(f"Test set: {x_test.shape[0]} samples")
        
        return (x_train_final, y_train_final), (x_val, y_val), (x_test, y_test)
        
    def create_cnn_model(self, config: ExperimentConfig, input_shape=(32, 32, 3)):
        """Create CNN model based on configuration"""
        model = keras.Sequential()
        
        # Add convolutional layers
        for i, (filters, kernel_size) in enumerate(zip(config.conv_layers, config.kernel_sizes)):
            if i == 0:
                model.add(keras.layers.Conv2D(filters, kernel_size, activation='relu', 
                                            input_shape=input_shape, padding='same'))
            else:
                model.add(keras.layers.Conv2D(filters, kernel_size, activation='relu', padding='same'))
                
            # Add pooling layer after each conv layer
            if config.pooling_type == 'max':
                model.add(keras.layers.MaxPooling2D(2, 2))
            else:
                model.add(keras.layers.AveragePooling2D(2, 2))
                
        # Add classification head
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(128, activation='relu'))
        model.add(keras.layers.Dense(10, activation='softmax'))
        
        # Compile model
        model.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
        
        return model
    
    def train_model(self, model, train_data, val_data, epochs=20, batch_size=32):
        """Train the model"""
        x_train, y_train = train_data
        x_val, y_val = val_data
        
        history = model.fit(x_train, y_train,
                          batch_size=batch_size,
                          epochs=epochs,
                          validation_data=(x_val, y_val),
                          verbose=1)
        
        return history
    
    def evaluate_model(self, model, test_data):
        """Evaluate model and return macro F1-score"""
        x_test, y_test = test_data
        
        # Get predictions
        y_pred_proba = model.predict(x_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Calculate macro F1-score
        f1_macro = f1_score(y_test, y_pred, average='macro')
        
        return f1_macro, y_pred
    
    def plot_training_history(self, histories, title_prefix=""):
        """Plot training and validation loss for multiple experiments"""
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 2, 1)
        for name, history in histories.items():
            plt.plot(history.history['loss'], label=f'{name} - Train')
            plt.plot(history.history['val_loss'], label=f'{name} - Val', linestyle='--')
        plt.title(f'{title_prefix} - Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        for name, history in histories.items():
            plt.plot(history.history['accuracy'], label=f'{name} - Train')
            plt.plot(history.history['val_accuracy'], label=f'{name} - Val', linestyle='--')
        plt.title(f'{title_prefix} - Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def run_conv_layers_experiment(self, train_data, val_data, test_data):
        """Experiment with different numbers of convolutional layers"""
        print("=== Experiment: Number of Convolutional Layers ===")
        
        configs = [
            ExperimentConfig("2_conv_layers", [32, 64], [3, 3], "max"),
            ExperimentConfig("3_conv_layers", [32, 64, 128], [3, 3, 3], "max"),
            ExperimentConfig("4_conv_layers", [32, 64, 128, 256], [3, 3, 3, 3], "max")
        ]
        
        results = {}
        histories = {}
        models = {}
        
        for config in configs:
            print(f"\nTraining {config.name}...")
            model = self.create_cnn_model(config)
            history = self.train_model(model, train_data, val_data, epochs=15)
            f1_score_result, _ = self.evaluate_model(model, test_data)
            
            results[config.name] = f1_score_result
            histories[config.name] = history
            models[config.name] = model
            
            print(f"{config.name} - Test Macro F1-Score: {f1_score_result:.4f}")
        
        self.plot_training_history(histories, "Number of Conv Layers")
        
        print("\n=== Results Summary: Number of Conv Layers ===")
        for name, score in results.items():
            print(f"{name}: {score:.4f}")
        
        return results, models
    
    def run_filters_experiment(self, train_data, val_data, test_data):
        """Experiment with different numbers of filters per layer"""
        print("\n=== Experiment: Number of Filters per Layer ===")
        
        configs = [
            ExperimentConfig("small_filters", [16, 32, 64], [3, 3, 3], "max"),
            ExperimentConfig("medium_filters", [32, 64, 128], [3, 3, 3], "max"),
            ExperimentConfig("large_filters", [64, 128, 256], [3, 3, 3], "max")
        ]
        
        results = {}
        histories = {}
        models = {}
        
        for config in configs:
            print(f"\nTraining {config.name}...")
            model = self.create_cnn_model(config)
            history = self.train_model(model, train_data, val_data, epochs=15)
            f1_score_result, _ = self.evaluate_model(model, test_data)
            
            results[config.name] = f1_score_result
            histories[config.name] = history
            models[config.name] = model
            
            print(f"{config.name} - Test Macro F1-Score: {f1_score_result:.4f}")
        
        self.plot_training_history(histories, "Number of Filters")
        
        print("\n=== Results Summary: Number of Filters ===")
        for name, score in results.items():
            print(f"{name}: {score:.4f}")
        
        return results, models
    
    def run_kernel_size_experiment(self, train_data, val_data, test_data):
        """Experiment with different kernel sizes"""
        print("\n=== Experiment: Kernel Sizes ===")
        
        configs = [
            ExperimentConfig("kernel_3x3", [32, 64, 128], [3, 3, 3], "max"),
            ExperimentConfig("kernel_5x5", [32, 64, 128], [5, 5, 5], "max"),
            ExperimentConfig("kernel_mixed", [32, 64, 128], [3, 5, 3], "max")
        ]
        
        results = {}
        histories = {}
        models = {}
        
        for config in configs:
            print(f"\nTraining {config.name}...")
            model = self.create_cnn_model(config)
            history = self.train_model(model, train_data, val_data, epochs=15)
            f1_score_result, _ = self.evaluate_model(model, test_data)
            
            results[config.name] = f1_score_result
            histories[config.name] = history
            models[config.name] = model
            
            print(f"{config.name} - Test Macro F1-Score: {f1_score_result:.4f}")
        
        self.plot_training_history(histories, "Kernel Sizes")
        
        print("\n=== Results Summary: Kernel Sizes ===")
        for name, score in results.items():
            print(f"{name}: {score:.4f}")
        
        return results, models
    
    def run_pooling_experiment(self, train_data, val_data, test_data):
        """Experiment with different pooling types"""
        print("\n=== Experiment: Pooling Types ===")
        
        configs = [
            ExperimentConfig("max_pooling", [32, 64, 128], [3, 3, 3], "max"),
            ExperimentConfig("avg_pooling", [32, 64, 128], [3, 3, 3], "avg")
        ]
        
        results = {}
        histories = {}
        models = {}
        
        for config in configs:
            print(f"\nTraining {config.name}...")
            model = self.create_cnn_model(config)
            history = self.train_model(model, train_data, val_data, epochs=15)
            f1_score_result, _ = self.evaluate_model(model, test_data)
            
            results[config.name] = f1_score_result
            histories[config.name] = history
            models[config.name] = model
            
            print(f"{config.name} - Test Macro F1-Score: {f1_score_result:.4f}")
        
        self.plot_training_history(histories, "Pooling Types")
        
        print("\n=== Results Summary: Pooling Types ===")
        for name, score in results.items():
            print(f"{name}: {score:.4f}")
        
        return results, models
    
    def save_model_weights(self, model, filename):
        """Save model weights"""
        model.save_weights(filename)
        print(f"Model weights saved to {filename}")
    
    def test_from_scratch_implementation(self, keras_model, test_data, model_config):
        """Test the from-scratch implementation against Keras"""
        print("\n=== Testing From-Scratch Implementation ===")
        
        x_test, y_test = test_data
        
        # Create from-scratch model
        scratch_model = CNNFromScratch()
        
        # Add convolutional and pooling layers based on config
        for i, (filters, kernel_size) in enumerate(zip(model_config.conv_layers, model_config.kernel_sizes)):
            scratch_model.add_layer(Conv2D(filters, kernel_size, padding='same'))
            if model_config.pooling_type == 'max':
                scratch_model.add_layer(MaxPool2D(2, 2))
            else:
                scratch_model.add_layer(AvgPool2D(2, 2))
        
        # Add flatten layer
        scratch_model.add_layer(Flatten())
        
        # Calculate the correct flattened size
        input_shape = (32, 32, 3)  # CIFAR-10 image shape
        flattened_size = scratch_model.calculate_flattened_size(
            input_shape, model_config.conv_layers, model_config.kernel_sizes, model_config.pooling_type
        )
        
        print(f"Calculated flattened size: {flattened_size}")
        
        # Add dense layers (using the Layer class from Tugas Besar 1)
        try:
            from Layer import Layer
            from activation import relu, softmax
            
            # Create dense layers with correct input sizes
            dense1 = Layer(flattened_size, 128, activation=relu)
            dense2 = Layer(128, 10, activation=softmax)
            
            scratch_model.add_layer(dense1)
            scratch_model.add_layer(dense2)
            
        except ImportError:
            print("Warning: Could not import Layer or activation modules")
            print("Please ensure Layer.py and activation.py are available")
            return None, None
        
        # Load weights from Keras model
        try:
            scratch_model.load_keras_weights(keras_model)
            print("Successfully loaded weights from Keras model")
        except Exception as e:
            print(f"Error loading weights: {e}")
            return None, None
        
        # Test on a small batch first to verify implementation
        batch_size = 100
        x_batch = x_test[:batch_size]
        y_batch = y_test[:batch_size]
        
        print(f"Testing on batch of {batch_size} samples...")
        
        try:
            # Keras predictions
            keras_pred = keras_model.predict(x_batch, verbose=0)
            keras_pred_classes = np.argmax(keras_pred, axis=1)
            
            # From-scratch predictions
            scratch_pred = scratch_model.predict(x_batch)
            
            # Handle potential shape issues
            if scratch_pred.ndim == 1:
                scratch_pred = scratch_pred.reshape(batch_size, -1)
            
            scratch_pred_classes = np.argmax(scratch_pred, axis=1)
            
            # Compare results
            keras_f1 = f1_score(y_batch, keras_pred_classes, average='macro')
            scratch_f1 = f1_score(y_batch, scratch_pred_classes, average='macro')
            
            print(f"Keras Model F1-Score: {keras_f1:.4f}")
            print(f"From-Scratch Model F1-Score: {scratch_f1:.4f}")
            print(f"Difference: {abs(keras_f1 - scratch_f1):.4f}")
            
            # Check if predictions are similar
            matching_predictions = np.sum(keras_pred_classes == scratch_pred_classes)
            match_percentage = matching_predictions/batch_size*100
            print(f"Matching predictions: {matching_predictions}/{batch_size} ({match_percentage:.2f}%)")
            
            # Additional debugging information
            print(f"Keras predictions shape: {keras_pred.shape}")
            print(f"Scratch predictions shape: {scratch_pred.shape}")
            print(f"Sample keras prediction: {keras_pred[0][:5]}")
            print(f"Sample scratch prediction: {scratch_pred[0][:5]}")
            
            # Test on full test set if small batch works well
            if match_percentage > 90:  # If more than 90% match, test on full set
                print("\nTesting on full test set...")
                keras_pred_full = keras_model.predict(x_test, verbose=0)
                keras_pred_classes_full = np.argmax(keras_pred_full, axis=1)
                
                scratch_pred_full = scratch_model.predict(x_test)
                if scratch_pred_full.ndim == 1:
                    scratch_pred_full = scratch_pred_full.reshape(len(x_test), -1)
                scratch_pred_classes_full = np.argmax(scratch_pred_full, axis=1)
                
                keras_f1_full = f1_score(y_test, keras_pred_classes_full, average='macro')
                scratch_f1_full = f1_score(y_test, scratch_pred_classes_full, average='macro')
                
                print(f"Full Test Set:")
                print(f"Keras Model F1-Score: {keras_f1_full:.4f}")
                print(f"From-Scratch Model F1-Score: {scratch_f1_full:.4f}")
                print(f"Difference: {abs(keras_f1_full - scratch_f1_full):.4f}")
                
                return keras_f1_full, scratch_f1_full
            
            return keras_f1, scratch_f1
            
        except Exception as e:
            print(f"Error during prediction comparison: {e}")
            import traceback
            traceback.print_exc()
            return None, None

def main():
    """Main function to run all experiments"""
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Initialize experimenter
    experimenter = CIFAR10Experimenter()
    
    # Load and prepare data
    print("Loading and preparing CIFAR-10 data...")
    train_data, val_data, test_data = experimenter.load_and_prepare_data()
    
    # Run experiments
    conv_results, conv_models = experimenter.run_conv_layers_experiment(train_data, val_data, test_data)
    filter_results, filter_models = experimenter.run_filters_experiment(train_data, val_data, test_data)
    kernel_results, kernel_models = experimenter.run_kernel_size_experiment(train_data, val_data, test_data)
    pooling_results, pooling_models = experimenter.run_pooling_experiment(train_data, val_data, test_data)
    
    # Save best model weights
    best_model_name = max(conv_results, key=conv_results.get)
    best_model = conv_models[best_model_name]
    experimenter.save_model_weights(best_model, f"best_model_weights_{best_model_name}.h5")
    
    # Test from-scratch implementation
    config = ExperimentConfig("test", [32, 64, 128], [3, 3, 3], "max")
    experimenter.test_from_scratch_implementation(best_model, test_data, config)

if __name__ == "__main__":
    main()