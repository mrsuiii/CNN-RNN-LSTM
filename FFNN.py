from Layer import Layer
from typing import List, Callable
from Value import Value
from init import zero_init
from activation import activations_map
from loss import loss_fn_map
import numpy as np
import json

class FFNN:
    def __init__(
        self,
        layers : list[Layer] = None,
        layer_sizes: List[int] = None,
        activations: List[Callable[[Value], Value]]= None,
        loss_fn: Callable[[Value, Value], Value] = None,
        weight_init: Callable[[int, int], Value] = zero_init,
        lr: float = 0.01,
        lambda_l1: float = 0.0,
        lambda_l2: float = 0.0
    ):
        # assert len(activations) == len(layer_sizes) - 1, "Each layer (except input) must have an activation function"

        self.layers = [
            Layer(
                n_inputs=layer_sizes[i],
                n_neurons=layer_sizes[i + 1],
                activation=activations[i],
                weight_init=weight_init
            )
            for i in range(len(layer_sizes) - 1)
        ] if layers is None else layers
        self.learning_rate = lr
        self.loss_fn = loss_fn
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2

    def __call__(self, x: Value) -> Value:
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> List[Value]:
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def backward(self, loss: Value):
        L_new = loss

        for layer in self.layers:
            for param in layer.parameters():
                if self.lambda_l1 > 0:
                    L_new += self.lambda_l1 * param.abs().sum()

                if self.lambda_l2 > 0:
                    L_new += self.lambda_l2 * (param * param).sum()

        L_new.backward()

    def update_weights(self):
        for param in self.parameters():
            grad = param.grad
            
            if grad.shape != param.data.shape:
                grad = grad.sum(axis=0)
            
            param.data -= self.learning_rate * grad


    def train(
        self, 
        training_data, 
        training_target, 
        max_epoch, 
        error_threshold,
        batch_size,  # parameter baru
        validation_data=None, 
        validation_target=None, 
        verbose=False
    ):
        """
        Train the neural network with optional validation.
        
        Parameters:
        - training_data: Input training data
        - training_target: Target values for training data
        - max_epoch: Maximum number of training epochs
        - error_threshold: Stopping criterion for training loss
        - batch_size: Size of each mini-batch for training
        - validation_data: Optional validation input data
        - validation_target: Optional validation target data
        - verbose: If True, print training progress
        
        Returns:
        - Dictionary containing training and validation loss history
        """
        training_loss_history = []
        validation_loss_history = []
        
        for epoch in range(max_epoch):
            training_loss = self._train_epoch(training_data, training_target, batch_size)
            training_loss_history.append(training_loss)
            
            validation_loss = None
            if validation_data is not None and validation_target is not None:
                validation_loss = self._validate(validation_data, validation_target)
                validation_loss_history.append(validation_loss)
            
            if verbose:
                output_str = f"Epoch {epoch + 1}/{max_epoch}: Training Loss = {training_loss}"
                if validation_loss is not None:
                    output_str += f", Validation Loss = {validation_loss}"
                print(output_str)
            
            if training_loss <= error_threshold:
                break
        
        return {
            'training_loss_history': training_loss_history,
            'validation_loss_history': validation_loss_history
        }

    def _train_epoch(self, training_data : Value, training_target:Value, batch_size: int):
        total_loss = 0
        num_samples = len(training_data)
        num_batches = (num_samples + batch_size - 1) // batch_size  # menangani batch terakhir yang mungkin kurang dari batch_size

        for i in range(0, num_samples, batch_size):

            batch_data = training_data[i:i+batch_size]
            batch_target = training_target[i:i+batch_size]
            
            # Lakukan forward pass untuk batch
            output = self(batch_data)
            
            # Hitung loss untuk batch
            loss = self.loss_fn(output, batch_target)
            total_loss += loss.data
            
            # Backward pass dan update weights
            # Reset gradien untuk setiap parameter
            for param in self.parameters():
                param.zero_grad()
            self.backward(loss)
            self.update_weights()
        
        # Rata-rata loss per batch
        average_loss = total_loss / num_batches
        return average_loss

    def _validate(self, validation_data: Value, validation_target : Value):
        total_loss = 0
        for x, y in zip(validation_data, validation_target):
            output = self(x)
            loss = self.loss_fn(output, y)
            total_loss += loss.data
        return total_loss / len(validation_data)
    
    def save(self, file_path: str):
        model_data = {
            "layer_sizes": [layer.W.data.shape[1] for layer in self.layers] + [self.layers[-1].W.data.shape[0]],
            "weights": [layer.W.data.tolist() for layer in self.layers],
            "biases": [layer.b.data.tolist() for layer in self.layers],
            "rmsnorm": [layer.rmsnorm for layer in self.layers],
            "gamma": [layer.gamma.data.tolist() if layer.rmsnorm else None for layer in self.layers],
            "activations": [layer.activation.__name__ for layer in self.layers],
            "loss_fn": self.loss_fn.__name__ if self.loss_fn else None,
            "learning_rate": self.learning_rate
        }
        with open(file_path, "w") as f:
            json.dump(model_data, f, indent=4)

    @classmethod
    def load(cls, file_path: str):
        with open(file_path, "r") as f:
            model_data = json.load(f)

        layer_sizes = model_data["layer_sizes"]
        activations = [activations_map[name] for name in model_data["activations"]]
        loss_fn = loss_fn_map.get(model_data["loss_fn"], None)
        layers = []

        for i in range(len(layer_sizes) - 1):
            layer = Layer(
                n_inputs=layer_sizes[i],
                n_neurons=layer_sizes[i + 1],
                activation=activations[i],
                weight_init=zero_init,
                rmsnorm=model_data["rmsnorm"][i]
            )
            layer.W.data = np.array(model_data["weights"][i])
            layer.b.data = np.array(model_data["biases"][i])
            if model_data["rmsnorm"][i]:
                layer.gamma.data = np.array(model_data["gamma"][i])
            layers.append(layer)

        return cls(layers=layers, loss_fn=loss_fn, lr=model_data["learning_rate"])