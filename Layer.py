from activation import linear
from init import he_init 
import numpy as np
from Value import Value
from typing import List, Optional, Dict, Callable

class Layer:
    def __init__(self, n_inputs: int, n_neurons: int, 
                 activation: Callable[[Value], Value] = linear,
                 weight_init: Callable[..., np.ndarray] = lambda n_in, n_out: np.zeros((n_out, n_in)),
                 weight_init_kwargs: Optional[Dict] = None,
                 rmsnorm: bool = False,
                 eps: float = 1e-8):
        if weight_init_kwargs is None:
            weight_init_kwargs = {}

        self.activation = activation
        self.W = Value(data=weight_init(n_inputs, n_neurons, **weight_init_kwargs))
        self.b = Value(data=np.zeros(n_neurons))
        self.rmsnorm = rmsnorm
        self.eps = eps
        if self.rmsnorm:
            self.gamma = Value(np.random.randn(1, n_neurons))  
        else:
            self.gamma = None

    def parameters(self) -> List[Value]:
        param = [self.W, self.b]
        if self.gamma is not None:
            param.append(self.gamma)
        return param

    def rmsnorm_func(self, x: Value) -> Value:
        ms = (x * x).sum(axis=1, keepdims=True) / x.data.shape[1]
        rms = (ms + self.eps).sqrt()
        normed = x / rms
        if self.gamma is not None:
            normed = normed * self.gamma
        return normed
    def set_weights(self, weights: np.ndarray, biases: np.ndarray):
        """
        Sets the weights and biases for the layer.
        Keras Dense kernel (weights) has shape (input_dim, units).
        self.W has shape (units, input_dim). So, transpose is needed.
        Keras Dense biases has shape (units,).
        self.b has shape (units,) or (1, units).
        """
        if weights.shape != (self.W.data.shape[1], self.W.data.shape[0]):
            raise ValueError(f"Expected weights shape {(self.W.data.shape[1], self.W.data.shape[0])} but got {weights.shape}")
        if biases.shape != self.b.data.shape and biases.shape != (self.b.data.shape[-1],):
             raise ValueError(f"Expected biases shape {self.b.data.shape} or {(self.b.data.shape[-1],)} but got {biases.shape}")

        self.W.data = weights.T # Keras: (input_dim, units) -> self.W: (units, input_dim)
        self.b.data = biases.reshape(self.b.data.shape)
        
    def __call__(self, x: Value) -> Value:
        z = x.matmul(self.W.T) + self.b
        if self.rmsnorm:
            z = self.rmsnorm_func(z)
        return self.activation(z)
