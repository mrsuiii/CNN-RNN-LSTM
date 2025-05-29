import numpy as np
from Value import Value
from activation import tanh, sigmoid
from typing import Callable, List, Optional, Tuple

class RNN:
    def __init__(self, input_size: int, hidden_size: int, activation: Callable[[Value], Value] = tanh, weight_init_func=None):
        """
        Initializes a Simple RNN Layer.

        Args:
            input_size (int): The number of expected features in the input x.
            hidden_size (int): The number of features in the hidden state h.
            activation (Callable[[Value], Value]): The activation function to use. Defaults to tanh.
            weight_init_func (Callable, optional): Function to initialize weights.
                                                 Should take (n_inputs, n_outputs) and return np.ndarray.
                                                 Defaults to a standard random initialization if None.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation = activation

        if weight_init_func is None:
            # Default initialization (glorot/xavier uniform)
            limit_xh = np.sqrt(6.0 / (input_size + hidden_size))
            limit_hh = np.sqrt(6.0 / (hidden_size + hidden_size))
            self.W_xh = Value(np.random.uniform(-limit_xh, limit_xh, (input_size, hidden_size))) # Weights for input x
            self.W_hh = Value(np.random.uniform(-limit_hh, limit_hh, (hidden_size, hidden_size))) # Weights for hidden state h
            self.b_h = Value(np.zeros((1, hidden_size))) # Bias for hidden state
        else:
            self.W_xh = Value(weight_init_func(input_size, hidden_size))
            self.W_hh = Value(weight_init_func(hidden_size, hidden_size))
            self.b_h = Value(np.zeros((1, hidden_size)))


    def parameters(self) -> List[Value]:
        """Returns the list of parameters of the layer."""
        return [self.W_xh, self.W_hh, self.b_h]

    def set_weights(self, W_xh: np.ndarray, W_hh: np.ndarray, b_h: np.ndarray):
        """
        Sets the weights for the SimpleRNN layer.
        Keras kernel (W_xh) has shape (input_dim, units).
        Keras recurrent_kernel (W_hh) has shape (units, units).
        Keras bias (b_h) has shape (units,).

        Our W_xh shape: (input_size, hidden_size)
        Our W_hh shape: (hidden_size, hidden_size)
        Our b_h shape: (1, hidden_size)
        """
        if W_xh.shape != self.W_xh.data.shape:
            raise ValueError(f"Expected W_xh shape {self.W_xh.data.shape} but got {W_xh.shape}")
        if W_hh.shape != self.W_hh.data.shape:
            raise ValueError(f"Expected W_hh shape {self.W_hh.data.shape} but got {W_hh.shape}")
        if b_h.shape != (self.b_h.data.shape[-1],): # Keras bias is 1D
             raise ValueError(f"Expected b_h shape {(self.b_h.data.shape[-1],)} but got {b_h.shape}")

        self.W_xh.data = W_xh
        self.W_hh.data = W_hh
        self.b_h.data = b_h.reshape(self.b_h.data.shape)

    def __call__(self, x_sequence: Value, h_prev: Optional[Value] = None) -> Tuple[Value, Value]:
        """
        Performs the forward pass for a sequence of inputs.

        Args:
            x_sequence (Value): Input sequence. Data shape (batch_size, sequence_length, input_size).
            h_prev (Value, optional): Initial hidden state. Data shape (batch_size, hidden_size).
                                     Defaults to zeros if None.

        Returns:
            Tuple[Value, Value]:
                - outputs_value (Value): Hidden states for each time step. Data shape (batch_size, sequence_length, hidden_size).
                - h_t (Value): The last hidden state. Data shape (batch_size, hidden_size).
        """
        batch_size, sequence_length, _ = x_sequence.data.shape

        if h_prev is None:
            h_prev = Value(np.zeros((batch_size, self.hidden_size)))

        outputs_list = []

        for t in range(sequence_length):
            # Get input for the current time step: x_t
            # x_t data shape: (batch_size, input_size)
            x_t_data = x_sequence.data[:, t, :]
            if x_t_data.ndim == 1: # Ensure x_t_data is 2D for single feature input
                x_t_data = np.atleast_2d(x_t_data)
            if x_t_data.shape[0] != batch_size: # if batch_size is 1 and previous step made it (feature_size,)
                 x_t_data = x_t_data.reshape(batch_size, -1)

            x_t = Value(x_t_data)


            # Hidden state calculation: h_t = activation(x_t @ W_xh + h_prev @ W_hh + b_h)
            term_xh = x_t.matmul(self.W_xh)
            term_hh = h_prev.matmul(self.W_hh)
            h_t = self.activation(term_xh + term_hh + self.b_h)

            outputs_list.append(h_t.data) # Store numpy data for stacking
            h_prev = h_t # Update h_prev for the next time step

        # Stack outputs along the sequence_length dimension
        # outputs_list contains numpy arrays of shape (batch_size, hidden_size)
        stacked_outputs_data = np.stack(outputs_list, axis=1) # (batch_size, sequence_length, hidden_size)
        outputs_value = Value(stacked_outputs_data)

        return outputs_value, h_t # h_t is the last hidden state Value object