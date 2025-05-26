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
            x_sequence (Value): Input sequence of shape (batch_size, sequence_length, input_size).
                                Note: The Value class currently handles 2D numpy arrays.
                                For sequence processing, we'll iterate through the sequence dimension.
                                The input for a single time step x_t should be (batch_size, input_size).
            h_prev (Value, optional): Initial hidden state of shape (batch_size, hidden_size).
                                     Defaults to zeros if None.

        Returns:
            Tuple[Value, Value]:
                - outputs (Value): Hidden states for each time step, shape (batch_size, sequence_length, hidden_size).
                                   (Currently, this will be a list of Value objects, each for a time step's output)
                - h_t (Value): The last hidden state, shape (batch_size, hidden_size).
        """
        # x_sequence.data is expected to be (batch_size, sequence_length, input_size)
        # However, Value objects wrap 2D arrays. We'll iterate over sequence_length.

        batch_size, sequence_length, _ = x_sequence.data.shape

        if h_prev is None:
            h_prev = Value(np.zeros((batch_size, self.hidden_size)))

        outputs_list = []

        for t in range(sequence_length):
            # Get input for the current time step: x_t
            # x_t should be a Value object with data of shape (batch_size, input_size)
            x_t = Value(x_sequence.data[:, t, :]) # Slicing numpy data and wrapping in new Value

            # Hidden state calculation: h_t = activation(x_t @ W_xh + h_prev @ W_hh + b_h)
            term_xh = x_t.matmul(self.W_xh)
            term_hh = h_prev.matmul(self.W_hh)
            h_t = self.activation(term_xh + term_hh + self.b_h)

            outputs_list.append(h_t)
            h_prev = h_t # Update h_prev for the next time step

        # To return a single Value object for all outputs, we'd need to stack them.
        # The Value class doesn't have a direct stacking op that preserves graph for list of Values.
        # For now, returning the list of Value objects (one per timestep) and the final hidden state.
        # Or, more practically for now, we can concatenate the numpy data and wrap in a new Value
        # This is tricky for autograd if not handled carefully.
        # For a simple forward pass as requested, we can create a new Value from concatenated data.
        # However, for backprop through time, each h_t needs to be part of the graph.

        # Let's return the final hidden state and the list of hidden states for each step.
        # If a single tensor output is needed for 'outputs', further modification for stacking `Value` objects
        # or a different approach for handling sequences within `Value` would be required.
        # For now, let's create a new Value object from the stacked numpy arrays of the outputs
        # This is a simplification for forward pass only.
        stacked_outputs_data = np.stack([out.data for out in outputs_list], axis=1) # (batch_size, sequence_length, hidden_size)
        outputs_value = Value(stacked_outputs_data) # This Value won't be part of the original graph for BPTT

        return outputs_value, h_t # h_t is the last hidden state from the graph

