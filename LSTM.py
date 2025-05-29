import numpy as np
from Value import Value
from activation import tanh, sigmoid
from typing import Callable, List, Optional, Tuple

class LSTM:
    def __init__(self, input_size: int, hidden_size: int, weight_init_func=None):
        """
        Initializes an LSTM Layer.

        Args:
            input_size (int): The number of expected features in the input x.
            hidden_size (int): The number of features in the hidden state h and cell state c.
            weight_init_func (Callable, optional): Function to initialize weights.
                                                 Should take (n_inputs, n_outputs) and return np.ndarray.
                                                 Defaults to a standard random initialization if None.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size

        if weight_init_func is None:
            # Default initialization (glorot/xavier uniform like)
            def init_weights(n_in, n_out):
                limit = np.sqrt(6.0 / (n_in + n_out))
                return np.random.uniform(-limit, limit, (n_in, n_out))
            self.W_f = Value(init_weights(input_size, hidden_size)) # Forget gate input weights
            self.U_f = Value(init_weights(hidden_size, hidden_size)) # Forget gate hidden weights
            self.b_f = Value(np.zeros((1, hidden_size)))              # Forget gate bias

            self.W_i = Value(init_weights(input_size, hidden_size)) # Input gate input weights
            self.U_i = Value(init_weights(hidden_size, hidden_size)) # Input gate hidden weights
            self.b_i = Value(np.zeros((1, hidden_size)))              # Input gate bias

            self.W_c = Value(init_weights(input_size, hidden_size)) # Cell state candidate input weights
            self.U_c = Value(init_weights(hidden_size, hidden_size)) # Cell state candidate hidden weights
            self.b_c = Value(np.zeros((1, hidden_size)))              # Cell state candidate bias

            self.W_o = Value(init_weights(input_size, hidden_size)) # Output gate input weights
            self.U_o = Value(init_weights(hidden_size, hidden_size)) # Output gate hidden weights
            self.b_o = Value(np.zeros((1, hidden_size)))              # Output gate bias
        else:
            self.W_f = Value(weight_init_func(input_size, hidden_size))
            self.U_f = Value(weight_init_func(hidden_size, hidden_size))
            self.b_f = Value(np.zeros((1, hidden_size)))

            self.W_i = Value(weight_init_func(input_size, hidden_size))
            self.U_i = Value(weight_init_func(hidden_size, hidden_size))
            self.b_i = Value(np.zeros((1, hidden_size)))

            self.W_c = Value(weight_init_func(input_size, hidden_size))
            self.U_c = Value(weight_init_func(hidden_size, hidden_size))
            self.b_c = Value(np.zeros((1, hidden_size)))

            self.W_o = Value(weight_init_func(input_size, hidden_size))
            self.U_o = Value(weight_init_func(hidden_size, hidden_size))
            self.b_o = Value(np.zeros((1, hidden_size)))


    def parameters(self) -> List[Value]:
        """Returns the list of parameters of the layer."""
        return [
            self.W_f, self.U_f, self.b_f,
            self.W_i, self.U_i, self.b_i,
            self.W_c, self.U_c, self.b_c,
            self.W_o, self.U_o, self.b_o,
        ]
    def set_weights(self, W_all: np.ndarray, U_all: np.ndarray, b_all: np.ndarray):
        """
        Sets the weights for the LSTM layer from Keras format.
        Keras kernel (W_all) has shape (input_dim, 4 * units).
        Keras recurrent_kernel (U_all) has shape (units, 4 * units).
        Keras bias (b_all) has shape (4 * units,).
        Order of gates in Keras is typically i, f, c, o.

        Our W_gate shape: (input_size, hidden_size)
        Our U_gate shape: (hidden_size, hidden_size)
        Our b_gate shape: (1, hidden_size)
        """
        hidden_size = self.hidden_size
        input_size = self.input_size # from __init__

        expected_W_all_shape = (input_size, 4 * hidden_size)
        if W_all.shape != expected_W_all_shape:
            raise ValueError(f"Expected W_all shape {expected_W_all_shape} but got {W_all.shape}")

        W_i_k, W_f_k, W_c_k, W_o_k = np.split(W_all, 4, axis=1)
        self.W_i.data = W_i_k
        self.W_f.data = W_f_k
        self.W_c.data = W_c_k
        self.W_o.data = W_o_k

        expected_U_all_shape = (hidden_size, 4 * hidden_size)
        if U_all.shape != expected_U_all_shape:
            raise ValueError(f"Expected U_all shape {expected_U_all_shape} but got {U_all.shape}")

        U_i_k, U_f_k, U_c_k, U_o_k = np.split(U_all, 4, axis=1)
        self.U_i.data = U_i_k
        self.U_f.data = U_f_k
        self.U_c.data = U_c_k
        self.U_o.data = U_o_k

        expected_b_all_shape = (4 * hidden_size,)
        if b_all.shape != expected_b_all_shape:
             raise ValueError(f"Expected b_all shape {expected_b_all_shape} but got {b_all.shape}")

        b_i_k, b_f_k, b_c_k, b_o_k = np.split(b_all, 4, axis=0)
        self.b_i.data = b_i_k.reshape(self.b_i.data.shape)
        self.b_f.data = b_f_k.reshape(self.b_f.data.shape)
        self.b_c.data = b_c_k.reshape(self.b_c.data.shape)
        self.b_o.data = b_o_k.reshape(self.b_o.data.shape)

    def __call__(self, x_sequence: Value, initial_states: Optional[Tuple[Value, Value]] = None) -> Tuple[Value, Tuple[Value, Value]]:
        """
        Performs the forward pass for a sequence of inputs.

        Args:
            x_sequence (Value): Input sequence. Data shape (batch_size, sequence_length, input_size).
            initial_states (Tuple[Value, Value], optional): Tuple of (h_prev, c_prev).
                h_prev: Initial hidden state, data shape (batch_size, hidden_size).
                c_prev: Initial cell state, data shape (batch_size, hidden_size).
                Defaults to zeros if None.

        Returns:
            Tuple[Value, Tuple[Value, Value]]:
                - outputs_value (Value): Hidden states for each time step. Data shape (batch_size, sequence_length, hidden_size).
                - (h_t, c_t) (Tuple[Value, Value]): The last hidden state and cell state.
        """
        batch_size, sequence_length, _ = x_sequence.data.shape

        if initial_states is None:
            h_prev = Value(np.zeros((batch_size, self.hidden_size)))
            c_prev = Value(np.zeros((batch_size, self.hidden_size)))
        else:
            h_prev, c_prev = initial_states

        outputs_list = [] # To store numpy arrays of h_t for each time step

        for t in range(sequence_length):
            # Get input for the current time step: x_t
            # x_t data shape: (batch_size, input_size)
            x_t_data = x_sequence.data[:, t, :]
            if x_t_data.ndim == 1: # Ensure x_t_data is 2D
                x_t_data = np.atleast_2d(x_t_data)
            if x_t_data.shape[0] != batch_size: # if batch_size is 1 and previous step made it (feature_size,)
                 x_t_data = x_t_data.reshape(batch_size, -1)
            x_t = Value(x_t_data)


            # Forget gate: f_t = sigmoid(x_t @ W_f + h_prev @ U_f + b_f)
            f_t = sigmoid(x_t.matmul(self.W_f) + h_prev.matmul(self.U_f) + self.b_f)

            # Input gate: i_t = sigmoid(x_t @ W_i + h_prev @ U_i + b_i)
            i_t = sigmoid(x_t.matmul(self.W_i) + h_prev.matmul(self.U_i) + self.b_i)

            # Candidate cell state: c_tilde_t = tanh(x_t @ W_c + h_prev @ U_c + b_c)
            c_tilde_t = tanh(x_t.matmul(self.W_c) + h_prev.matmul(self.U_c) + self.b_c)

            # Cell state: c_t = f_t * c_prev + i_t * c_tilde_t
            c_t = f_t * c_prev + i_t * c_tilde_t

            # Output gate: o_t = sigmoid(x_t @ W_o + h_prev @ U_o + b_o)
            o_t = sigmoid(x_t.matmul(self.W_o) + h_prev.matmul(self.U_o) + self.b_o)

            # Hidden state: h_t = o_t * tanh(c_t)
            h_t = o_t * tanh(c_t)

            outputs_list.append(h_t.data) # Store numpy data
            h_prev = h_t
            c_prev = c_t

        # Stack outputs along the sequence_length dimension
        stacked_outputs_data = np.stack(outputs_list, axis=1) # (batch_size, sequence_length, hidden_size)
        outputs_value = Value(stacked_outputs_data)

        return outputs_value, (h_t, c_t) # h_t and c_t are the last states (Value objects)