from Value import Value
from RNN import RNN
from LSTM import LSTM
import numpy as np

# Example Usage (assuming Value, activation modules are available)

# --- SimpleRNNLayer Example ---
batch_size = 2
sequence_length = 3
input_features = 4
hidden_features_rnn = 5

# Create a dummy input sequence
# x_seq_data should be (batch_size, sequence_length, input_features)
dummy_x_sequence_data_rnn = np.random.rand(batch_size, sequence_length, input_features)
x_sequence_rnn = Value(dummy_x_sequence_data_rnn) # Note: Value wraps 2D, so this direct wrap might need adjustment based on Value's design for 3D.
                                              # The layers expect to iterate over sequence_length internally.

rnn_layer = RNN(input_size=input_features, hidden_size=hidden_features_rnn)
# For the __call__ method, we directly pass the x_sequence_rnn Value object
# The layer internally handles slicing x_sequence_rnn.data
rnn_outputs, rnn_last_hidden = rnn_layer(x_sequence_rnn)

print("--- SimpleRNN ---")
print("RNN Outputs (data shape):", rnn_outputs.data.shape) # (batch_size, sequence_length, hidden_features_rnn)
print("RNN Last Hidden State (data shape):", rnn_last_hidden.data.shape) # (batch_size, hidden_features_rnn)
# print("RNN Last Hidden State grad_fn:", rnn_last_hidden._op) # To check graph connection

# --- LSTMLayer Example ---
hidden_features_lstm = 6
dummy_x_sequence_data_lstm = np.random.rand(batch_size, sequence_length, input_features)
x_sequence_lstm = Value(dummy_x_sequence_data_lstm)

lstm_layer = LSTM(input_size=input_features, hidden_size=hidden_features_lstm)
lstm_outputs, (lstm_last_hidden, lstm_last_cell) = lstm_layer(x_sequence_lstm)

print("\n--- LSTM ---")
print("LSTM Outputs (data shape):", lstm_outputs.data.shape) # (batch_size, sequence_length, hidden_features_lstm)
print("LSTM Last Hidden State (data shape):", lstm_last_hidden.data.shape) # (batch_size, hidden_features_lstm)
print("LSTM Last Cell State (data shape):", lstm_last_cell.data.shape) # (batch_size, hidden_features_lstm)
# print("LSTM Last Hidden State grad_fn:", lstm_last_hidden._op)