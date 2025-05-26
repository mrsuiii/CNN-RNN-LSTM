# Pastikan file recurrent_layers.py, Value.py, Layer.py, activation.py berada di direktori yang sama
# atau dapat diimpor dengan benar.

import numpy as np
from Value import Value
# Asumsikan recurrent_layers.py sekarang berisi SimpleRNNLayer dan LSTMLayer yang telah Anda buat
from LSTM import LSTM
from RNN import RNN
from Layer import Layer # Implementasi Dense Layer dari Tubes 1
from activation import softmax, tanh #
from typing import List, Union, Optional, Tuple

class StackedSequenceClassifier:
    def __init__(self,
                 input_size: int,
                 recurrent_hidden_size: int,
                 num_classes: int,
                 num_recurrent_layers: int = 1, # Jumlah layer rekuren yang ditumpuk
                 recurrent_type: str = 'rnn', # 'rnn' atau 'lstm'
                 rnn_activation: callable = tanh,
                 weight_init_func=None,
                 return_sequences_for_all_but_last_recurrent: bool = True): # Kontrol output sekuens antar layer rekuren
        """
        Initializes a Stacked Sequence Classifier model.

        Args:
            input_size (int): Number of features in the input sequence elements for the first recurrent layer.
            recurrent_hidden_size (int): Number of features in the recurrent layer's hidden state.
                                         (Untuk kesederhanaan, semua layer rekuren menggunakan ukuran ini).
            num_classes (int): Number of output classes for classification.
            num_recurrent_layers (int): Number of recurrent layers to stack.
            recurrent_type (str): Type of recurrent layer to use ('rnn' or 'lstm').
            rnn_activation (callable): Activation function for SimpleRNNLayer (if used).
            weight_init_func (callable, optional): Function to initialize weights for layers.
            return_sequences_for_all_but_last_recurrent (bool): If True, recurrent layers (except the last one)
                                                                will pass their full sequence output to the next
                                                                recurrent layer. The last recurrent layer will
                                                                only pass its final hidden state to the classifier.
                                                                (Implementasi pada SimpleRNNLayer/LSTMLayer
                                                                 perlu mendukung ini secara eksplisit atau disesuaikan).
        """
        if num_recurrent_layers < 1:
            raise ValueError("num_recurrent_layers must be at least 1.")

        self.input_size = input_size
        self.recurrent_hidden_size = recurrent_hidden_size
        self.num_classes = num_classes
        self.num_recurrent_layers = num_recurrent_layers
        self.recurrent_type = recurrent_type.lower()
        self.return_sequences = return_sequences_for_all_but_last_recurrent

        self.recurrent_layers = []
        current_input_size = input_size

        for i in range(num_recurrent_layers):
            if self.recurrent_type == 'rnn':
                layer = RNN(
                    input_size=current_input_size,
                    hidden_size=recurrent_hidden_size,
                    activation=rnn_activation,
                    weight_init_func=weight_init_func
                )
            elif self.recurrent_type == 'lstm':
                layer = LSTM(
                    input_size=current_input_size,
                    hidden_size=recurrent_hidden_size,
                    weight_init_func=weight_init_func
                )
            else:
                raise ValueError("Invalid recurrent_type. Choose 'rnn' or 'lstm'.")
            self.recurrent_layers.append(layer)
            current_input_size = recurrent_hidden_size # Input untuk layer rekuren berikutnya adalah output hidden state dari layer saat ini

        # Dense layer for classification
        self.classification_layer = Layer(
            n_inputs=recurrent_hidden_size, # Inputnya adalah hidden state dari layer rekuren terakhir
            n_neurons=num_classes,
            activation=softmax,
            weight_init=weight_init_func if weight_init_func else lambda n_in, n_out: np.random.randn(n_out, n_in) * 0.01
        )

    def __call__(self, x_sequence: Value, initial_recurrent_states_list: Optional[List[Union[Value, Tuple[Value, Value]]]] = None) -> Value:
        """
        Performs the forward pass for sequence classification with stacked recurrent layers.

        Args:
            x_sequence (Value): Input sequence of shape (batch_size, sequence_length, input_size).
            initial_recurrent_states_list (Optional[List]): List of initial hidden states for each recurrent layer.
                                                            Each element is h_prev for RNN or (h_prev, c_prev) for LSTM.
                                                            If None, defaults to zeros for all layers.

        Returns:
            Value: Output probabilities for each class, shape (batch_size, num_classes).
        """
        current_sequence_input = x_sequence
        last_hidden_state_from_layer = None # Untuk menyimpan hidden state terakhir dari setiap layer

        if initial_recurrent_states_list is None:
            initial_recurrent_states_list = [None] * self.num_recurrent_layers
        elif len(initial_recurrent_states_list) != self.num_recurrent_layers:
            raise ValueError(f"Expected {self.num_recurrent_layers} initial states, got {len(initial_recurrent_states_list)}")

        for i in range(self.num_recurrent_layers):
            recurrent_layer = self.recurrent_layers[i]
            initial_states_for_this_layer = initial_recurrent_states_list[i]

            if isinstance(recurrent_layer, RNN):
                outputs_sequence, last_hidden_state_from_layer = recurrent_layer(
                    current_sequence_input,
                    h_prev=initial_states_for_this_layer
                )
            elif isinstance(recurrent_layer, LSTM):
                outputs_sequence, (last_hidden_state_from_layer, _) = recurrent_layer( 
                    current_sequence_input,
                    initial_states=initial_states_for_this_layer
                )
            else:
                raise TypeError("Recurrent layer not properly initialized.")

            if i < self.num_recurrent_layers - 1: # Jika ada layer rekuren setelah ini
                if self.return_sequences:
                     # outputs_sequence (Value object) dari layer saat ini menjadi input untuk layer berikutnya
                    current_sequence_input = outputs_sequence
                else:
                    
                    current_sequence_input = outputs_sequence
            else: # Ini adalah layer rekuren terakhir
                # Hanya hidden state terakhir yang akan diteruskan ke classification_layer
                pass


        # last_hidden_state_from_layer sekarang adalah hidden state terakhir dari layer rekuren paling atas/terakhir.
        output_probabilities = self.classification_layer(last_hidden_state_from_layer)

        return output_probabilities

    def parameters(self) -> List[Value]:
        """Returns the list of all trainable parameters in the model."""
        params = []
        for r_layer in self.recurrent_layers:
            params.extend(r_layer.parameters())
        params.extend(self.classification_layer.parameters())
        return params

# --- Contoh Penggunaan ---
if __name__ == '__main__':
    batch_s = 2
    seq_len = 5
    feat_size = 10
    rnn_hidden = 20
    n_classes = 3
    num_stacked_recurrent_layers = 2 # Misal kita mau 2 layer RNN ditumpuk

    dummy_input_data = np.random.rand(batch_s, seq_len, feat_size)
    input_sequence_value = Value(dummy_input_data)

    print(f"--- Stacked ({num_stacked_recurrent_layers}-layer) RNN Classifier Example ---")
    stacked_rnn_classifier = StackedSequenceClassifier(
        input_size=feat_size,
        recurrent_hidden_size=rnn_hidden,
        num_classes=n_classes,
        num_recurrent_layers=num_stacked_recurrent_layers,
        recurrent_type='rnn'
    )
    # Untuk initial states list, jika ada 2 layer, kita butuh list berisi 2 None (atau 2 state)
    # Jika initial_recurrent_states_list tidak diberikan, akan default ke [None, None, ...]
    rnn_predictions = stacked_rnn_classifier(input_sequence_value)
    print("Stacked RNN Predictions (data shape):", rnn_predictions.data.shape)
    print("Stacked RNN Predictions (sample):", rnn_predictions.data[0])
    print("Total parameters in Stacked RNN Classifier:", len(stacked_rnn_classifier.parameters()))


    print(f"\n--- Stacked ({num_stacked_recurrent_layers}-layer) LSTM Classifier Example ---")
    stacked_lstm_classifier = StackedSequenceClassifier(
        input_size=feat_size,
        recurrent_hidden_size=rnn_hidden,
        num_classes=n_classes,
        num_recurrent_layers=num_stacked_recurrent_layers,
        recurrent_type='lstm'
    )
    # Membuat list initial states (misalnya, semua nol)
    # initial_lstm_states = []
    # for _ in range(num_stacked_recurrent_layers):
    #     h0 = Value(np.zeros((batch_s, rnn_hidden)))
    #     c0 = Value(np.zeros((batch_s, rnn_hidden)))
    #     initial_lstm_states.append((h0, c0))
    # lstm_predictions = stacked_lstm_classifier(input_sequence_value, initial_recurrent_states_list=initial_lstm_states)

    lstm_predictions = stacked_lstm_classifier(input_sequence_value) # Menggunakan default None untuk initial states
    print("Stacked LSTM Predictions (data shape):", lstm_predictions.data.shape)
    print("Stacked LSTM Predictions (sample):", lstm_predictions.data[0])
    print("Total parameters in Stacked LSTM Classifier:", len(stacked_lstm_classifier.parameters()))