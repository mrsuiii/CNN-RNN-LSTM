import h5py
import tensorflow as tf # Untuk membuat model Keras dummy dan menyimpannya sebagai H5
import numpy as np
from Value import Value
from LSTM import LSTM
from RNN import RNN 
# Asumsikan kelas Layer, SimpleRNNLayer, LSTMLayer, (EmbeddingLayer jika ada),
# dan StackedSequenceClassifier sudah didefinisikan dan memiliki metode set_weights.

# --- Contoh Membuat dan Menyimpan Model Keras (untuk pengujian) ---
def create_and_save_keras_model_for_testing(h5_path="keras_model.h5"):
    keras_vocab_size = 20
    keras_embedding_dim = 10
    keras_seq_len = 5
    keras_lstm_units = 8
    keras_dense_units = 3 # num_classes

    keras_input = tf.keras.layers.Input(shape=(keras_seq_len,), dtype='int32', name="input_tokens")
    x = tf.keras.layers.Embedding(
        input_dim=keras_vocab_size,
        output_dim=keras_embedding_dim,
        name="embedding"
    )(keras_input)
    # Layer LSTM pertama
    x = tf.keras.layers.LSTM(
        units=keras_lstm_units,
        return_sequences=True, # Karena ada LSTM layer berikutnya
        name="lstm_1"
    )(x)
    # Layer LSTM kedua
    lstm_output = tf.keras.layers.LSTM(
        units=keras_lstm_units,
        return_sequences=False, # Hanya hidden state terakhir
        name="lstm_2"
    )(x)
    output = tf.keras.layers.Dense(units=keras_dense_units, activation='softmax', name="classifier_dense")(lstm_output)

    model = tf.keras.Model(inputs=keras_input, outputs=output)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    # Dummy fit untuk inisialisasi bobot (opsional, tapi Keras butuh build)
    dummy_data = np.random.randint(0, keras_vocab_size, size=(2, keras_seq_len))
    dummy_labels = np.random.randint(0, keras_dense_units, size=(2,))
    model.fit(dummy_data, dummy_labels, epochs=1, verbose=0)

    model.save_weights(h5_path)
    print(f"Keras model with 2 LSTM layers saved to {h5_path}")
    return model


def load_keras_weights_to_scratch_model(scratch_model, keras_model_h5_path: str, keras_layer_names: list):
    """
    Loads weights from a Keras H5 file into a scratch model.

    Args:
        scratch_model: An instance of your scratch model (e.g., StackedSequenceClassifier).
        keras_model_h5_path (str): Path to the Keras .h5 weights file.
        keras_layer_names (list): List of Keras layer names IN THE ORDER they appear
                                   in the Keras model AND correspond to scratch_model layers.
                                   Example: ["embedding", "lstm_1", "lstm_2", "classifier_dense"]
    """
    h5_file = h5py.File(keras_model_h5_path, 'r')
    
    keras_weight_idx = 0 # Untuk mengambil nama layer Keras secara berurutan

    # 1. Embedding Layer (jika scratch_model Anda memilikinya dan Keras juga)
    if hasattr(scratch_model, 'embedding_layer') and scratch_model.embedding_layer is not None:
        keras_emb_name = keras_layer_names[keras_weight_idx]
        print(f"Loading weights for Embedding Layer from Keras layer: {keras_emb_name}")
        try:
            # Keras Embedding layer memiliki grup dengan namanya sendiri, lalu 'embedding_matrix:0' atau serupa
            # atau terkadang langsung di bawah nama layer ada 'embeddings:0' atau 'weight_names'
            # Cara paling umum adalah mengambil dataset pertama di grup layer embedding.
            # Cek nama dataset di dalam grup layer
            # print(list(h5_file[keras_emb_name].attrs['weight_names'])) -> ['embedding_1/embeddings:0']
            # Nama weight di Keras bisa jadi 'embeddings:0' atau 'kernel:0' atau semacamnya.
            # Biasanya hanya ada satu set bobot untuk embedding.
            # Kita cari dataset di grup layer tsb.
            keras_embedding_weights = None
            if keras_emb_name in h5_file:
                group = h5_file[keras_emb_name]
                # Cari dataset di dalam grup (biasanya hanya satu)
                for key in group.keys():
                    if isinstance(group[key], h5py.Dataset):
                        keras_embedding_weights = group[key][:]
                        break
            
            if keras_embedding_weights is None and f"{keras_emb_name}/{keras_emb_name}" in h5_file: # struktur lama
                 keras_embedding_weights = h5_file[f"{keras_emb_name}/{keras_emb_name}/embeddings:0"][:]
            elif keras_embedding_weights is None: # struktur Keras 3
                # Nama bobot bisa jadi ada di atribut 'weight_names'
                weight_name_key = list(h5_file[keras_emb_name].attrs['weight_names'])[0] # misal 'embedding_1/embeddings:0'
                keras_embedding_weights = h5_file[keras_emb_name][weight_name_key.split('/')[-1]][:]


            if keras_embedding_weights is not None:
                scratch_model.embedding_layer.set_weights(keras_embedding_weights)
                print(f"  Successfully loaded embedding weights. Shape: {keras_embedding_weights.shape}")
            else:
                print(f"  ERROR: Could not find weights for Keras embedding layer {keras_emb_name}")

        except Exception as e:
            print(f"  ERROR loading weights for Embedding layer {keras_emb_name}: {e}")
        keras_weight_idx += 1


    # 2. Recurrent Layers (LSTM/SimpleRNN)
    for i, r_layer_scratch in enumerate(scratch_model.recurrent_layers):
        if keras_weight_idx >= len(keras_layer_names):
            print(f"Warning: Not enough Keras layer names provided for all scratch recurrent layers.")
            break
        keras_rnn_name = keras_layer_names[keras_weight_idx]
        print(f"Loading weights for Scratch Recurrent Layer {i+1} from Keras layer: {keras_rnn_name}")
        try:
            # Bobot Keras RNN/LSTM biasanya ada 3: kernel, recurrent_kernel, bias
            # Nama bisa jadi 'kernel:0', 'recurrent_kernel:0', 'bias:0' di dalam grup layer
            # Keras 3 menyimpan nama weight di atribut
            if keras_rnn_name not in h5_file:
                print(f"  ERROR: Keras layer group '{keras_rnn_name}' not found in H5 file.")
                keras_weight_idx += 1
                continue

            keras_weights_list = []
            if 'weight_names' in h5_file[keras_rnn_name].attrs: # Keras 3 style
                ordered_weight_keys = [name.split('/')[-1] for name in h5_file[keras_rnn_name].attrs['weight_names']]
                for key in ordered_weight_keys:
                    keras_weights_list.append(h5_file[keras_rnn_name][key][:])
            else: # Gaya lama, mungkin perlu diurutkan atau diambil berdasarkan nama standar
                 # Ini mungkin perlu disesuaikan tergantung versi Keras yang menyimpan H5
                keras_weights_list.append(h5_file[keras_rnn_name][f'{keras_rnn_name}/kernel:0'][:])
                keras_weights_list.append(h5_file[keras_rnn_name][f'{keras_rnn_name}/recurrent_kernel:0'][:])
                keras_weights_list.append(h5_file[keras_rnn_name][f'{keras_rnn_name}/bias:0'][:])


            if isinstance(r_layer_scratch, RNN):
                if len(keras_weights_list) == 3:
                    W_xh_k, W_hh_k, b_h_k = keras_weights_list[0], keras_weights_list[1], keras_weights_list[2]
                    r_layer_scratch.set_weights(W_xh_k, W_hh_k, b_h_k)
                    print(f"  Successfully loaded SimpleRNN weights. Shapes: W_xh={W_xh_k.shape}, W_hh={W_hh_k.shape}, b_h={b_h_k.shape}")
                else:
                    print(f"  ERROR: Expected 3 weight arrays for SimpleRNN, got {len(keras_weights_list)}")
            elif isinstance(r_layer_scratch, LSTM):
                if len(keras_weights_list) == 3:
                    W_all_k, U_all_k, b_all_k = keras_weights_list[0], keras_weights_list[1], keras_weights_list[2]
                    r_layer_scratch.set_weights(W_all_k, U_all_k, b_all_k)
                    print(f"  Successfully loaded LSTM weights. Shapes: W_all={W_all_k.shape}, U_all={U_all_k.shape}, b_all={b_all_k.shape}")
                else:
                    print(f"  ERROR: Expected 3 weight arrays for LSTM, got {len(keras_weights_list)}")
        except Exception as e:
            print(f"  ERROR loading weights for Recurrent layer {keras_rnn_name}: {e}")
        keras_weight_idx += 1

    # 3. Classification Layer (Dense)
    if hasattr(scratch_model, 'classification_layer') and scratch_model.classification_layer is not None:
        if keras_weight_idx >= len(keras_layer_names):
            print(f"Warning: Not enough Keras layer names provided for scratch classification layer.")
        else:
            keras_dense_name = keras_layer_names[keras_weight_idx]
            print(f"Loading weights for Classification Layer from Keras layer: {keras_dense_name}")
            try:
                if keras_dense_name not in h5_file:
                    print(f"  ERROR: Keras layer group '{keras_dense_name}' not found in H5 file.")
                else:
                    keras_weights_list_dense = []
                    if 'weight_names' in h5_file[keras_dense_name].attrs: # Keras 3 style
                        ordered_weight_keys_dense = [name.split('/')[-1] for name in h5_file[keras_dense_name].attrs['weight_names']]
                        for key in ordered_weight_keys_dense:
                             keras_weights_list_dense.append(h5_file[keras_dense_name][key][:])
                    else: # Gaya lama
                        keras_weights_list_dense.append(h5_file[keras_dense_name][f'{keras_dense_name}/kernel:0'][:])
                        keras_weights_list_dense.append(h5_file[keras_dense_name][f'{keras_dense_name}/bias:0'][:])

                    if len(keras_weights_list_dense) == 2:
                        W_k, b_k = keras_weights_list_dense[0], keras_weights_list_dense[1]
                        scratch_model.classification_layer.set_weights(W_k, b_k)
                        print(f"  Successfully loaded Dense weights. Shapes: W={W_k.shape}, b={b_k.shape}")
                    else:
                        print(f"  ERROR: Expected 2 weight arrays for Dense, got {len(keras_weights_list_dense)}")
            except Exception as e:
                print(f"  ERROR loading weights for Dense layer {keras_dense_name}: {e}")
            keras_weight_idx += 1
    
    h5_file.close()
    print("Finished loading weights.")

