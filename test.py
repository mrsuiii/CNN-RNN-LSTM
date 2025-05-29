import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, LSTM as KerasLSTM, Dense as KerasDense, Input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report
import os
import h5py
import traceback

# Impor kelas-kelas dari scratch yang Anda sediakan
# Pastikan file-file ini berada di direktori yang sama atau PYTHONPATH
from Value import Value
from LSTM import LSTM as ScratchLSTM
from Layer import Layer as ScratchDenseLayer
from activation import softmax # Diasumsikan ada di file activation.py
from SequenceClassifier import StackedSequenceClassifier # Diasumsikan ada di file SequenceClassifier.py

# --- 0. Konfigurasi Dasar ---
VOCAB_SIZE = 10000
EMBEDDING_DIM = 128
MAX_LENGTH = 100
LSTM_UNITS = 64
NUM_RECURRENT_LAYERS = 2  # <--- KONFIGURASI JUMLAH LAYER LSTM DI SINI (misal: 1, 2, atau 3)
NUM_CLASSES = 3  # Akan disesuaikan berdasarkan data
BATCH_SIZE = 32
EPOCHS = 5 # Dibuat singkat untuk debugging, bisa dinaikkan lagi
# Nama file bobot dinamis berdasarkan jumlah layer LSTM
KERAS_MODEL_WEIGHTS_PATH = f'keras_{NUM_RECURRENT_LAYERS}lstm_sentiment_weights.weights.h5'

# Menggunakan nama file yang diunggah pengguna
PATH_TO_TRAIN_CSV = 'train(1).csv'
PATH_TO_TEST_CSV = 'test(1).csv'

# --- 1. Persiapan Data ---
def load_and_preprocess_data(train_path, test_path, vocab_size, max_length):
    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
    except FileNotFoundError:
        print(f"Error: Pastikan file '{train_path}' dan '{test_path}' telah diunggah dan path-nya benar.")
        return None, None, None, None, None, None, None, 0
    except Exception as e:
        print(f"Error saat membaca file CSV: {e}")
        return None, None, None, None, None, None, None, 0

    print("Data Latih Awal (5 baris pertama):")
    print(train_df.head())
    
    required_cols = ['text', 'label']
    if not all(col in train_df.columns for col in required_cols) or \
       not all(col in test_df.columns for col in required_cols):
        print(f"Error: Kolom yang dibutuhkan ({required_cols}) tidak ditemukan di salah satu file CSV.")
        return None, None, None, None, None, None, None, 0

    train_df.dropna(subset=['text', 'label'], inplace=True)
    test_df.dropna(subset=['text', 'label'], inplace=True)
    
    print(f"\nJumlah data latih setelah dropna: {len(train_df)}")
    print(f"Distribusi kelas pada data latih:\n{train_df['label'].value_counts()}")
    print(f"Jumlah data uji setelah dropna: {len(test_df)}")
    print(f"Distribusi kelas pada data uji:\n{test_df['label'].value_counts()}")

    train_texts = train_df['text'].astype(str).values
    test_texts = test_df['text'].astype(str).values

    label_encoder = LabelEncoder()
    train_labels_encoded = label_encoder.fit_transform(train_df['label'])
    test_labels_encoded = label_encoder.transform(test_df['label'])
    
    detected_num_classes = len(label_encoder.classes_)
    print(f"\nLabel unik setelah encoding: {np.unique(train_labels_encoded)}")
    print(f"Mapping Label: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")
    print(f"Jumlah kelas terdeteksi: {detected_num_classes}")

    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<oov>")
    tokenizer.fit_on_texts(train_texts)

    train_sequences = tokenizer.texts_to_sequences(train_texts)
    test_sequences = tokenizer.texts_to_sequences(test_texts)

    train_padded = pad_sequences(train_sequences, maxlen=max_length, padding='post', truncating='post')
    test_padded = pad_sequences(test_sequences, maxlen=max_length, padding='post', truncating='post')
    
    actual_vocab_size = len(tokenizer.word_index) + 1 # +1 untuk padding token jika 0 tidak dihitung oleh num_words
    print(f"Ukuran kosakata aktual (termasuk oov): {actual_vocab_size}")

    return train_padded, train_labels_encoded, test_padded, test_labels_encoded, tokenizer, label_encoder, actual_vocab_size, detected_num_classes

# --- Helper Function for HDF5 Weight Loading ---
def get_keras_weights_container_from_h5_group(h5_layer_group, layer_type_for_debug=""):
    """
    Navigates into the HDF5 group for a Keras layer to find the actual weights container.
    """
    weights_container = None
    group_name_for_debug = h5_layer_group.name
    # print(f"DEBUG: Inspecting HDF5 group '{group_name_for_debug}' for {layer_type_for_debug} layer.")

    if 'cell' in h5_layer_group and isinstance(h5_layer_group['cell'], h5py.Group):
        cell_group = h5_layer_group['cell']
        # print(f"DEBUG:   Found 'cell' subgroup: {cell_group.name}. Keys: {list(cell_group.keys())}")
        if 'vars' in cell_group and isinstance(cell_group['vars'], h5py.Group):
            weights_container = cell_group['vars']
            # print(f"DEBUG:     Using 'vars' subgroup within 'cell': {weights_container.name}")
        else:
            weights_container = cell_group
            # print(f"DEBUG:     WARNING: No 'vars' in 'cell'. Using 'cell' group: {weights_container.name}")
    elif 'vars' in h5_layer_group and isinstance(h5_layer_group['vars'], h5py.Group):
        weights_container = h5_layer_group['vars']
        # print(f"DEBUG:   Found 'vars' subgroup directly: {weights_container.name}. Keys: {list(weights_container.keys())}")
    else:
        weights_container = h5_layer_group
        # print(f"DEBUG:   WARNING: No 'cell/vars' or 'vars' subgroup found directly. Using main group: {weights_container.name}. Keys: {list(weights_container.keys())}")
    
    if weights_container is None:
        raise KeyError(f"Failed to determine weights container in HDF5 group '{group_name_for_debug}'.")
    # print(f"DEBUG:   Final weights container for {layer_type_for_debug}: '{weights_container.name}'. Keys: {list(weights_container.keys())}")
    return weights_container

def main():
    global NUM_CLASSES, EPOCHS # Allow modification of global vars

    print("Memuat dan memproses data...")
    (train_padded, train_labels, test_padded, test_labels,
     tokenizer, label_encoder, actual_vocab_size, detected_num_classes) = load_and_preprocess_data(
        PATH_TO_TRAIN_CSV, PATH_TO_TEST_CSV, VOCAB_SIZE, MAX_LENGTH
    )

    if train_padded is None:
        print("Gagal memuat data. Proses dihentikan.")
        return

    NUM_CLASSES = detected_num_classes 
    print(f"NUM_CLASSES diatur menjadi: {NUM_CLASSES}")
    print(f"NUM_RECURRENT_LAYERS diatur menjadi: {NUM_RECURRENT_LAYERS}")
    print(f"EPOCHS diatur ke: {EPOCHS} untuk testing cepat.")
    print(f"Path bobot Keras akan disimpan di: {KERAS_MODEL_WEIGHTS_PATH}")


    # --- 2. Model Keras ---
    print(f"\n--- Model Keras ({NUM_RECURRENT_LAYERS} LSTM Layers) ---")
    keras_input_layer = Input(shape=(MAX_LENGTH,), name="input_layer")
    keras_embedding_out = Embedding(input_dim=actual_vocab_size, # Gunakan actual_vocab_size
                                      output_dim=EMBEDDING_DIM,
                                      input_length=MAX_LENGTH,
                                      name="embedding_layer")(keras_input_layer)
    
    x = keras_embedding_out
    # Dynamically create LSTM layers
    for i in range(NUM_RECURRENT_LAYERS):
        is_last_lstm = (i == NUM_RECURRENT_LAYERS - 1)
        # Keras layer names: lstm_1_layer, lstm_2_layer, ...
        x = KerasLSTM(LSTM_UNITS, 
                      return_sequences=not is_last_lstm, 
                      name=f"lstm_{i+1}_layer")(x) 
    
    keras_output_layer = KerasDense(NUM_CLASSES, activation='softmax', name="dense_layer")(x)
    keras_model = Model(inputs=keras_input_layer, outputs=keras_output_layer)

    keras_model.compile(optimizer='adam',
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])
    print(keras_model.summary())

    print("\nMelatih model Keras...")
    history = keras_model.fit(train_padded, train_labels,
                              epochs=EPOCHS,
                              batch_size=BATCH_SIZE,
                              validation_split=0.1,
                              verbose=2)

    print("\nMenyimpan bobot model Keras...")
    keras_model.save_weights(KERAS_MODEL_WEIGHTS_PATH)
    print(f"Bobot Keras disimpan di {KERAS_MODEL_WEIGHTS_PATH}")

    loss_keras, acc_keras = keras_model.evaluate(test_padded, test_labels, verbose=0)
    print(f"\nEvaluasi Keras pada data uji: Loss = {loss_keras:.4f}, Accuracy = {acc_keras:.4f}")
    
    keras_predictions_proba = keras_model.predict(test_padded, verbose=0)
    keras_predictions = np.argmax(keras_predictions_proba, axis=1)

    # --- 3. Model From Scratch ---
    print(f"\n--- Model From Scratch ({NUM_RECURRENT_LAYERS} LSTM Layers) ---")
    scratch_classifier = StackedSequenceClassifier(
        input_size=EMBEDDING_DIM,
        recurrent_hidden_size=LSTM_UNITS,
        num_classes=NUM_CLASSES,
        num_recurrent_layers=NUM_RECURRENT_LAYERS, 
        recurrent_type='lstm'
    )
    print(f"Model scratch berhasil diinisiasi dengan {NUM_RECURRENT_LAYERS} layer LSTM.")
    
    # Layer names as defined in Keras model construction
    expected_keras_embedding_layer_name = "embedding_layer"
    expected_keras_lstm_layer_names_user_defined = [f"lstm_{i+1}_layer" for i in range(NUM_RECURRENT_LAYERS)]
    expected_keras_dense_layer_name = "dense_layer"

    # HDF5 group names for Keras LSTM layers (how save_weights usually names them)
    # First LSTM layer group: "lstm"
    # Subsequent LSTM layer groups: "lstm_1", "lstm_2", ...
    h5_group_names_for_keras_lstm_layers = []
    if NUM_RECURRENT_LAYERS > 0:
        h5_group_names_for_keras_lstm_layers.append("lstm") # For the first Keras LSTM
        for i in range(1, NUM_RECURRENT_LAYERS):
            h5_group_names_for_keras_lstm_layers.append(f"lstm_{i}") # For Keras lstm_1, lstm_2, ...

    print(f"\nMencoba memuat bobot dari Keras model ke scratch model...")
    print(f"  Nama layer Embedding Keras: {expected_keras_embedding_layer_name}")
    for i in range(NUM_RECURRENT_LAYERS):
        print(f"  Scratch LSTM Layer ke-{i+1} akan memuat dari HDF5 group Keras: '{h5_group_names_for_keras_lstm_layers[i]}' (terkait Keras layer '{expected_keras_lstm_layer_names_user_defined[i]}')")
    print(f"  Nama layer Dense Keras: {expected_keras_dense_layer_name}")
    try:
        with h5py.File(KERAS_MODEL_WEIGHTS_PATH, 'r') as h5_file:
            print(f"\nInspeksi file H5: {KERAS_MODEL_WEIGHTS_PATH}")
            print(f"Top-level keys/groups di H5 file: {list(h5_file.keys())}")
            
            # Optional: Detailed H5 structure printing (can be verbose)
            # def print_h5_structure(group, prefix=""):
            #     for key in group.keys():
            #         item = group[key]
            #         path = f"{prefix}/{key}"
            #         print(f"   H5 Item: {path} (Type: {type(item)})")
            #         if isinstance(item, h5py.Group):
            #             print(f"     Group Members: {list(item.keys())}")
            #             # print_h5_structure(item, prefix=path) # Recursive call for full depth
            # if "layers" in h5_file: # Common structure
            #      print_h5_structure(h5_file["layers"], prefix="/layers")
            # else:
            #      print_h5_structure(h5_file)


            # Keras `save_weights` typically saves layers under a "layers" group in TF3/Keras3 .weights.h5 format
            # If your model.name is 'model', it might be under 'model/layers'.
            # Let's try to be a bit flexible or assume 'layers' if model.name is not 'model'.
            base_path_prefix = ""
            if "layers" in h5_file: # Standard for functional models
                base_path_prefix = "layers/"
            elif keras_model.name in h5_file and "layers" in h5_file[keras_model.name]: # For models with a name
                 base_path_prefix = f"{keras_model.name}/layers/"


            # Bobot Embedding diambil dari model Keras aktif (tidak dari H5 untuk scratch)
            print(f"\nBobot Embedding akan diambil dari model Keras aktif saat inferensi.")

            # Memuat bobot untuk setiap layer LSTM
            for i, scratch_lstm_layer in enumerate(scratch_classifier.recurrent_layers):
                keras_lstm_h5_group_name = h5_group_names_for_keras_lstm_layers[i]
                print(f"\nMemuat bobot untuk Scratch LSTM Layer ke-{i+1} dari Keras H5 group '{keras_lstm_h5_group_name}'")
                
                # Construct the path to the LSTM layer weights in HDF5
                # It could be directly under model_name or under 'layers' group
                # Common Keras structure is model_name/layers/layer_name or layers/layer_name
                # The h5_group_name ('lstm', 'lstm_1') refers to the Keras layer's group name in H5
                
                current_lstm_h5_path = f"{base_path_prefix}{keras_lstm_h5_group_name}"
                print(f"  Path H5 yang dicoba untuk LSTM ke-{i+1}: {current_lstm_h5_path}")

                if current_lstm_h5_path not in h5_file:
                    # Fallback for older Keras or different saving convention if layers are top-level
                    current_lstm_h5_path_alt = keras_lstm_h5_group_name 
                    if current_lstm_h5_path_alt in h5_file:
                        current_lstm_h5_path = current_lstm_h5_path_alt
                        print(f"  Menggunakan path alternatif: {current_lstm_h5_path}")
                    else:
                         raise KeyError(f"Grup layer LSTM '{current_lstm_h5_path}' (atau '{current_lstm_h5_path_alt}') tidak ditemukan di file H5. Keys tersedia: {list(h5_file.keys())}")

                if not isinstance(h5_file[current_lstm_h5_path], h5py.Group):
                    raise ValueError(f"Path H5 '{current_lstm_h5_path}' bukan grup.")

                keras_lstm_w_group = h5_file[current_lstm_h5_path]
                weights_container = get_keras_weights_container_from_h5_group(keras_lstm_w_group, f"LSTM {i+1}")
                
                actual_keys_in_container = list(weights_container.keys())
                print(f"    Keys dalam container bobot LSTM '{weights_container.name}': {actual_keys_in_container}")
                
                keras_lstm_numerical_keys = ['0', '1', '2'] # Keras LSTM: kernel, recurrent_kernel, bias
                lstm_kernel, lstm_recurrent_kernel, lstm_bias = None, None, None
                loaded_k, loaded_rk, loaded_b = False, False, False

                if keras_lstm_numerical_keys[0] in weights_container:
                    lstm_kernel = weights_container[keras_lstm_numerical_keys[0]][:]
                    loaded_k = True
                if keras_lstm_numerical_keys[1] in weights_container:
                    lstm_recurrent_kernel = weights_container[keras_lstm_numerical_keys[1]][:]
                    loaded_rk = True
                if keras_lstm_numerical_keys[2] in weights_container:
                    lstm_bias = weights_container[keras_lstm_numerical_keys[2]][:]
                    loaded_b = True
                
                if not (loaded_k and loaded_rk and loaded_b): # Fallback to named keys
                    print(f"    Gagal memuat LSTM ke-{i+1} dengan key numerik. Mencoba nama standar...")
                    named_keys_map = {'kernel': ['kernel', 'kernel:0'], 
                                      'recurrent_kernel': ['recurrent_kernel', 'recurrent_kernel:0'], 
                                      'bias': ['bias', 'bias:0']}
                    if not loaded_k:
                        for name in named_keys_map['kernel']:
                            if name in weights_container: lstm_kernel = weights_container[name][:]; loaded_k = True; break
                    if not loaded_rk:
                        for name in named_keys_map['recurrent_kernel']:
                            if name in weights_container: lstm_recurrent_kernel = weights_container[name][:]; loaded_rk = True; break
                    if not loaded_b:
                         for name in named_keys_map['bias']:
                            if name in weights_container: lstm_bias = weights_container[name][:]; loaded_b = True; break
                
                if not (loaded_k and loaded_rk and loaded_b):
                    raise KeyError(f"Bobot LSTM (kernel, recurrent_kernel, atau bias) untuk layer ke-{i+1} ('{keras_lstm_h5_group_name}') tidak ditemukan di '{weights_container.name}'. Keys: {actual_keys_in_container}")

                print(f"    Shape Kernel LSTM Keras: {lstm_kernel.shape}, Recurrent Kernel: {lstm_recurrent_kernel.shape}, Bias: {lstm_bias.shape}")
                scratch_lstm_layer.set_weights(lstm_kernel, lstm_recurrent_kernel, lstm_bias)
                print(f"    Bobot untuk Scratch LSTM Layer ke-{i+1} dari '{weights_container.name}' berhasil dimuat.")

            # Memuat bobot Dense
            keras_dense_h5_group_name = expected_keras_dense_layer_name.replace("_layer", "") # e.g., "dense"
            print(f"\nMemuat bobot untuk Scratch Dense Layer dari Keras H5 group '{keras_dense_h5_group_name}'")
            
            current_dense_h5_path = f"{base_path_prefix}{keras_dense_h5_group_name}"
            print(f"  Path H5 yang dicoba untuk Dense Layer: {current_dense_h5_path}")

            if current_dense_h5_path not in h5_file:
                current_dense_h5_path_alt = keras_dense_h5_group_name
                if current_dense_h5_path_alt in h5_file:
                    current_dense_h5_path = current_dense_h5_path_alt
                    print(f"  Menggunakan path alternatif: {current_dense_h5_path}")
                else:
                    raise KeyError(f"Grup layer Dense '{current_dense_h5_path}' (atau '{current_dense_h5_path_alt}') tidak ditemukan. Keys: {list(h5_file.keys())}")

            if not isinstance(h5_file[current_dense_h5_path], h5py.Group):
                 raise ValueError(f"Path H5 '{current_dense_h5_path}' bukan grup.")

            keras_dense_w_group = h5_file[current_dense_h5_path]
            weights_container_dense = get_keras_weights_container_from_h5_group(keras_dense_w_group, "Dense")
            
            actual_keys_in_dense_container = list(weights_container_dense.keys())
            print(f"    Keys dalam container bobot Dense '{weights_container_dense.name}': {actual_keys_in_dense_container}")

            dense_kernel_keys = ['kernel', 'kernel:0', '0']
            dense_bias_keys = ['bias', 'bias:0', '1']
            dense_kernel, dense_bias = None, None
            loaded_dk, loaded_db = False, False

            for key_name in dense_kernel_keys:
                if key_name in weights_container_dense:
                    dense_kernel = weights_container_dense[key_name][:]
                    loaded_dk = True; break
            for key_name in dense_bias_keys:
                if key_name in weights_container_dense:
                    dense_bias = weights_container_dense[key_name][:]
                    loaded_db = True; break
            
            if not (loaded_dk and loaded_db):
                raise KeyError(f"Bobot Dense (kernel atau bias) tidak ditemukan di '{weights_container_dense.name}'. Keys: {actual_keys_in_dense_container}")

            print(f"    Shape Kernel Dense Keras: {dense_kernel.shape}, Bias: {dense_bias.shape}")
            scratch_classifier.classification_layer.set_weights(dense_kernel, dense_bias)
            print(f"    Bobot Dense dari '{weights_container_dense.name}' berhasil dimuat.")

        print("\nSemua bobot Keras berhasil dimuat ke model scratch.")

    except Exception as e:
        print(f"Error saat memuat bobot Keras ke model scratch: {e}")
        traceback.print_exc()
        print("Pastikan nama layer di Keras (lihat output model.summary()), struktur H5, dan konfigurasi NUM_RECURRENT_LAYERS sudah benar.")
        scratch_classifier = None # Invalidate classifier if loading fails
    # --- 4. Inferensi dan Perbandingan ---
    if scratch_classifier:
        print("\nMelakukan inferensi dengan model scratch (batch processing)...")
        keras_embedding_matrix = keras_model.get_layer(expected_keras_embedding_layer_name).get_weights()[0]
        
        embedded_test_data = keras_embedding_matrix[test_padded]  # Shape: (num_test_samples, MAX_LENGTH, EMBEDDING_DIM)
        
        scratch_predictions_list = []
        num_test_samples = embedded_test_data.shape[0]
        
        print(f"Memproses {num_test_samples} sampel uji dengan model scratch menggunakan BATCH_SIZE={BATCH_SIZE}...")

        for i in range(0, num_test_samples, BATCH_SIZE):
            batch_embedded_data = embedded_test_data[i:i+BATCH_SIZE] # Shape: (current_batch_size, MAX_LENGTH, EMBEDDING_DIM)
            
            # Ensure batch_embedded_data is always 3D, even if it's the last smaller batch
            if batch_embedded_data.ndim == 2: # Should not happen if embedded_test_data is 3D and slicing is correct
                batch_embedded_data = np.expand_dims(batch_embedded_data, axis=0)
            
            scratch_input_batch_sequence = Value(batch_embedded_data) 
            
            # __call__ dari StackedSequenceClassifier sekarang mengharapkan input batch
            # dan mengembalikan output batch
            prediction_batch_value = scratch_classifier(scratch_input_batch_sequence)
            
            # prediction_batch_value.data akan memiliki shape (current_batch_size, NUM_CLASSES)
            scratch_predictions_list.append(prediction_batch_value.data)
            
            if (i // BATCH_SIZE + 1) % 10 == 0 or (i + BATCH_SIZE) >= num_test_samples: # Print progress per 10 batches or at the end
                 print(f"  Batch ke-{i // BATCH_SIZE + 1}/{(num_test_samples + BATCH_SIZE - 1) // BATCH_SIZE} diproses.")

        # Menggabungkan hasil prediksi dari semua batch
        scratch_predictions_proba = np.concatenate(scratch_predictions_list, axis=0)
        scratch_predictions = np.argmax(scratch_predictions_proba, axis=1)

        print("\n--- Hasil Perbandingan ---")
        print("Label Sebenarnya (5 sampel pertama):", test_labels[:5])
        print("Prediksi Keras (5 sampel pertama):", keras_predictions[:5])
        print("Prediksi Scratch (5 sampel pertama):", scratch_predictions[:5])

        f1_keras = f1_score(test_labels, keras_predictions, average='macro', zero_division=0)
        f1_scratch = f1_score(test_labels, scratch_predictions, average='macro', zero_division=0)

        print(f"\nMacro F1-Score Keras: {f1_keras:.4f}")
        print(f"Macro F1-Score Scratch: {f1_scratch:.4f}")

        print("\nLaporan Klasifikasi Keras:")
        print(classification_report(test_labels, keras_predictions, target_names=label_encoder.classes_, zero_division=0))

        print("\nLaporan Klasifikasi Scratch:")
        print(classification_report(test_labels, scratch_predictions, target_names=label_encoder.classes_, zero_division=0))

        # Perbandingan probabilitas
        tolerance = 1e-4 
        if np.allclose(keras_predictions_proba, scratch_predictions_proba, atol=tolerance):
            print(f"\nSUCCESS: Output probabilitas dari Keras dan Scratch sangat mirip (toleransi={tolerance})!")
        else:
            print(f"\nWARNING: Output probabilitas dari Keras dan Scratch berbeda signifikan (toleransi={tolerance}).")
            max_diff = np.max(np.abs(keras_predictions_proba - scratch_predictions_proba))
            avg_diff = np.mean(np.abs(keras_predictions_proba - scratch_predictions_proba))
            print(f"  Perbedaan maksimum absolut: {max_diff:.6e}")
            print(f"  Perbedaan rata-rata absolut: {avg_diff:.6e}")
            print("  Probabilitas (5 sampel pertama) untuk perbandingan:")
            for k_idx in range(min(5, len(keras_predictions_proba))):
                print(f"    Sampel {k_idx}:")
                print(f"      Keras  : {keras_predictions_proba[k_idx]}")
                print(f"      Scratch: {scratch_predictions_proba[k_idx]}")
                print(f"      Diff   : {keras_predictions_proba[k_idx] - scratch_predictions_proba[k_idx]}")
            print("  Jika perbedaan kecil, mungkin karena presisi floating point atau perbedaan implementasi minor.")
            print("  Jika besar, periksa implementasi layer scratch (LSTM, Dense, Softmax), proses pemuatan bobot, dan urutan operasi.")
    else:
        print("\nInferensi dengan model scratch tidak dapat dilakukan karena gagal memuat bobot atau model tidak terinisiasi.")

    # Hapus file bobot sementara jika tidak ingin disimpan (opsional)
    # if os.path.exists(KERAS_MODEL_WEIGHTS_PATH):
    #     os.remove(KERAS_MODEL_WEIGHTS_PATH)
    #     print(f"File bobot sementara {KERAS_MODEL_WEIGHTS_PATH} dihapus.")

if __name__ == '__main__':
    # Pastikan file Value.py, LSTM.py, Layer.py, activation.py, SequenceClassifier.py ada di path
    # Jika belum ada file CSV, Anda bisa buat dummy CSV sederhana untuk tes awal.
    # Contoh dummy train(1).csv:
    # text,label
    # "ini adalah teks positif",positif
    # "saya benci produk ini",negatif
    # "lumayan saja sih",netral
    # "sangat bagus sekali",positif
    
    # Contoh dummy test(1).csv:
    # text,label
    # "tidak terlalu buruk",netral
    # "sangat mengecewakan",negatif
    
    # Buat file dummy jika tidak ada untuk pengujian awal
    dummy_train_file = PATH_TO_TRAIN_CSV
    dummy_test_file = PATH_TO_TEST_CSV

    if not os.path.exists(dummy_train_file):
        print(f"Membuat file dummy {dummy_train_file} untuk pengujian.")
        dummy_train_data = {
            'text': ["ini adalah teks positif yang sangat menyenangkan", 
                     "saya benci produk ini dan tidak akan beli lagi", 
                     "lumayan saja sih tidak ada yang spesial", 
                     "sangat bagus sekali kualitasnya mantap",
                     "pengalaman buruk pelayanan lambat",
                     "netral saja tidak ada komentar lebih",
                     "cinta banget sama barang ini",
                     "kecewa berat dengan hasilnya jelek"],
            'label': ["positif", "negatif", "netral", "positif", "negatif", "netral", "positif", "negatif"]
        }
        pd.DataFrame(dummy_train_data).to_csv(dummy_train_file, index=False)

    if not os.path.exists(dummy_test_file):
        print(f"Membuat file dummy {dummy_test_file} untuk pengujian.")
        dummy_test_data = {
            'text': ["tidak terlalu buruk cukup oke", 
                     "sangat mengecewakan dan merugikan",
                     "biasa saja seperti yang lain",
                     "luar biasa ini yang terbaik"],
            'label': ["netral", "negatif", "netral", "positif"]
        }
        pd.DataFrame(dummy_test_data).to_csv(dummy_test_file, index=False)
        
    main()