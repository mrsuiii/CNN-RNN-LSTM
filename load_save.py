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