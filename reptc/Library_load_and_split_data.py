import numpy as np
import struct
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold
import os
# IDX file reading functions
def read_idx3(filename):
    with open(filename, 'rb') as file:
        magic, num_images, rows, cols = struct.unpack('>IIII', file.read(16))
        if magic != 2051:
            raise ValueError("Invalid IDX3 file format")
        images = np.fromfile(file, dtype=np.uint8).reshape(num_images, rows * cols)  # Flatten images
        return images


def read_idx1(filename):
    with open(filename, 'rb') as file:
        magic, num_items = struct.unpack('>II', file.read(8))
        if magic != 2049:
            raise ValueError("Invalid IDX1 file format")
        labels = np.fromfile(file, dtype=np.uint8)
        return labels




# ---------------------------------------------------------------
# Folder containing the preprocessed strategy files:
#   strategy1.idx3 / strategy1.idx1 ... strategy8.idx3 / strategy8.idx1
# Replace with your own data directory.
# ---------------------------------------------------------------
BASE_DIR = r'/put/your/data/directory/here'

def load_and_preprocess_data(strategy_name, input_shape, num_classes, test_size=0.1, encode_labels=False):
    if not strategy_name:
        raise ValueError("strategy_name must be provided.")

    base_dir = BASE_DIR
    strategy_file_idx3 = os.path.join(base_dir, f"{strategy_name}.idx3")
    strategy_file_idx1 = os.path.join(base_dir, f"{strategy_name}.idx1")

    # Ensure the strategy files exist
    if not os.path.exists(strategy_file_idx3) or not os.path.exists(strategy_file_idx1):
        raise FileNotFoundError(f"Files for strategy {strategy_name} not found in {base_dir}")

    # Load and process data
    X = read_idx3(strategy_file_idx3)
    y = read_idx1(strategy_file_idx1)

    # Restrict input data size based on the first dimension of input_shape
    feature_count = input_shape[0] if isinstance(input_shape, tuple) else input_shape
    X = X[:, :feature_count]  # Use only the number of features specified by input_shape

    # Split data
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    # Preprocess data
    X_train_val_processed, y_train_val_processed = preprocess_data(X_train_val, y_train_val, num_classes, encode_labels)
    X_test_processed, y_test_processed = preprocess_data(X_test, y_test, num_classes, encode_labels)

    return X_train_val_processed, y_train_val_processed, X_test_processed, y_test_processed



def preprocess_data(X, y, num_classes=24, encode_labels=False):
    X = X.reshape((X.shape[0], -1, 1)) / 255.0
    if encode_labels:
        y = to_categorical(y, num_classes)
    return X, y


# Example usage, kept for reference.
# The framework calls these functions directly from B01_NAS.py,
# so nothing is loaded when this module is imported.
#
# X_train_val, y_train_val, X_test, y_test = load_and_preprocess_data(
#     strategy_name="strategy2",
#     input_shape=784,
#     num_classes=11
# )

def fold_index(X, y_scalar, num_splits=4):
    skf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=42)
    return list(skf.split(X, y_scalar))


def get_fold_split(X, Y, folds, i_fold):
    train_ind, val_ind = folds[i_fold]
    X_train = X[train_ind]
    Y_train = Y[train_ind]
    X_val = X[val_ind]
    Y_val = Y[val_ind]
    return X_train, Y_train, X_val, Y_val


# folds = fold_index(X_train_val, y_train_val, num_splits=5)
