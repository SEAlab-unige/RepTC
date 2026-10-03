import tensorflow as tf
#from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tensorflow.keras.utils import to_categorical

# Set up mixed precision
#policy = mixed_precision.Policy('mixed_float16')
#mixed_precision.set_policy(policy)

import Library_load_and_split_data
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from Library_load_and_split_data import get_fold_split
import gc
from sklearn.metrics import accuracy_score
import numpy as np
import Library_compute_stats
from keras_flops import get_flops
from Library_Block import Block
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.utils import Sequence


# ---------------------------------------------------------------
# Number of classes in the dataset.
# Change it here and in B01_NAS.py when switching dataset.
# 11 = ISCX VPN-nonVPN
# ---------------------------------------------------------------
NUM_CLASSES = 11


class Net:
    def __init__(self, block_list, nas_saver_name, preloaded_data=None):
        self.block_list = block_list
        self.nas_saver_name = nas_saver_name
        self.trained_fully = False
        self.last_proxy_epochs_used = 0
        # Directly use the preloaded data if provided
        if preloaded_data:
            self.X_train_val, self.y_train_val, self.X_test, self.y_test = preloaded_data
        else:
            self.X_train_val = self.y_train_val = self.X_test = self.y_test = None

    def get_architecture_family(self):
        """
        Return the architecture family used by this network.

        Backward compatibility:
        Old Block objects that do not contain block_type are treated as CNN.
        """
        if not self.block_list:
            return "cnn"

        return getattr(self.block_list[0], "block_type", "cnn").lower()

    def fetch_data(self, strategy_name, input_shape, num_classes=NUM_CLASSES, test_size=0.1, encode_labels=False):
        if not strategy_name:
            raise ValueError("strategy_name must be provided.")

        print(f"Loading and preprocessing data for strategy {strategy_name} with input shape {input_shape}...")
        X_train_val, y_train_val, X_test, y_test = Library_load_and_split_data.load_and_preprocess_data(
            strategy_name=strategy_name,
            input_shape=input_shape,
            num_classes=num_classes,
            test_size=test_size,
            encode_labels=encode_labels
        )
        self.X_train_val, self.y_train_val = X_train_val, y_train_val
        self.X_test, self.y_test = X_test, y_test
        return self.X_train_val, self.y_train_val, self.X_test, self.y_test

    """"""""""
    def fetch_data(self, num_classes=11, test_size=0.1, encode_labels=False):
        if self.X_train_val is None or self.X_test is None:
            raise ValueError("Data should have been preloaded but was not found.")
        return self.X_train_val, self.y_train_val, self.X_test, self.y_test
    """""""""

    def short_description(self):
        hw_params = self.hw_measures()

        self.log_message(
            f"Net: len = {len(self.block_list)}, "
            f"architecture = {self.get_architecture_family()}, "
            f"n_params = {hw_params[0]}, "
            f"max_tens = {hw_params[1]}, "
            f"flops = {hw_params[2]}, "
            f"flash_size = {hw_params[3]} bytes, "
            f"ram_size = {hw_params[4]} bytes, "
            f"strategy = {getattr(self, 'strategy', 'unknown')}, "
            f"input_shape = {getattr(self, 'input_shape', 'unknown')}"
        )

        return True

    def dump(self):
        self.log_message(' NET: ')
        for i in range(len(self.block_list)):
            self.block_list[i].dump()

    def ins_keras_model(self, input_shape, load_weigths=False):
        model = tf.keras.models.Sequential()

        # Architecture family:
        # - defaults to CNN for backward compatibility
        # - MLP is explicitly selected through Block.block_type
        architecture_family = self.get_architecture_family()

        for i, block in enumerate(self.block_list):
            if i == 0:
                # First block uses the provided input_shape.
                #
                # CNN:
                #   Conv1D receives (Ls, 1)
                #
                # MLP:
                #   Block.create_layer() internally adds Flatten()
                #   before the first Dense layer.
                keras_layers = block.create_layer(
                    input_shape=input_shape,
                    is_first_layer=True
                )
            else:
                keras_layers = block.create_layer(
                    is_first_layer=False
                )

            for layer in keras_layers:
                model.add(layer)

        # ==============================================================
        # Architecture-specific final feature aggregation
        # ==============================================================

        if architecture_family == "cnn":
            # CNN output is still a temporal tensor:
            #     (batch, sequence_length, channels)
            #
            # Preserve the original CNN behavior.
            model.add(tf.keras.layers.GlobalAveragePooling1D())

        elif architecture_family == "mlp":
            # Nothing is required here.
            #
            # The first MLP block already contains Flatten(), so after
            # the final MLP block the tensor has shape:
            #     (batch, n_units)
            #
            # It can therefore feed directly into the classifier.
            pass

        else:
            raise ValueError(
                f"Unsupported architecture family: {architecture_family}"
            )

        # Final classifier remains common to CNN and MLP
        num_classes = NUM_CLASSES
        model.add(
            tf.keras.layers.Dense(
                num_classes,
                activation='softmax'
            )
        )

        if load_weigths:
            for i in range(0, len(model.weights) - 2, 2):
                if self.block_list[int(i / 2)].has_trained_weigths:
                    model.weights[i] = self.block_list[int(i / 2)].trained_weights[0]
                    model.weights[i + 1] = self.block_list[int(i / 2)].trained_weights[1]

            if self.trained_fully is not None:
                model.weights[-2] = self.trained_fully[0]
                model.weights[-1] = self.trained_fully[1]

        from tensorflow.keras.optimizers import Adam

        model.compile(
            optimizer=Adam(learning_rate=0.004),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def train_routine(self, is_train, folds):
        learning_rate_cb = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=5)
        early_stop_cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=12)
        n_epochs = 60
        multistart = 1
        batch_size = 64

        # Ensure data is fetched with the correct input shape
        input_shape = (self.block_list[0].input_size, 1)  # Extract input size from the first block
        self.fetch_data(num_classes=NUM_CLASSES, test_size=0.1, encode_labels=False, input_shape=input_shape)
        y_test_c = self.y_test  # Assuming y_test is already one-hot encoded

        results = []
        for i_fold in range(len(folds)):
            X_train, y_train, X_val, y_val = Library_load_and_split_data.get_fold_split(
                self.X_train_val, self.y_train_val, folds, i_fold)

            all_test_metrics = np.zeros(5)  # Extend to capture the average of accuracy, precision, recall, and F1-score

            for i_mult in range(multistart):
                model = self.ins_keras_model(input_shape=input_shape)
                if is_train:
                    model.fit(X_train, y_train, epochs=n_epochs, batch_size=batch_size, validation_data=(X_val, y_val),
                              callbacks=[learning_rate_cb, early_stop_cb])

                p_val = model.predict(X_val)
                metrics = np.array(Library_compute_stats.compute_descriptors(y_val, p_val))
                val_score = np.mean(metrics[:4])  # Exclude average from validation score

                p_test = model.predict(self.X_test)
                test_metrics = np.array(Library_compute_stats.compute_descriptors(y_test_c, p_test))
                test_metrics = np.append(test_metrics, np.mean(test_metrics[:4]))  # Append the average of first four test metrics  # Append the average of test metrics
                all_test_metrics += test_metrics

                del model
                gc.collect()
                tf.keras.backend.clear_session()

            avg_test_metrics = all_test_metrics / multistart
            results.append((val_score, avg_test_metrics.tolist()))
            print(f"Fold {i_fold + 1} - Validation Metrics: Avg Score={val_score}")
            print(
                f"Fold {i_fold + 1} - Average Test Metrics: Accuracy={avg_test_metrics[0]}, Precision={avg_test_metrics[1]}, Recall={avg_test_metrics[2]}, F1={avg_test_metrics[3]}, Average={avg_test_metrics[4]}")

        return results

    def proxy_train_routine(self, is_train_proxy, selected_fold_index=0, validation_split=0.11, folds=None,
                            strategy_name=None, input_shape=None):
        learning_rate_cb = ReduceLROnPlateau(monitor='val_loss', factor=0.6, patience=14)
        early_stop_cb = EarlyStopping(monitor='val_loss', patience=20)
        n_epochs = 100
        multistart = 1
        batch_size = 2048

        if not strategy_name or input_shape is None:
            raise ValueError("Both strategy_name and input_shape must be provided.")

        print(f"Debug: Using strategy={strategy_name}, input_shape={input_shape}")

        # === Step 1: Dynamically fetch data (retains mutation of preprocessing/input length)
        self.fetch_data(
            num_classes=NUM_CLASSES,
            test_size=0.1,
            encode_labels=False,  # scalar labels only
            strategy_name=strategy_name,
            input_shape=input_shape
        )
        y_test_scalar = self.y_test

        best_val_accs = []
        best_test_accs = []

        # default in case something unexpected happens
        self.last_proxy_epochs_used = 0

        # === Step 2: Get fold split (retain scalar labels)
        print(f"Selected fold index: {selected_fold_index}")
        X_train_fold, y_train_fold_scalar, X_val_fold, y_val_fold_scalar = get_fold_split(
            self.X_train_val, self.y_train_val, folds, selected_fold_index
        )

        # === Step 3: Prepare data indices
        train_idx = np.arange(len(X_train_fold))
        val_idx = np.arange(len(X_val_fold))
        test_idx = np.arange(len(self.X_test))

        for i_mult in range(multistart):
            # === Step 4: Create generators
            train_gen = FlatSessionBatchSequence(
                X_train_fold, y_train_fold_scalar, train_idx,
                batch_size=batch_size, num_classes=NUM_CLASSES, shuffle=True
            )
            val_gen = FlatSessionBatchSequence(
                X_val_fold, y_val_fold_scalar, val_idx,
                batch_size=batch_size, num_classes=NUM_CLASSES, shuffle=False
            )
            test_gen = FlatSessionBatchSequence(
                self.X_test, y_test_scalar, test_idx,
                batch_size=batch_size, num_classes=NUM_CLASSES, shuffle=False
            )

            # === Step 5: Build model
            model = self.ins_keras_model(input_shape=input_shape)

            # === Step 6: Train
            if is_train_proxy:
                history = model.fit(
                    train_gen,
                    validation_data=val_gen,
                    epochs=n_epochs,
                    callbacks=[learning_rate_cb, early_stop_cb],
                    workers=1,
                    use_multiprocessing=False,
                    max_queue_size=16,
                    verbose=1
                )
                epochs_used = len(history.history['loss'])
                del history
            else:
                epochs_used = 0

            self.last_proxy_epochs_used = epochs_used
            print(f"Actual epochs used: {epochs_used}")

            # === Step 7: Evaluate
            val_probs = model.predict(val_gen, verbose=0)
            val_preds = np.argmax(val_probs, axis=1)
            val_true = y_val_fold_scalar
            val_acc = accuracy_score(val_true, val_preds)

            if val_acc > max(best_val_accs, default=0):
                test_probs = model.predict(test_gen, verbose=0)
                test_preds = np.argmax(test_probs, axis=1)
                test_true = y_test_scalar
                test_acc = accuracy_score(test_true, test_preds)
                best_test_acc = test_acc

            best_val_accs.append(val_acc)
            if 'best_test_acc' in locals():
                best_test_accs.append(best_test_acc)

            del model
            gc.collect()
            tf.keras.backend.clear_session()

        avg_best_val_acc = sum(best_val_accs) / len(best_val_accs)
        avg_best_test_acc = sum(best_test_accs) / len(best_test_accs) if best_test_accs else 0

        print(f"Average Best validation accuracy: {avg_best_val_acc:.4f}")
        print(f"Average Best test accuracy: {avg_best_test_acc:.4f}")
        return [(avg_best_val_acc, avg_best_test_acc)]

    def hw_measures(self):
        # Ensure block_list is valid and input_size is accessible
        if not self.block_list or not hasattr(self.block_list[0], 'input_size'):
            raise ValueError(
                "Error: block_list is empty or input_size is not defined."
            )

        # Session representation stays identical for CNN and MLP:
        #     (Ls, 1)
        input_size = self.block_list[0].input_size
        input_shape = (input_size, 1)

        # Validate consistency if Net.input_shape exists
        if hasattr(self, "input_shape") and self.input_shape is not None:
            if input_size != self.input_shape[0]:
                raise ValueError(
                    f"Inconsistent input size: "
                    f"block[0].input_size={input_size} does not match "
                    f"input_shape[0]={self.input_shape[0]}"
                )

        architecture_family = self.get_architecture_family()

        print(
            f"Debug: Using architecture={architecture_family}, "
            f"input_shape={input_shape} for Keras model"
        )

        # Build either CNN or MLP according to the blocks
        model = self.ins_keras_model(
            input_shape=input_shape
        )

        # Calculate parameters
        n_params = model.count_params()

        # Maximum intermediate tensor size
        tensor_sizes = []

        for layer in model.layers:
            try:
                layer_output_shape = layer.output_shape

                if (
                        layer_output_shape is not None
                        and None not in layer_output_shape[1:]
                ):
                    tensor_sizes.append(
                        np.prod(layer_output_shape[1:])
                    )

            except (AttributeError, TypeError):
                # Some Keras layers/versions may not expose output_shape
                # identically. Ignore such layers rather than breaking NAS.
                continue

        max_tens = max(
            tensor_sizes,
            default=0
        )

        # Calculate FLOPs using keras_flops
        try:
            flops = get_flops(
                model,
                batch_size=1
            )

        except Exception as e:
            print(f"Error calculating FLOPs: {e}")
            flops = 0

        # Hardware-memory proxies remain unchanged
        flash_size = 4 * n_params
        ram_size = 4 * max_tens

        print(
            f"Debug: Architecture={architecture_family}, "
            f"FLOPs={flops}, "
            f"Params={n_params}, "
            f"Max Tensor={max_tens}, "
            f"Flash Size={flash_size}, "
            f"RAM Size={ram_size}"
        )

        return [
            n_params,
            max_tens,
            flops,
            flash_size,
            ram_size
        ]

    def log_message(self, message):
        """Logs a message to the specified NAS log file."""
        with open(self.nas_saver_name + '.txt', 'a') as log_file:
            log_file.write(message + '\n')

class FlatSessionBatchSequence(Sequence):
    def __init__(self, X, y, idxs, batch_size, num_classes, shuffle=True):
        self.X = X
        self.y = y
        self.idxs = np.array(idxs)
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.idxs) / self.batch_size))

    def __getitem__(self, index):
        batch_ids = self.idxs[index * self.batch_size:(index + 1) * self.batch_size]
        X_batch = self.X[batch_ids]
        y_batch = self.y[batch_ids]
        y_cat = tf.keras.utils.to_categorical(y_batch, num_classes=self.num_classes)
        return X_batch, y_cat

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.idxs)