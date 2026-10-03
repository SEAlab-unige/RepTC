import json
from tensorflow.keras import layers
import tensorflow as tf


class Block:
    def __init__(self, n_filters=None, kernel_size=None, activation="relu", padding="valid",
                 is_pool=False, pool_size=2, input_size=None, is_dropout=False, dropout_rate=0.5,
                 stride=2, nas_saver_name="NAS_logger", input_shape=None, is_max_pool=False,
                 is_avg_pool=False, block_type="cnn", n_units=None):

        # ------------------------------------------------------------------
        # Common / original attributes
        # ------------------------------------------------------------------
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.padding = padding
        self.is_pool = is_pool
        self.pool_size = pool_size
        self.input_size = input_size
        self.is_dropout = is_dropout
        self.dropout_rate = dropout_rate
        self.stride = stride
        self.nas_saver_name = nas_saver_name
        self.input_shape = input_shape
        self.is_max_pool = is_max_pool
        self.is_avg_pool = is_avg_pool

        # ------------------------------------------------------------------
        # Architecture-family extension
        #
        # IMPORTANT:
        # "cnn" remains the default so that all existing calls to Block(...)
        # continue to behave exactly as before.
        # ------------------------------------------------------------------
        self.block_type = block_type.lower() if isinstance(block_type, str) else "cnn"

        if self.block_type not in ["cnn", "mlp"]:
            raise ValueError(
                f"Unsupported block_type '{self.block_type}'. "
                "Supported types are 'cnn' and 'mlp'."
            )

        # MLP-specific parameter.
        #
        # n_filters is accepted as a fallback for n_units. This is intentional:
        # it helps preserve compatibility with NAS routines that may currently
        # manipulate n_filters before they are made architecture-family aware.
        self.n_units = n_units

        if self.block_type == "mlp" and self.n_units is None:
            self.n_units = self.n_filters

        # ------------------------------------------------------------------
        # CNN-specific consistency
        # ------------------------------------------------------------------
        if self.block_type == "cnn":
            # Ensure that either is_max_pool or is_avg_pool is True
            # when is_pool is True
            if self.is_pool and not (self.is_max_pool or self.is_avg_pool):
                self.is_max_pool = True

        # MLP blocks do not use convolution/pooling operations.
        # We leave the original attributes present because other parts of
        # the framework may access them.
        elif self.block_type == "mlp":
            self.is_pool = False
            self.is_max_pool = False
            self.is_avg_pool = False

        self.output_size = self.calculate_output_size()

    def calculate_output_size(self):
        """
        Calculate the output size of the block.

        CNN:
            Preserves the original convolution + pooling output-size logic.

        MLP:
            The output dimension is simply the number of Dense units.
        """

        # ==============================================================
        # MLP BLOCK
        # ==============================================================
        if self.block_type == "mlp":
            if self.n_units is None:
                raise ValueError(
                    "n_units must be provided for an MLP block."
                )

            if self.n_units <= 0:
                raise ValueError(
                    "n_units must be greater than 0 for an MLP block."
                )

            return int(self.n_units)

        # ==============================================================
        # CNN BLOCK
        # Original behavior kept as closely as possible
        # ==============================================================
        if self.input_size is None or self.kernel_size is None:
            raise ValueError(
                "input_size and kernel_size must be provided "
                "to calculate output size."
            )

        # Strict constraint: Avoid "valid" padding if input size is less than 16
        if self.input_size < 16 and self.padding == "valid":
            self.padding = "same"

        # Adjust kernel size if it exceeds input size
        if self.kernel_size > self.input_size:
            self.kernel_size = self.input_size

        # Enforce valid padding type
        if self.padding not in ["same", "valid"]:
            self.padding = "same"

        # Switch to "same" padding if "valid" is infeasible
        if self.padding == "valid" and self.input_size < self.kernel_size:
            self.padding = "same"

        # Convolution output size calculation
        if self.padding == "same":
            output_size = (
                self.input_size + self.stride - 1
            ) // self.stride

        elif self.padding == "valid":
            output_size = (
                self.input_size - self.kernel_size + self.stride
            ) // self.stride

        # Pooling output size calculation (if applicable)
        if self.is_pool:
            if output_size < self.pool_size:
                # Disable pooling if output_size < pool_size
                self.is_pool = False
                self.is_max_pool = False
                self.is_avg_pool = False

            else:
                output_size = (
                    output_size + self.pool_size - 1
                ) // self.pool_size

        # Ensure output size is valid
        if output_size <= 0:
            # Automatically adjust parameters to avoid invalid configurations
            self.padding = "same"
            self.kernel_size = min(
                self.kernel_size,
                self.input_size
            )
            self.stride = max(1, self.stride)

            output_size = (
                self.input_size + self.stride - 1
            ) // self.stride

        return output_size

    def dump(self):
        """
        Dump block configuration.

        Kept unchanged so logging/saving code that relies on the complete
        Block.__dict__ continues to work for both CNN and MLP blocks.
        """
        config_str = json.dumps(self.__dict__, indent=4)

        with open(f'{self.nas_saver_name}.txt', 'a') as nas_logger:
            nas_logger.write(config_str + "\n")

    def create_layer(self, input_shape=None, is_first_layer=False):
        """
        Create Keras layers corresponding to this block.

        CNN:
            Conv1D -> [BatchNorm] -> Activation ->
            [Pooling] -> [Dropout]

        MLP:
            [Flatten if first block] -> Dense ->
            [BatchNorm] -> Activation -> [Dropout]

        The same function name and interface are retained for compatibility
        with Library_Net and the rest of the NAS framework.
        """

        layers_list = []

        # ==============================================================
        # MLP BLOCK
        # ==============================================================
        if self.block_type == "mlp":

            if self.n_units is None:
                raise ValueError(
                    "n_units must be provided for an MLP block."
                )

            # ----------------------------------------------------------
            # First MLP block:
            #
            # Current traffic data are typically represented as:
            #     (batch, Ls, 1)
            #
            # A normal Dense layer applied directly to this tensor would
            # operate on the last dimension independently.
            #
            # Therefore, the first MLP block flattens the session into:
            #     (batch, Ls)
            #
            # This keeps the same dataset/input-representation pipeline
            # and allows Ls to remain a searched variable.
            # ----------------------------------------------------------
            if is_first_layer:

                flatten_kwargs = {}

                if input_shape is not None:
                    if isinstance(input_shape, int):
                        input_shape = (input_shape, 1)

                    flatten_kwargs["input_shape"] = input_shape

                layers_list.append(
                    tf.keras.layers.Flatten(**flatten_kwargs)
                )

            # Dense transformation
            layers_list.append(
                tf.keras.layers.Dense(
                    units=int(self.n_units),
                    activation=None
                )
            )

            # Keep the same convention currently used by the CNN block:
            # BatchNormalization is not inserted after the first block.
            if not is_first_layer:
                layers_list.append(
                    tf.keras.layers.BatchNormalization()
                )

            # Activation layer
            if self.activation:
                layers_list.append(
                    tf.keras.layers.Activation(self.activation)
                )

            # Dropout
            if self.is_dropout:
                layers_list.append(
                    tf.keras.layers.Dropout(self.dropout_rate)
                )

            return layers_list

        # ==============================================================
        # CNN BLOCK
        # Original logic preserved
        # ==============================================================

        # Ensure input_shape is a tuple
        if input_shape and isinstance(input_shape, int):
            input_shape = (input_shape, 1)

        # Convolutional layer
        conv_kwargs = {
            "filters": self.n_filters,
            "kernel_size": self.kernel_size,
            "activation": None,
            "padding": self.padding,
            "strides": self.stride,
        }

        if is_first_layer:
            conv_kwargs["input_shape"] = input_shape

        layers_list.append(
            tf.keras.layers.Conv1D(**conv_kwargs)
        )

        # Add BatchNormalization only if it's not the first layer
        if not is_first_layer:
            layers_list.append(
                tf.keras.layers.BatchNormalization()
            )

        # Activation layer
        if self.activation:
            layers_list.append(
                tf.keras.layers.Activation(self.activation)
            )

        # Pooling layer (skip pooling for the first block)
        if self.is_pool and not is_first_layer:
            if self.is_max_pool:
                layers_list.append(
                    tf.keras.layers.MaxPooling1D(
                        pool_size=self.pool_size
                    )
                )

            elif self.is_avg_pool:
                layers_list.append(
                    tf.keras.layers.AveragePooling1D(
                        pool_size=self.pool_size
                    )
                )

        # Dropout layer
        if self.is_dropout:
            layers_list.append(
                tf.keras.layers.Dropout(self.dropout_rate)
            )

        return layers_list


