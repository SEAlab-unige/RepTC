import Library_NAS
import Library_load_and_split_data
import numpy as np
from tensorflow.keras.utils import to_categorical
from datetime import datetime
import Library_Net
import time


# ==============================================================
# Dataset
#
# num_classes must match your dataset.
# 11 = ISCX VPN-nonVPN.
# When you change it, change NUM_CLASSES in Library_Net.py too.
#
# The data directory is set in Library_load_and_split_data.py.
# ==============================================================
num_classes = 11
test_size = 0.1


# ==============================================================
# Training mode
#
# use_full_training (set in the NAS call below) chooses the routine
# used to evaluate candidates:
#   False -> proxy routine, one fold, much faster, used during search
#   True  -> full routine, all folds
#
# The two flags below decide whether that routine actually calls fit().
# Setting both to False runs the search without training, which is
# useful to check the hardware filtering on a new dataset.
# ==============================================================
is_train_proxy = True   # fit() in the proxy routine
is_train = False        # fit() in the full routine


# ==============================================================
# Architecture family
#
# "cnn" -> 1-D convolutional blocks (default)
# "mlp" -> dense blocks
# ==============================================================
architecture_family = "cnn"
#architecture_family = "mlp"


# ==============================================================
# MLP search space
#
# Used only when architecture_family == "mlp", ignored for CNN.
# Depth is not set here: blocks are added and removed during the
# search, bounded by max_depth.
# ==============================================================
mlp_units_range = (16, 128)


# ==============================================================
# Starting point of the search
#
# These are only the initial values. When search_Sh and search_Ls
# are True, both evolve during the search.
# ==============================================================
parent_strategy = "strategy2"
parent_input_shape = 784


# Load the initial data
X_train_val, y_train_val, X_test, y_test = (
    Library_load_and_split_data.load_and_preprocess_data(
        strategy_name=parent_strategy,
        input_shape=parent_input_shape,
        num_classes=num_classes,
        test_size=test_size,
        encode_labels=False
    )
)


# Generate stratified K-fold indices
folds = Library_load_and_split_data.fold_index(
    X_train_val,
    y_train_val
)


# Pack the preloaded data for the search
preloaded_data = (
    X_train_val,
    y_train_val,
    X_test,
    y_test
)


# Create a unique timestamp for this run
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

nas_saver_name = f"NAS_logger_{timestamp}"
partial_saver_name = f"Partial_saver_logger_{timestamp}"


# Initialize the RepTC joint search
NAS = Library_NAS.NAS(

    architecture_family=architecture_family,

    # Used only for the MLP family
    mlp_units_range=mlp_units_range,


    # ==========================================================
    # What the search is free to change
    #
    # Set one to False to hold that variable at its fixed value
    # below, which reduces the search to the remaining variables.
    # ==========================================================
    search_A=True,    # architecture
    search_Sh=True,   # header preprocessing
    search_Ls=True,   # input length

    preserve_architecture=False,

    fixed_Sh="strategy2",   # value used when search_Sh=False
    fixed_Ls=784,           # value used when search_Ls=False


    is_train_proxy=is_train_proxy,
    is_train=is_train,


    # ==========================================================
    # Architecture depth, for both CNN and MLP
    # ==========================================================
    max_depth_father=5,
    max_depth=4,


    # ==========================================================
    # Hardware limits
    #
    # Checked before a candidate is trained. Candidates that
    # exceed any limit are discarded. Adjust to your target
    # device, or set a limit to np.inf to leave it unconstrained.
    # ==========================================================
    check_hw=True,

    params_thr=60000,      # parameters, maps to Flash
    flops_thr=500000,      # compute per inference
    flash_thr=250000,      # bytes
    ram_thr=60000,         # bytes
    max_tens_thr=4000,     # peak intermediate tensor, maps to RAM


    # ==========================================================
    # Search effort
    # ==========================================================
    n_generations=50,
    n_child=20,
    n_mutations=2,


    partial_save_steps=5,
    smart_start=True,
    is_random_walk=True,


    nas_saver_name=nas_saver_name,
    partial_saver_name=partial_saver_name,


    preloaded_data=preloaded_data,
    use_full_training=False
)


# Run the search
start_time = time.perf_counter()

generated_net, best_absolute = NAS.run_NAS(folds)

end_time = time.perf_counter()
total_nas_time_sec = end_time - start_time


print(f"Total search wall-clock time: {total_nas_time_sec:.2f} seconds")
print(f"Total search wall-clock time: {total_nas_time_sec / 60:.2f} minutes")
print(f"Total search wall-clock time: {total_nas_time_sec / 3600:.2f} hours")
