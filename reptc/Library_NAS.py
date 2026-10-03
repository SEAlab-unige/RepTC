import random
import copy
import numpy as np
import Library_Net
import Library_Block
import Library_load_and_split_data
import os
from datetime import datetime
import time
import gc
import tensorflow as tf
import time
#import Library_Dist_Net
class NAS:
    def __init__(self, is_train_proxy, is_train, max_depth_father=3, max_depth=10, check_hw=True, params_thr=np.inf, flops_thr=np.inf, max_tens_thr=np.inf, flash_thr=np.inf, ram_thr=np.inf,
                 n_generations=50, n_child=5, n_mutations=1, partial_save_steps=5, smart_start=True, is_random_walk=True,
                 nas_saver_name="NAS_logger", partial_saver_name="Partial_saver_logger", preloaded_data=None, use_full_training=False,
                 search_Sh=True, search_Ls=True, search_A=True, preserve_architecture=False, fixed_Sh=None,fixed_Ls=None, architecture_family="cnn",mlp_units_range=(16, 128)):

        self.search_Sh = search_Sh
        self.search_Ls = search_Ls
        self.search_A = search_A
        self.preserve_architecture = preserve_architecture

        self.fixed_Sh = fixed_Sh
        self.fixed_Ls = fixed_Ls

        if self.preserve_architecture and self.search_A:
            raise ValueError(
                "Invalid NAS configuration: "
                "preserve_architecture=True requires search_A=False."
            )

        # =========================================================
        # Architecture-family configuration
        #
        # Backward compatibility:
        # CNN remains the default, so existing NAS calls continue
        # to behave exactly as before.
        # =========================================================
        self.architecture_family = str(architecture_family).lower()

        if self.architecture_family not in ["cnn", "mlp"]:
            raise ValueError(
                f"Unsupported architecture_family "
                f"'{self.architecture_family}'. "
                f"Supported values are 'cnn' and 'mlp'."
            )

        # Search range used only by the MLP architecture space.
        # Keeping it here, rather than inside Library_Block,
        # keeps the search-space definition inside Library_NAS.
        self.mlp_units_range = tuple(mlp_units_range)

        if (
                len(self.mlp_units_range) != 2
                or self.mlp_units_range[0] < 1
                or self.mlp_units_range[1] < self.mlp_units_range[0]
        ):
            raise ValueError(
                "mlp_units_range must be a tuple "
                "(min_units, max_units) with positive values."
            )

        self.use_full_training = use_full_training
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.nas_saver_name = f"{nas_saver_name}_{timestamp}"
        self.partial_saver_name = f"{partial_saver_name}_{timestamp}"
        self.max_depth_father = max_depth_father
        self.max_depth = max_depth
        self.is_train_proxy = is_train_proxy
        self.is_train = is_train
        self.preloaded_data = preloaded_data
        self.loaded_datasets = {}
       # self.input_size = input_size

        self.n_generations = n_generations
        self.n_child = n_child
        self.n_mutations = n_mutations
        #self.nas_saver_name = nas_saver_name
        self.log_message('-------------------------NEW NAS--------------------', mode='w')
        self.log_message(
            f"Architecture family: {self.architecture_family}"
        )

        self.check_hw = check_hw
        self.flops_thr = flops_thr
        self.flash_thr = flash_thr
        self.ram_thr = ram_thr
        self.params_thr = params_thr
        self.max_tens_thr = max_tens_thr

        self.partial_save_steps = partial_save_steps
        self.smart_start = smart_start
        self.is_random_walk = is_random_walk


        #self.partial_saver_name = partial_saver_name
        self.base_dropout_rate = 0.1
        self.max_dropout_rate = 0.5
        # Add preprocessing-related attributes
        self.strategy_pool = [f'strategy{i}' for i in range(1, 9)]  # Strategy names: strategy1 to strategy8
        #self.parent_strategy = random.choice(self.strategy_pool)  # Initialize with a random strategy
        self.parent_strategy = fixed_Sh if fixed_Sh is not None else "strategy2"
        #self.parent_input_shape = random.randint(196, 784)
        self.parent_input_shape = fixed_Ls if fixed_Ls is not None else 784# Set a fixed input size
        # Initialize with a random input size

        #self.is_dist = is_dist
        self.total_candidates_evaluated = 0
        self.total_hw_rejections = 0


    def run_NAS(self, folds):
        # Ensure data is loaded from outside this function
        if self.preloaded_data is None:
            print("Error: Data must be preloaded and provided to NAS.")
            return

        absolute_best_score = 0
        # ------------------------------------------------------
        # Reference architecture a0 from architecture-only NAS
        # ------------------------------------------------------

        # ------------------------------------------------------
        # Initial architecture a0
        #
        # CNN:
        #   Preserve the existing reference architecture exactly.
        #
        # MLP:
        #   Build a lightweight initial MLP using the same Block
        #   abstraction. The evolutionary NAS will subsequently
        #   add/remove/change these blocks as usual.
        # ------------------------------------------------------

        if self.architecture_family == "cnn":

            # ==================================================
            # ORIGINAL CNN INITIAL ARCHITECTURE
            # ==================================================

            b1 = Library_Block.Block(
                n_filters=27,
                kernel_size=7,
                activation="relu",
                padding="same",
                is_pool=False,
                pool_size=None,
                input_size=self.parent_input_shape,
                is_dropout=True,
                dropout_rate=0.10,
                stride=7,
                nas_saver_name=self.nas_saver_name,
                is_max_pool=False,
                is_avg_pool=False,
                block_type="cnn"
            )

            b1_output_size = b1.calculate_output_size()

            b2 = Library_Block.Block(
                n_filters=82,
                kernel_size=5,
                activation="relu",
                padding="valid",
                is_pool=False,
                pool_size=None,
                input_size=b1_output_size,
                is_dropout=True,
                dropout_rate=0.23,
                stride=7,
                nas_saver_name=self.nas_saver_name,
                is_max_pool=False,
                is_avg_pool=False,
                block_type="cnn"
            )

            b2_output_size = b2.calculate_output_size()

            b3 = Library_Block.Block(
                n_filters=38,
                kernel_size=1,
                activation="relu",
                padding="valid",
                is_pool=False,
                pool_size=3,
                input_size=b2_output_size,
                is_dropout=True,
                dropout_rate=0.37,
                stride=2,
                nas_saver_name=self.nas_saver_name,
                is_max_pool=False,
                is_avg_pool=False,
                block_type="cnn"
            )

            b3_output_size = b3.calculate_output_size()

            b4 = Library_Block.Block(
                n_filters=63,
                kernel_size=4,
                activation="relu",
                padding="same",
                is_pool=False,
                pool_size=2,
                input_size=b3_output_size,
                is_dropout=True,
                dropout_rate=0.50,
                stride=6,
                nas_saver_name=self.nas_saver_name,
                is_max_pool=False,
                is_avg_pool=False,
                block_type="cnn"
            )

            initial_blocks = [b1, b2, b3, b4]

        elif self.architecture_family == "mlp":

            # ==================================================
            # INITIAL MLP ARCHITECTURE
            # ==================================================
            #
            # Start from a lightweight feasible architecture.
            # Use the minimum allowed number of units so the
            # starting parent is unlikely to violate the HW budget.
            #
            # This initial point does NOT restrict the search:
            # mutations can subsequently resample units and depth.
            # ==================================================

            initial_blocks = []

            initial_depth = max(
                1,
                min(self.max_depth_father, self.max_depth)
            )

            current_input_size = self.parent_input_shape

            for i_block in range(initial_depth):

                if initial_depth > 1:
                    dropout_rate = (
                            self.base_dropout_rate
                            +
                            (
                                    self.max_dropout_rate
                                    - self.base_dropout_rate
                            )
                            * i_block
                            / (initial_depth - 1)
                    )
                else:
                    dropout_rate = self.max_dropout_rate

                dropout_rate = round(dropout_rate, 2)

                n_units = self.mlp_units_range[0]

                block = Library_Block.Block(
                    # n_filters is intentionally mirrored for
                    # compatibility with legacy code/logging that
                    # may still inspect this attribute.
                    n_filters=n_units,

                    # CNN-only fields remain present but are unused
                    # by an MLP Block.
                    kernel_size=None,
                    activation="relu",
                    padding="valid",
                    is_pool=False,
                    pool_size=None,

                    input_size=current_input_size,

                    is_dropout=True,
                    dropout_rate=dropout_rate,

                    stride=1,
                    nas_saver_name=self.nas_saver_name,

                    is_max_pool=False,
                    is_avg_pool=False,

                    block_type="mlp",
                    n_units=n_units
                )

                initial_blocks.append(block)

                # For MLP:
                # output_size == number of Dense units
                current_input_size = block.output_size

        parent = Library_Net.Net(
            block_list=initial_blocks,
            nas_saver_name=self.nas_saver_name,
            preloaded_data=self.preloaded_data
        )

        parent.strategy = self.parent_strategy
        parent.input_shape = (self.parent_input_shape, 1)

        # Load data for the parent network using the initial strategy and input size
        print(f"Debug: Using parent_strategy={self.parent_strategy}, input_shape={self.parent_input_shape}")
        parent.fetch_data(strategy_name=parent.strategy, input_shape=parent.input_shape)

        parent.short_description()

        #else:
            #parent = self.generate_random_network_and_control()

        for i_gen in range(self.n_generations):
            gen_start = time.perf_counter()
            self.log_message('GENERATION ' + str(i_gen) + '\n')

            # Perform one NAS step
            parent, gen_best_score, test_metrics = self.one_nas_step(parent, folds)
            gen_end = time.perf_counter()
            gen_time = gen_end - gen_start
            self.log_message(f"Generation {i_gen} wall-clock time: {gen_time:.2f} seconds")

            # Delay for cooling after each generation
            print(f"Delaying 15 seconds for cooling down after generation {i_gen}")
            time.sleep(15)  # Delay for cooling

            # Check if a new absolute best score is found
            if gen_best_score > absolute_best_score:
                absolute_best_score = gen_best_score
                best_network = copy.deepcopy(parent)

                self.log_message('-----------------------------------\n')
                self.log_message(f'NEW ABSOLUTE BEST FOUND AT GENERATION {i_gen}\n')

                # Pass the `input_shape` when calling `save_partial`
                self.save_partial(parent, best_network, i_gen, gen_best_score, absolute_best_score)

                # Additional logging and network description
                best_network.dump()
                best_network.short_description()

        self.log_message(f"Total evaluated candidates: {self.total_candidates_evaluated}")
        self.log_message(f"Total hardware rejections: {self.total_hw_rejections}")

        return parent, best_network

    def train_generation(self, child_set, folds):
        gen_per = []
        average_test_metrics_per_child = []  # To store average test metrics for each child

        for net in child_set:
            # Debug to verify strategy and input shape
            print(
                f"Debug: Training child with strategy={getattr(net, 'strategy', None)}, input_shape={getattr(net, 'input_shape', None)}")

            if not getattr(net, 'strategy', None) or not getattr(net, 'input_shape', None):
                raise ValueError("Child network is missing 'strategy' or 'input_shape'.")
            if self.use_full_training:
                # Use the full training routine, which returns a list of (average validation score, test_metrics)
                train_results = net.train_routine(self.is_train, folds)
                # Collect the average validation scores from the returned results
                avg_val_score = sum(result[0] for result in train_results) / len(train_results)
                # Calculate average test metrics across all folds
                total_test_metrics = np.zeros(5)  # Assuming test_metrics are in the form [acc, prec, rec, f1]
                for result in train_results:
                    total_test_metrics += np.array(result[1])  # Assuming result[1] is a list/tuple of metrics
                avg_test_metrics = total_test_metrics / len(train_results)

                gen_per.append(avg_val_score)
                average_test_metrics_per_child.append(avg_test_metrics.tolist())  # Convert to list for consistency
            else:
                # Since proxy_train_routine now returns only a single tuple
                proxy_result = net.proxy_train_routine(
                    is_train_proxy=True,
                    selected_fold_index=0,
                    validation_split=0.11,
                    folds=folds,
                    strategy_name=getattr(net, 'strategy', None),  # Dynamically get strategy
                    input_shape=getattr(net, 'input_shape', None)  # Dynamically get input_shape
                )[0]
                avg_val_score, avg_test_acc = proxy_result

                epochs_used = getattr(net, 'last_proxy_epochs_used', None)
                if epochs_used is not None:
                    self.log_message(
                        f"Child proxy training metadata: strategy={getattr(net, 'strategy', None)}, "
                        f"input_shape={getattr(net, 'input_shape', None)}, "
                        f"epochs_used={epochs_used}, val_acc={avg_val_score}, test_acc={avg_test_acc}"
                    )

                gen_per.append(avg_val_score)
                average_test_metrics_per_child.append([avg_test_acc])  # Store it as a list of one element

            # Delay after training each child network
            print(f"Delaying 5 seconds for cooling down after training child network")
            time.sleep(5)  # Delay for 60 seconds
        return gen_per, average_test_metrics_per_child

    def best_child_selection(self, child_set, gen_per, test_metrics_list):
        self.log_message('Selection Starts: \n')

        best = -1  # Initialize with -1, assuming accuracy ranges [0, 1]
        best_child = None
        best_test_metrics = None  # To store the test metrics of the best child

        for i_c, score in enumerate(gen_per):
            child_strategy = getattr(child_set[i_c], 'strategy', 'unknown')
            child_input_shape = getattr(child_set[i_c], 'input_shape', 'unknown')
            self.log_message(
                f"Child {i_c} Val Acc: {score}, strategy = {child_strategy}, input_shape = {child_input_shape}\n"
            )
            if score > best:  # Looking for higher validation accuracy
                best = score
                best_child = copy.deepcopy(child_set[i_c])
                best_test_metrics = test_metrics_list[i_c]  # Retrieve corresponding test metrics for the best child

        self.log_message(f"Selection ended. Best Val Acc: {best}\n")
        if len(best_test_metrics) == 5:
            self.log_message(
                f"Best Child Test Metrics: Accuracy={best_test_metrics[0]}, Precision={best_test_metrics[1]}, "
                f"Recall={best_test_metrics[2]}, F1={best_test_metrics[3]}, Average={best_test_metrics[4]}\n"
            )
        else:
            self.log_message(f"Best Child Test Metrics: Test Accuracy={best_test_metrics[0]}\n")

        if best_child is not None:
            self.log_message(
                f"Best child selected with strategy = {getattr(best_child, 'strategy', 'unknown')}, "
                f"input_shape = {getattr(best_child, 'input_shape', 'unknown')}"
            )
            best_child.dump()
            best_child.short_description()
        return best_child, best

    def one_nas_step(self, parent, folds):
        """
        Mutates, trains, logs, and evaluates each child sequentially.

        The original NAS behavior is preserved when search_A=True.

        Ablation flags:
            search_A=False  -> architecture is not mutated
            search_Sh=False -> Sh is fixed
            search_Ls=False -> Ls is fixed
        """

        best_score = -1
        best_child = None
        best_test_metrics = None

        children_logs = []

        self.log_message(
            f"Starting NAS Step for Parent: "
            f"Architecture={self.architecture_family}, "
            f"Strategy={parent.strategy}, "
            f"Input Shape={parent.input_shape}, "
            f"search_A={self.search_A}, "
            f"search_Sh={self.search_Sh}, "
            f"search_Ls={self.search_Ls}"
        )

        for i_child in range(self.n_child):

            child_start = time.perf_counter()

            self.log_message(
                f"MUTATING CHILD {i_child}"
            )

            # =====================================================
            # Sh behavior
            # Original behavior preserved when search_Sh=True.
            # =====================================================
            if self.search_Sh:
                child_strategy = self.mutate_preprocessing_strategy(
                    parent.strategy,
                    self.strategy_pool
                )
            else:
                child_strategy = (
                    self.fixed_Sh
                    if self.fixed_Sh is not None
                    else parent.strategy
                )

            # =====================================================
            # Ls behavior
            # Original behavior preserved when search_Ls=True.
            # =====================================================
            if self.search_Ls:
                child_input_shape = self.mutate_input_shape(
                    parent.input_shape[0]
                )
            else:
                child_input_shape = (
                    self.fixed_Ls
                    if self.fixed_Ls is not None
                    else parent.input_shape[0]
                )

            self.log_message(
                f"Child {i_child} proposed representation: "
                f"strategy={child_strategy}, "
                f"Ls={child_input_shape}"
            )

            # =====================================================
            # First mutation/candidate generation
            # =====================================================
            child = self.mutate_network_and_control(
                parent,
                self.preloaded_data,
                child_input_shape,
                child_strategy
            )

            # =====================================================
            # Additional mutations
            #
            # OLD NAS behavior is preserved for search_A=True:
            # use the originally proposed Sh and Ls again.
            #
            # When search_A=False there is no reason to perform
            # repeated architecture mutations.
            # =====================================================
            if self.search_A and self.n_mutations > 1:

                for i_mut in range(self.n_mutations - 1):
                    child = self.mutate_network_and_control(
                        child,
                        self.preloaded_data,
                        child_input_shape,
                        child_strategy
                    )

            # =====================================================
            # Read actual accepted values
            # Important because HW fallback may reduce Ls.
            # =====================================================
            actual_strategy = child.strategy
            actual_input_shape = child.input_shape[0]

            self.log_message(
                f"Child {i_child} accepted representation: "
                f"strategy={actual_strategy}, "
                f"Ls={actual_input_shape}"
            )

            # Log architecture
            self.log_message(
                f"\nChild {i_child} Architecture:"
            )

            child.dump()
            child.short_description()

            # =====================================================
            # Train immediately
            # =====================================================
            self.log_message(
                f"Training Child {i_child}"
            )

            gen_per_child, test_metrics_child = (
                self.train_generation(
                    [child],
                    folds
                )
            )

            avg_val_score = gen_per_child[0]
            avg_test_metrics = test_metrics_child[0]

            # =====================================================
            # Store actual accepted representation
            # =====================================================
            children_logs.append(
                (
                    i_child,
                    actual_strategy,
                    actual_input_shape,
                    avg_val_score,
                    avg_test_metrics
                )
            )

            self.total_candidates_evaluated += 1

            # =====================================================
            # Best child of this generation
            # =====================================================
            if avg_val_score > best_score:

                if best_child is not None:
                    del best_child
                    gc.collect()
                    tf.keras.backend.clear_session()

                best_score = avg_val_score
                best_child = copy.deepcopy(child)
                best_test_metrics = avg_test_metrics

            child_end = time.perf_counter()
            child_time = child_end - child_start

            self.log_message(
                f"Child {i_child} total wall-clock time: "
                f"{child_time:.2f} seconds"
            )

            # Cleanup
            del child
            gc.collect()
            tf.keras.backend.clear_session()

            self.log_message("")

        # =========================================================
        # Generation summary
        # =========================================================
        self.log_message(
            "Selection Starts:"
        )

        for child_log in children_logs:
            (
                i_child,
                strategy,
                input_shape,
                val_acc,
                test_metrics
            ) = child_log

            self.log_message(
                f"Child {i_child} "
                f"Val Acc={val_acc}, "
                f"strategy={strategy}, "
                f"input_shape={input_shape}"
            )

        self.log_message(
            f"Selection ended. Best Val Acc: {best_score}"
        )

        if best_test_metrics is not None:

            if len(best_test_metrics) == 5:

                self.log_message(
                    f"Best Child Test Metrics: "
                    f"Accuracy={best_test_metrics[0]}, "
                    f"Precision={best_test_metrics[1]}, "
                    f"Recall={best_test_metrics[2]}, "
                    f"F1={best_test_metrics[3]}, "
                    f"Average={best_test_metrics[4]}"
                )

            else:

                self.log_message(
                    f"Best Child Test Metrics: "
                    f"Test Accuracy={best_test_metrics[0]}"
                )

        if best_child is not None:
            self.log_message(
                f"Best child selected with "
                f"strategy={best_child.strategy}, "
                f"input_shape={best_child.input_shape}"
            )

            best_child.dump()
            best_child.short_description()

        return best_child, best_score, best_test_metrics

    def mutate_preprocessing_strategy(self, parent_strategy, strategy_pool):
        if random.random() < 0.5:  # 50% probability
            # Retain the parent's preprocessing strategy
            return parent_strategy
        else:  # 50% probability
            # Select a different strategy from the pool, excluding the parent strategy
            remaining_strategies = [s for s in strategy_pool if s != parent_strategy]
            return random.choice(remaining_strategies)

    def reflective_clipping(self, size, lower=196, upper=784):
        """
        Cyclic reflective clipping.
        If the size exceeds the boundaries, the remaining steps are reflected to the opposite side of the range.
        """
        range_size = upper - lower  # The total range size
        while size < lower or size > upper:
            if size < lower:
                # Reflect back to the upper boundary
                size = upper - (lower - size)
            elif size > upper:
                # Reflect back to the lower boundary
                size = lower + (size - upper)

        return size

    def mutate_input_shape(self, parent_size, delta_max=128):
        """
        Mutate input shape with cyclic reflective clipping.
        """
        lower = 196  # Define lower bound
        upper = 784  # Define upper bound

        if random.random() < 0.5:  # 50% chance to retain the parent's input shape
            return parent_size
        else:
            if random.random() < 0.8:  # 80% chance for small delta mutation
                # Generate a random even delta within [-delta_max, +delta_max]
                delta = random.randint(-delta_max // 2, delta_max // 2) * 2
                mutated_size = parent_size + delta
            else:  # 20% chance for forced exploration
                # Randomly select a size within the full range
                mutated_size = random.randint(lower // 2, 784 // 2) * 2

            # Apply cyclic reflective clipping
            mutated_size = self.reflective_clipping(mutated_size, lower=lower, upper=upper)

            return mutated_size

    def mutate_network_and_control(
            self,
            parent,
            preloaded_data,
            child_input_shape,
            child_strategy
    ):
        """
        Generate a candidate under hardware constraints.

        IMPORTANT:
        For search_A=True, this preserves the original NAS retry logic:
            - try architecture mutations;
            - after 20 HW failures:
                * if Ls is searchable, reduce Ls by 32;
                * if Ls is fixed, keep retrying architecture mutations.

        Additional ablation behavior:
            search_A=False:
                architecture is copied instead of mutated.

            preserve_architecture=True:
                correct_blocklist() uses its strict fixed-architecture branch.
        """

        if self.check_hw:

            check_pass = False
            attempts = 0

            # ORIGINAL value
            max_attempts = 20

            while not check_pass:

                # =================================================
                # ORIGINAL HW FALLBACK BEHAVIOR
                # =================================================
                if attempts >= max_attempts:

                    # ---------------------------------------------
                    # Ls is searchable:
                    # ORIGINAL behavior = reduce Ls by 32.
                    # ---------------------------------------------
                    if self.search_Ls:

                        original_input = child_input_shape

                        child_input_shape = max(
                            196,
                            child_input_shape - 32
                        )

                        self.log_message(
                            f"Too many failed attempts ({attempts}). "
                            f"Reducing input size from "
                            f"{original_input} to "
                            f"{child_input_shape} and retrying."
                        )

                        print(
                            f"Too many failed attempts ({attempts}). "
                            f"Reducing input size from "
                            f"{original_input} to "
                            f"{child_input_shape} and retrying."
                        )

                        attempts = 0

                    # ---------------------------------------------
                    # Ls is fixed.
                    # ORIGINAL behavior for architecture search:
                    # continue trying other architectures.
                    # ---------------------------------------------
                    else:

                        # Architecture can still mutate:
                        # preserve original behavior.
                        if self.search_A:

                            self.log_message(
                                f"Too many failed attempts ({attempts}) "
                                f"with fixed input size "
                                f"{child_input_shape}. "
                                f"Retrying architecture mutation "
                                f"without changing Ls."
                            )

                            attempts = 0

                        # -----------------------------------------
                        # Fixed architecture + fixed Ls.
                        #
                        # There is nothing left that can change.
                        # This case did not exist in the original NAS
                        # and is only needed for the ablation flags.
                        # -----------------------------------------
                        else:

                            raise RuntimeError(
                                "Fixed architecture with fixed Ls "
                                f"{child_input_shape} cannot satisfy "
                                "the hardware constraints."
                            )

                # =================================================
                # CREATE CANDIDATE
                # =================================================

                # -------------------------------------------------
                # NORMAL / JOINT / ARCHITECTURE SEARCH
                #
                # This is the original behavior.
                # -------------------------------------------------
                if self.search_A:

                    child_net = self.mutate_network(
                        parent,
                        preloaded_data,
                        input_shape=child_input_shape
                    )

                # -------------------------------------------------
                # ABLATION: FIXED ARCHITECTURE
                # -------------------------------------------------
                else:

                    child_blocks = copy.deepcopy(
                        parent.block_list
                    )

                    child_blocks = self.correct_blocklist(
                        child_blocks,
                        input_shape=child_input_shape
                    )

                    # Strict preserve mode may reject an Ls
                    if child_blocks is None:
                        self.total_hw_rejections += 1
                        attempts += 1

                        self.log_message(
                            f"Representation rejected: "
                            f"strategy={child_strategy}, "
                            f"Ls={child_input_shape}, "
                            f"reason=fixed architecture "
                            f"incompatible with Ls"
                        )

                        continue

                    child_net = Library_Net.Net(
                        child_blocks,
                        nas_saver_name=self.nas_saver_name,
                        preloaded_data=preloaded_data
                    )

                # =================================================
                # ORIGINAL SAFETY REPAIRS
                #
                # Apply them only during architecture search.
                #
                # For preserve_architecture=True, correct_blocklist()
                # is responsible for strict validation instead.
                # =================================================
                # =================================================
                # ORIGINAL CNN SAFETY REPAIRS
                #
                # These operations are convolution-specific.
                # MLP blocks do not require kernel/stride/padding
                # repair.
                # =================================================
                if self.search_A:

                    for block in child_net.block_list:

                        block_type = getattr(
                            block,
                            "block_type",
                            "cnn"
                        ).lower()

                        if block_type == "cnn":

                            if block.input_size < 16:
                                block.padding = "same"

                            if (
                                    block.padding == "valid"
                                    and
                                    block.input_size
                                    < block.kernel_size
                            ):
                                block.padding = "same"

                            block.kernel_size = min(
                                block.kernel_size,
                                block.input_size
                            )

                            block.stride = min(
                                block.stride,
                                block.input_size
                            )

                        elif block_type == "mlp":

                            # Nothing to repair spatially.
                            #
                            # Only guarantee that units remain valid.
                            if (
                                    block.n_units is None
                                    or block.n_units < 1
                            ):
                                raise ValueError(
                                    "Invalid MLP block encountered "
                                    "during architecture mutation: "
                                    f"n_units={block.n_units}"
                                )

                # =================================================
                # HARDWARE MEASURES
                # =================================================
                hw_meas = child_net.hw_measures()

                n_params = hw_meas[0]
                max_tens = hw_meas[1]
                flops = hw_meas[2]
                flash_size = hw_meas[3]
                ram_size = hw_meas[4]

                # =================================================
                # ORIGINAL HW ACCEPTANCE TEST
                # =================================================
                if (
                        flops < self.flops_thr
                        and
                        flash_size < self.flash_thr
                        and
                        ram_size < self.ram_thr
                        and
                        n_params < self.params_thr
                        and
                        max_tens < self.max_tens_thr
                ):

                    check_pass = True

                else:

                    self.total_hw_rejections += 1
                    attempts += 1

                    self.log_message(
                        f"Hardware rejection: "
                        f"strategy={child_strategy}, "
                        f"Ls={child_input_shape}, "
                        f"params={n_params}/{self.params_thr}, "
                        f"max_tens={max_tens}/{self.max_tens_thr}, "
                        f"flops={flops}/{self.flops_thr}, "
                        f"flash={flash_size}/{self.flash_thr}, "
                        f"ram={ram_size}/{self.ram_thr}"
                    )

                    del child_net
                    gc.collect()
                    tf.keras.backend.clear_session()

        # =========================================================
        # HW CHECK DISABLED
        # =========================================================
        else:

            if self.search_A:

                child_net = self.mutate_network(
                    parent,
                    preloaded_data,
                    input_shape=child_input_shape
                )

            else:

                child_blocks = copy.deepcopy(
                    parent.block_list
                )

                child_blocks = self.correct_blocklist(
                    child_blocks,
                    input_shape=child_input_shape
                )

                if child_blocks is None:
                    raise RuntimeError(
                        "Fixed architecture is incompatible "
                        f"with Ls={child_input_shape}."
                    )

                child_net = Library_Net.Net(
                    child_blocks,
                    nas_saver_name=self.nas_saver_name,
                    preloaded_data=preloaded_data
                )

        # =========================================================
        # FINALIZE CHILD
        # =========================================================
        child_net.input_shape = (
            child_net.block_list[0].input_size,
            1
        )

        child_net.strategy = child_strategy

        self.log_message(
            f"Accepted candidate: "
            f"architecture={self.architecture_family}, "
            f"search_A={self.search_A}, "
            f"preserve_architecture={self.preserve_architecture}, "
            f"strategy={child_strategy}, "
            f"Ls={child_net.input_shape[0]}"
        )

        return child_net

    def mutate_network(self, parent, preloaded_data, input_shape):
        self.log_message('MUTATION ')

        # Deepcopy the parent's blocks to preserve the original
        parent_blocks = copy.deepcopy(parent.block_list)

        # Perform mutation action: remove, add, or change a block
        act = random.randint(0, 2)  # 0 remove, 1 change, 2 add
        if act == 0:
            self.remove_block(parent_blocks)
        elif act == 1:
            self.add_block(parent_blocks)
        elif act == 2:
            self.change_block(parent_blocks)

        #if parent_blocks:
            #parent_blocks[0].input_size = input_shape

        # Correct the block list after all mutations
        child_blocks = self.correct_blocklist(parent_blocks, input_shape=input_shape)

        self.log_message(" net_depth = " + str(len(child_blocks)) + "; ")

        # Create the new child network
        child_net = Library_Net.Net(child_blocks, nas_saver_name=self.nas_saver_name, preloaded_data=preloaded_data)

        return child_net

    def generate_random_block(
            self,
            input_size,
            block_index,
            current_architecture_depth
    ):
        """
        Generate one random block from the currently selected
        architecture family.

        CNN:
            Preserves the original CNN search space.

        MLP:
            Searches Dense units while retaining the same
            depth-dependent dropout policy used by the NAS.
        """

        # =========================================================
        # COMMON DROPOUT SCHEDULE
        # =========================================================
        base_dropout_rate = self.base_dropout_rate
        max_dropout_rate = self.max_dropout_rate

        if current_architecture_depth > 1:
            dropout_rate = (
                    base_dropout_rate
                    +
                    (max_dropout_rate - base_dropout_rate)
                    * block_index
                    / (current_architecture_depth - 1)
            )
        else:
            dropout_rate = max_dropout_rate

        dropout_rate = round(dropout_rate, 2)

        # =========================================================
        # MLP BLOCK
        # =========================================================
        if self.architecture_family == "mlp":
            n_units = random.randint(
                self.mlp_units_range[0],
                self.mlp_units_range[1]
            )

            return Library_Block.Block(
                # Compatibility alias.
                # Some existing code may still inspect n_filters.
                n_filters=n_units,

                # CNN-only fields remain available in the object
                # but are ignored by MLP create_layer().
                kernel_size=None,
                activation="relu",
                padding="valid",
                is_pool=False,
                pool_size=None,

                input_size=input_size,

                is_dropout=True,
                dropout_rate=dropout_rate,

                stride=1,
                nas_saver_name=self.nas_saver_name,

                is_max_pool=False,
                is_avg_pool=False,

                block_type="mlp",
                n_units=n_units
            )

        # =========================================================
        # CNN BLOCK
        #
        # Original search behavior preserved.
        # =========================================================

        n_filters_range = (16, 140)

        kernel_size = random.randint(
            1,
            min(7, input_size)
        )

        stride_range = (
            1,
            min(input_size, 7)
        )

        # Strict constraint:
        # avoid "valid" padding for small input sizes
        if input_size < 16:
            padding = "same"
        else:
            padding = (
                "same"
                if random.getrandbits(1)
                else "valid"
            )

        # Switch to same padding if valid is infeasible
        if (
                padding == "valid"
                and kernel_size > input_size
        ):
            padding = "same"

        kernel_size = min(
            kernel_size,
            input_size
        )

        # Prevent pooling in the first block
        if block_index == 0:

            is_pool = False
            pool_size = None
            is_max_pool = False
            is_avg_pool = False

        else:

            is_pool = bool(
                random.getrandbits(1)
            )

            min_pool_size = 2
            max_pool_size = min(
                3,
                input_size
            )

            if max_pool_size < min_pool_size:

                is_pool = False
                pool_size = None
                is_max_pool = False
                is_avg_pool = False

            else:

                pool_size = (
                    random.randint(
                        min_pool_size,
                        max_pool_size
                    )
                    if is_pool
                    else None
                )

                is_max_pool = (
                    bool(random.getrandbits(1))
                    if is_pool
                    else False
                )

                is_avg_pool = (
                    not is_max_pool
                    if is_pool
                    else False
                )

        stride = random.randint(
            *stride_range
        )

        return Library_Block.Block(
            n_filters=random.randint(
                *n_filters_range
            ),
            kernel_size=kernel_size,
            activation="relu",
            padding=padding,
            is_pool=is_pool,
            pool_size=pool_size,
            is_max_pool=is_max_pool,
            is_avg_pool=is_avg_pool,
            input_size=input_size,
            is_dropout=True,
            dropout_rate=dropout_rate,
            stride=stride,
            nas_saver_name=self.nas_saver_name,

            # Explicit for clarity/backward-safe logging
            block_type="cnn"
        )

    def remove_block(self, parent_blocks):
        self.log_message(' REMOVE')

        if len(parent_blocks) > 1:
            i_del = random.randint(1, len(parent_blocks) - 1)  # Start from 1 to preserve the input layer
            parent_blocks.pop(i_del)

            self.recalculate_dropout_rates(parent_blocks)
        else:
            self.log_message(" FORBIDDEN -> CHANGED")
            i_mod = random.randint(0, len(parent_blocks) - 1)
            parent_blocks[i_mod] = self.generate_random_block(parent_blocks[0].input_size, i_mod, len(parent_blocks))

            self.recalculate_dropout_rates(parent_blocks)

        # Correct blocklist after changes
        self.correct_blocklist(parent_blocks, input_shape=parent_blocks[0].input_size)

        return parent_blocks

    def change_block(self, parent_blocks):
        self.log_message(' CHANGE')

        if parent_blocks:
            i_mod = random.randint(0, len(parent_blocks) - 1)
            parent_blocks[i_mod] = self.generate_random_block(parent_blocks[0].input_size, i_mod, len(parent_blocks))

            self.recalculate_dropout_rates(parent_blocks)

        # Correct blocklist after changes
        self.correct_blocklist(parent_blocks, input_shape=parent_blocks[0].input_size)

        return parent_blocks

    def add_block(self, parent_blocks):
        self.log_message(' ADD')

        if len(parent_blocks) < self.max_depth:
            i_add = random.randint(0, len(parent_blocks))
            reset_input_size = parent_blocks[-1].output_size if parent_blocks else parent.input_shape[0]

            new_block = self.generate_random_block(reset_input_size, len(parent_blocks), len(parent_blocks) + 1)
            if i_add == len(parent_blocks):
                parent_blocks.append(new_block)
            else:
                parent_blocks.insert(i_add, new_block)

            self.recalculate_dropout_rates(parent_blocks)

        # Correct blocklist after changes
        self.correct_blocklist(parent_blocks, input_shape=parent_blocks[0].input_size)

        return parent_blocks

    def recalculate_dropout_rates(self, blocks):
        current_architecture_depth = len(blocks)
        for i, block in enumerate(blocks):
            if current_architecture_depth > 1:
                dropout_rate = self.base_dropout_rate + (self.max_dropout_rate - self.base_dropout_rate) * i / (
                        current_architecture_depth - 1)
            else:
                dropout_rate = self.max_dropout_rate
            block.dropout_rate = round(dropout_rate, 2)

    def correct_blocklist(self, child_blocks, input_shape):
        """
        Recalculate block input/output dimensions.

        CNN:
            Preserve the existing convolution/pooling repair rules.

        MLP:
            Propagate:
                first input_size = Ls
                next input_size  = previous Dense units
                output_size      = current Dense units

            No kernel/stride/padding/pooling repair is required.

        If self.preserve_architecture is True:
            architecture hyperparameters remain frozen and only
            tensor dimensions are propagated/validated.
        """

        # ---------------------------------------------------------
        # Convert input shape to scalar
        # ---------------------------------------------------------
        if isinstance(input_shape, tuple):
            scalar_input_size = input_shape[0]
        else:
            scalar_input_size = input_shape

        scalar_input_size = int(
            scalar_input_size
        )

        # ---------------------------------------------------------
        # Empty architecture should never be accepted.
        # ---------------------------------------------------------
        if not child_blocks:
            raise ValueError(
                "correct_blocklist received an empty block list."
            )

        # ---------------------------------------------------------
        # Ensure the block family agrees with the NAS family.
        #
        # Old blocks without block_type are interpreted as CNN.
        # ---------------------------------------------------------
        for i, block in enumerate(child_blocks):

            block_type = getattr(
                block,
                "block_type",
                "cnn"
            ).lower()

            if block_type != self.architecture_family:
                raise ValueError(
                    f"Architecture-family mismatch at block {i}: "
                    f"NAS family={self.architecture_family}, "
                    f"block_type={block_type}."
                )

        # =========================================================
        # STRICT FIXED-ARCHITECTURE MODE
        # =========================================================
        if self.preserve_architecture:

            if (
                    scalar_input_size < 196
                    or scalar_input_size > 784
            ):
                self.log_message(
                    f"Rejected Ls={scalar_input_size}: "
                    f"outside allowed range [196, 784]."
                )

                return None

            for i, block in enumerate(child_blocks):

                # -------------------------------------------------
                # Only tensor dimensions are propagated.
                # -------------------------------------------------
                if i == 0:
                    block.input_size = scalar_input_size
                else:
                    block.input_size = (
                        child_blocks[i - 1].output_size
                    )

                if (
                        block.input_size is None
                        or block.input_size < 1
                ):
                    self.log_message(
                        f"Fixed architecture infeasible "
                        f"at block {i}: "
                        f"input_size={block.input_size}"
                    )

                    return None

                block_type = getattr(
                    block,
                    "block_type",
                    "cnn"
                ).lower()

                # =================================================
                # FIXED MLP
                # =================================================
                if block_type == "mlp":

                    # Architecture itself is frozen:
                    # n_units/dropout/activation are NOT modified.

                    if (
                            block.n_units is None
                            or block.n_units < 1
                    ):
                        self.log_message(
                            f"Fixed MLP architecture infeasible "
                            f"at block {i}: "
                            f"n_units={block.n_units}"
                        )

                        return None

                    try:
                        block.output_size = (
                            block.calculate_output_size()
                        )

                    except Exception as e:

                        self.log_message(
                            f"Fixed MLP architecture infeasible "
                            f"at block {i}: "
                            f"calculate_output_size() failed: {e}"
                        )

                        return None

                    if (
                            block.output_size is None
                            or block.output_size < 1
                    ):
                        self.log_message(
                            f"Fixed MLP architecture infeasible "
                            f"at block {i}: "
                            f"output_size={block.output_size}"
                        )

                        return None

                    # MLP has no spatial pooling.
                    block.is_pool = False
                    block.is_max_pool = False
                    block.is_avg_pool = False

                    continue

                # =================================================
                # FIXED CNN
                #
                # Existing strict behavior preserved.
                # =================================================

                if (
                        block.kernel_size is None
                        or block.kernel_size < 1
                ):
                    self.log_message(
                        f"Fixed architecture infeasible "
                        f"at block {i}: "
                        f"kernel_size={block.kernel_size}"
                    )

                    return None

                if (
                        block.stride is None
                        or block.stride < 1
                ):
                    self.log_message(
                        f"Fixed architecture infeasible "
                        f"at block {i}: "
                        f"stride={block.stride}"
                    )

                    return None

                if (
                        block.padding == "valid"
                        and
                        block.kernel_size > block.input_size
                ):
                    self.log_message(
                        f"Fixed architecture infeasible "
                        f"at block {i}: "
                        f"padding=valid, "
                        f"input_size={block.input_size}, "
                        f"kernel_size={block.kernel_size}"
                    )

                    return None

                try:

                    block.output_size = (
                        block.calculate_output_size()
                    )

                except Exception as e:

                    self.log_message(
                        f"Fixed architecture infeasible "
                        f"at block {i}: "
                        f"calculate_output_size() failed: {e}"
                    )

                    return None

                if (
                        block.output_size is None
                        or block.output_size < 1
                ):
                    self.log_message(
                        f"Fixed architecture infeasible "
                        f"at block {i}: "
                        f"output_size={block.output_size}"
                    )

                    return None

                if block.is_pool:

                    if (
                            block.pool_size is None
                            or block.pool_size < 1
                    ):
                        self.log_message(
                            f"Fixed architecture infeasible "
                            f"at block {i}: "
                            f"pool_size={block.pool_size}"
                        )

                        return None

            return child_blocks

        # =========================================================
        # NORMAL NAS MODE
        # =========================================================

        scalar_input_size = max(
            196,
            min(784, scalar_input_size)
        )

        # =========================================================
        # NORMAL MLP NAS
        # =========================================================
        if self.architecture_family == "mlp":

            for i, block in enumerate(child_blocks):

                # -------------------------------------------------
                # Propagate dimensions
                # -------------------------------------------------
                if i == 0:

                    block.input_size = scalar_input_size

                    print(
                        f"Debug: In correct_blocklist, "
                        f"MLP block[0].validated_input_size="
                        f"{block.input_size}"
                    )

                else:

                    block.input_size = (
                        child_blocks[i - 1].output_size
                    )

                # -------------------------------------------------
                # MLP blocks never use pooling.
                # -------------------------------------------------
                block.is_pool = False
                block.is_max_pool = False
                block.is_avg_pool = False

                # -------------------------------------------------
                # Validate Dense units
                # -------------------------------------------------
                if (
                        block.n_units is None
                        or block.n_units < 1
                ):
                    raise ValueError(
                        f"Invalid MLP block {i}: "
                        f"n_units={block.n_units}"
                    )

                # -------------------------------------------------
                # For MLP:
                # output_size = n_units
                # -------------------------------------------------
                block.output_size = (
                    block.calculate_output_size()
                )

            return child_blocks

        # =========================================================
        # NORMAL CNN NAS
        #
        # Existing behavior preserved below.
        # =========================================================

        forbid_pool = False

        for i in range(len(child_blocks)):

            # -----------------------------------------------------
            # First block
            # -----------------------------------------------------
            if i == 0:

                child_blocks[i].input_size = (
                    scalar_input_size
                )

                print(
                    f"Debug: In correct_blocklist, "
                    f"block[0].validated_input_size = "
                    f"{child_blocks[i].input_size}"
                )

                child_blocks[i].is_pool = False
                child_blocks[i].is_max_pool = False
                child_blocks[i].is_avg_pool = False

            # -----------------------------------------------------
            # Subsequent block input
            # -----------------------------------------------------
            else:

                child_blocks[i].input_size = (
                    child_blocks[i - 1].output_size
                )

            # -----------------------------------------------------
            # Original CNN repair rules
            # -----------------------------------------------------
            if child_blocks[i].input_size < 16:
                child_blocks[i].padding = "same"

            if (
                    child_blocks[i].padding == "valid"
                    and
                    child_blocks[i].input_size
                    < child_blocks[i].kernel_size
            ):
                child_blocks[i].padding = "same"

            child_blocks[i].kernel_size = min(
                child_blocks[i].kernel_size,
                child_blocks[i].input_size
            )

            child_blocks[i].stride = min(
                child_blocks[i].stride,
                child_blocks[i].input_size
            )

            # -----------------------------------------------------
            # Output dimension
            # -----------------------------------------------------
            child_blocks[i].output_size = (
                child_blocks[i].calculate_output_size()
            )

            # -----------------------------------------------------
            # Original pooling repair
            # -----------------------------------------------------
            if child_blocks[i].input_size < 32:
                forbid_pool = True

            if forbid_pool:

                child_blocks[i].is_pool = False
                child_blocks[i].is_max_pool = False
                child_blocks[i].is_avg_pool = False

                child_blocks[i].output_size = (
                    child_blocks[i].calculate_output_size()
                )

            elif child_blocks[i].is_pool:

                if (
                        child_blocks[i].pool_size is None
                        or
                        child_blocks[i].output_size
                        < child_blocks[i].pool_size
                ):
                    child_blocks[i].is_pool = False
                    child_blocks[i].is_max_pool = False
                    child_blocks[i].is_avg_pool = False

                    child_blocks[i].output_size = (
                        child_blocks[i].calculate_output_size()
                    )

        return child_blocks

    import os
    def save_partial(self, parent, best_network, i_gen, gen_best_score, absolute_best_score):
        """
        Save partial results, including the parent and best network models, along with logging.
        """
        # Ensure the directory exists for saving models
        save_dir = "../sessions_nas/Models"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Get input_shape from the parent network
        input_shape = getattr(parent, 'input_shape', None)
        if input_shape is None:
            raise ValueError("Input shape is not defined for the parent network.")

        # Save the parent model
        model_parent = parent.ins_keras_model(input_shape=input_shape)
        model_parent.save(f"{save_dir}/{self.nas_saver_name}_Partial_parent.h5")

        # Save the best network model
        model_best_network = best_network.ins_keras_model(input_shape=input_shape)
        model_best_network.save(f"{save_dir}/{self.nas_saver_name}_Partial_best_network.h5")

        # Log information about the saving process
        self.log_message(f"Saved parent model at generation {i_gen} with score {gen_best_score:.2f}")
        self.log_message(f"Saved best network at generation {i_gen} with absolute best score {absolute_best_score:.2f}")

    def log_message(self, message, mode='a'):
        with open(self.nas_saver_name + '.txt', mode) as log_file:
            log_file.write(message + '\n')
