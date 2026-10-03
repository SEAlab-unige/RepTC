# RepTC framework

This folder contains the search engine. It builds candidate networks from modular blocks, measures their hardware cost, and evolves the architecture together with the input representation under the limits you set.

---

## 🔍 What is searched

**Architecture.** Candidates are stacks of blocks. A CNN block is a 1-D convolution followed by batch normalization, activation, optional pooling, and dropout. The search tunes filters, kernel size, stride, pooling, and padding, and adds or removes blocks up to a maximum depth. The framework is built around 1-D CNNs, which suit raw byte sequences well, but the block abstraction is not tied to them. MLP blocks are already available, and other families can be added by extending `Library_Block.py`.

**Input length.** How many bytes of each session the model reads. Longer inputs carry more context and cost more memory and compute. The search mutates this value within a configurable range.

**Header preprocessing.** Which transformation is applied to the header fields: Ethernet removal, MAC zeroing, IP anonymization, IP zeroing, port zeroing, and their combinations. Each variant is a separate preprocessed copy of the dataset, and the search switches between them like any other variable.

Every candidate is checked against the hardware limits before training, and the ones that exceed any limit are discarded. The admissible child with the best validation accuracy becomes the parent of the next generation, and the best model found overall is tracked across the run.

---

## 📂 Files

**`B01_NAS.py`**: the entry point. Loads the data, configures the search, runs it, and reports wall-clock time.

**`Library_NAS.py`**: the search loop. Mutation of blocks, input length and preprocessing strategy, hardware filtering, selection, logging, and periodic checkpoints.

**`Library_Net.py`**: builds a Keras model from a block list, trains it with a full or proxy routine using cross-validation, and computes its hardware cost: parameters, peak intermediate tensor, FLOPs, Flash, and RAM.

**`Library_Block.py`**: the block abstraction. Defines CNN and MLP blocks, computes their output size, and emits the corresponding Keras layers.

**`Library_load_and_split_data.py`**: reads the session files, normalizes bytes to `[0,1]`, reshapes each session into a flat sequence, and builds stratified folds.

**`Library_compute_stats.py`**: accuracy, precision, recall, and F1.

---

## 💾 Data

The framework expects one pair of files per preprocessing strategy:

```
strategy1.idx3  strategy1.idx1
strategy2.idx3  strategy2.idx1
...
```

Sessions are stored in the IDX layout, which keeps them viewable as small grayscale images for quick inspection. At load time each session is read back as a flat byte sequence of the length the search currently uses, so the 2-D layout plays no role in the model.

Set the folder in `Library_load_and_split_data.py`:

```python
base_dir = r'/put/your/data/directory/here'
```

Use [`preprocessing/`](../preprocessing/) to generate these files from your own `.pcap` traffic, once per strategy.

---

## 🎛️ Configuring a run

All settings are variables at the top of `B01_NAS.py` and in the `NAS(...)` call below it. The values shown here are the ones currently in the file; change them to suit your target device and your dataset.

**Hardware limits.** Each limit is checked before a candidate is trained, and candidates that exceed any of them are discarded. Set them to whatever your target device allows:

```python
params_thr   = 60000     # parameters, maps to Flash
max_tens_thr = 4000      # peak intermediate tensor, maps to RAM
flops_thr    = 500000    # compute per inference
flash_thr    = 250000    # bytes
ram_thr      = 60000     # bytes
```

Set a limit to `np.inf` to leave that resource unconstrained.

**What the search is free to change.** Each of these switches one variable on or off:

```python
search_A  = True    # architecture
search_Sh = True    # header preprocessing
search_Ls = True    # input length

fixed_Sh  = "strategy2"   # value used when search_Sh = False
fixed_Ls  = 784           # value used when search_Ls = False
```

When all three are `True` the three variables evolve together. Turning one off holds it at its fixed value, which reduces the problem to a conventional search over the rest and is useful for isolating what joint optimization contributes.

**Architecture family:**

```python
architecture_family = "cnn"       # or "mlp"
mlp_units_range     = (16, 128)   # used only for "mlp"
```

Depth is searched in both cases, since blocks are added and removed during the search, bounded by `max_depth`.

**Search effort:**

```python
n_generations = 50    # generations
n_child       = 20    # candidates per generation
n_mutations   = 2     # mutations applied to each candidate
max_depth     = 4     # maximum number of blocks
```

**Training mode.** Two separate settings:

```python
use_full_training = False   # False: proxy routine, True: full routine
is_train_proxy    = True    # actually fit the model in the proxy routine
is_train          = False   # actually fit the model in the full routine
```

`use_full_training` chooses the routine used to evaluate candidates. The proxy routine trains on a single fold with an internal validation split and is much faster, so it is the default during the search. The full routine trains across all folds. The two `is_train` flags control whether the chosen routine actually calls `fit()`; setting them to `False` runs the loop without training, which is handy for checking the hardware filtering and the search mechanics on a new dataset.

**Dataset size.** Set `num_classes` in `B01_NAS.py` to match your dataset. The default is 11, for ISCX VPN-nonVPN.

---

## 📊 Output

Each run writes two timestamped logs:

- `NAS_logger_*.txt`: per-generation progress, the strategy and input length of every child, validation and test metrics, and the selected parent.
- `Partial_saver_logger_*.txt`: periodic checkpoints of the current parent and the best model found so far.
