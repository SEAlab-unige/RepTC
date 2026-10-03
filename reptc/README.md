# RepTC framework

The search engine. It builds candidate networks from modular blocks, measures their hardware cost, and evolves architecture and input representation together under the limits you set.

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
