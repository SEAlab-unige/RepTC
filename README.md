# RepTC

**Representation-Aware Optimization for Efficient Traffic Classification on Edge IoT Devices**

Representation-aware neural architecture search for session-level traffic classification under microcontroller-class resource limits.

Traffic classifiers are usually built by fixing the input representation in advance, deciding how many bytes of a session the model reads and how header fields are anonymized or stripped, and then searching only the network architecture. Those choices are not independent. The representation decides what information reaches the model and how much memory and compute the model needs to process it.

RepTC searches them together. A single evolutionary loop explores the architecture, the session input length, and the header preprocessing strategy under explicit hardware limits, and returns a configuration that fits the budget you set.

---

## 📦 Repository structure

### 🧠 [`reptc/`](./reptc/)
The search framework: modular blocks, network builder, hardware measures, and the evolutionary loop that mutates architecture and input representation together.
➡️ [details](./reptc/README.md)

### 📡 [`preprocessing/`](./preprocessing/)
Turns raw `.pcap` traffic into fixed-length session byte sequences, with configurable handling of MAC addresses, IP addresses, and ports. One run per preprocessing strategy.
➡️ [details](./preprocessing/README.md)

---

## ⚙️ How it works

1. **Preprocess.** Extract bidirectional sessions from your `.pcap` files, once per preprocessing strategy, into `.idx3` and `.idx1` files.
2. **Search.** Run the NAS loop. At each generation it mutates the architecture, the input length, and the preprocessing strategy, discards every candidate that violates the hardware limits, trains the admissible ones, and keeps the best.
3. **Deploy.** The selected model is small enough to quantize and run on a microcontroller or an edge gateway.

---

## 🚀 Quick start

```bash
git clone https://github.com/SEAlab-unige/RepTC.git
cd RepTC
```

```bash
# 1. sessions from pcap
cd preprocessing
python session_preprocessing.py

# 2. search
cd ../reptc
python B01_NAS.py
```

Set your own data directories in `preprocessing/session_preprocessing.py` and `reptc/Library_load_and_split_data.py` before running.

---

## 📚 Requirements

Python 3.x

```bash
pip install tensorflow keras-flops scikit-learn numpy scapy psutil
```

---

## 📄 Citation

A preprint is under review at arXiv. The citation will be added here once it is announced.

Related work from the same group: [ProtectIT_Unige](https://github.com/SEAlab-unige/ProtectIT_Unige), hardware-aware NAS for encrypted traffic classification under a fixed input representation.
