# RepTC

**Rep**resentation-aware **T**raffic **C**lassification.

This repository contains a search framework that designs traffic classifiers for devices with very little memory and compute, such as microcontrollers and edge gateways.

Traffic classifiers are usually built by fixing the input representation in advance, deciding how many bytes of a session the model reads and how header fields are anonymized or stripped, and then optimizing only the network architecture. Those choices are not independent. The representation decides what information reaches the model and how much memory and compute the model needs to process it.

RepTC optimizes them together. A single evolutionary loop explores the architecture, the session input length, and the header preprocessing strategy under explicit hardware limits, and returns a configuration that fits the budget you set. Typical results are models of a few tens of thousands of parameters, well under 100 kB of Flash and a few tens of kB of RAM.

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
2. **Search.** Run the search loop. At each generation it mutates the architecture, the input length, and the preprocessing strategy, discards every candidate that violates the hardware limits, trains the admissible ones, and keeps the best.
3. **Deploy.** The selected model is small enough to quantize and run on a microcontroller or an edge gateway.

The framework is dataset-agnostic: any `.pcap` capture works. The included defaults target the common benchmarks in this area, such as ISCX VPN-nonVPN, USTC-TFC2016, and Edge-IIoTset.

---

## 🚀 Quick start

```bash
git clone https://github.com/SEAlab-unige/RepTC.git
cd RepTC
```

Before running, set your own paths: the `.pcap` folders and output files at the top of `preprocessing/session_preprocessing.py`, and `BASE_DIR` in `reptc/Library_load_and_split_data.py`.

```bash
# 1. sessions from pcap
cd preprocessing
python session_preprocessing.py

# 2. search
cd ../reptc
python B01_NAS.py
```

---

## 📚 Requirements

Python 3.x

```bash
pip install tensorflow keras-flops scikit-learn numpy scapy psutil
```

---

## 📄 Citation

A preprint is under review at arXiv. The citation will be added here once it is announced.

Related work from the same group: [ProtectIT_Unige](https://github.com/SEAlab-unige/ProtectIT_Unige), hardware-aware architecture search for encrypted traffic classification under a fixed input representation.

---

**Keywords:** encrypted traffic classification, network traffic analysis, session-level classification, hardware-aware model design, TinyML, edge AI, IoT security, intrusion detection, microcontroller deployment, pcap, TensorFlow, Keras.
