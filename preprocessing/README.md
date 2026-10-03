# Session preprocessing

This folder contains the script that turns raw traffic into the inputs the search framework consumes. It extracts bidirectional sessions from `.pcap` files, applies a header preprocessing strategy, and writes fixed-length session bytes in IDX format.

Run it once per strategy. Each run produces the file pair that the search selects between.

---

## 🔧 What it does

Packets belonging to the same bidirectional session are grouped by their endpoint tuple, control packets and DNS traffic are filtered out, duplicate sessions within a class are dropped, and each session is truncated or zero-padded to a fixed length. Labels come from keywords in the file names, through a mapping defined at the top of the script.

Sessions are written as 28x28 byte matrices, which makes them easy to inspect as grayscale images. The search framework reads them back as flat sequences.

---

## 🧪 Header strategies

The transformation applied to each packet is set by the defaults of `extract_packet_data()`:

| Argument | Options |
|---|---|
| `mac_strategy` | `remove`, `zero`, `anonymize` |
| `ip_strategy` | `anonymize`, `zero`, `remove` |
| `zero_ports` | `True`, `False` |

Anonymization replaces an address with a short hash of itself, so the field keeps a consistent value without exposing the original address.

---

## ▶️ Usage

1. Edit `label_mapping` so that the keywords match your file names.
2. Set your input and output paths in `main()`.
3. Choose the strategy in `extract_packet_data()` and run:

```bash
python session_preprocessing.py
```

Repeat for each strategy you want the search to consider, writing to `strategy1`, `strategy2`, and so on.

Processing runs in parallel across files. Adjust `max_workers` in `main()` to match your machine.

---

## 📚 Requirements

```bash
pip install scapy numpy psutil
```
