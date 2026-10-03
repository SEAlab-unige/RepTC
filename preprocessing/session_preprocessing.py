import os
from scapy.all import rdpcap, IP, TCP, UDP, IPv6, Ether
import numpy as np
import struct
from scapy.utils import PcapReader
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
from hashlib import sha256
import traceback
import socket
import psutil
from scapy.packet import Raw
from scapy.compat import raw
from scapy.all import IP, TCP, UDP, ICMP, ARP, Ether, Raw
import gc


# ===============================================================
# Configuration
#
# Replace the paths with your own directories before running.
# ===============================================================

# Folders scanned recursively for .pcap files
PCAP_DIRECTORIES = [
    r'/put/your/pcap/directory/here',
]

# Output files, one pair per preprocessing strategy
OUTPUT_IDX3 = r'/put/your/output/directory/here/strategy2.idx3'
OUTPUT_IDX1 = r'/put/your/output/directory/here/strategy2.idx1'

# Header preprocessing strategy applied to every packet.
#   MAC_STRATEGY: 'remove', 'zero', 'anonymize'
#   IP_STRATEGY : 'anonymize', 'zero', 'remove'
#   ZERO_PORTS  : True or False
# The defaults below correspond to Ethernet removal with IP anonymization.
MAC_STRATEGY = 'remove'
IP_STRATEGY = 'anonymize'
ZERO_PORTS = False

# Bytes kept per session. Longer sessions are truncated, shorter ones
# are zero-padded. 784 fills a 28x28 matrix.
SESSION_LENGTH = 784

# Parallel workers, set to match your machine
MAX_WORKERS = 16



label_mapping = {
    # Normal traffic (Benign files)
    'benign_distance': ['distance'],
    'benign_flame_sensor': ['flame_sensor'],
    'benign_heart_rate': ['heart_rate'],
    'benign_ir_receiver': ['ir_receiver'],
    'benign_modbus': ['modbus'],
    'benign_phvalue': ['phvalue'],
    'benign_soil_moisture': ['soil_moisture'],
    'benign_sound_sensor': ['sound_sensor'],
    'benign_temperature_and_humidity': ['temperature_and_humidity'],
    'benign_water_level' : ['water_level'],

    # Attack traffic (Attack files)
    'attack_backdoor': ['backdoor_attack'],
    'attack_ddos_http_flood': ['ddos http flood'],
    'attack_ddos_icmp_flood': ['ddos icmp flood'],
    'attack_ddos_tcp_syn_flood': ['ddos tcp syn flood'],
    'attack_ddos_udp_flood': ['ddos udp flood'],
    'attack_mitm': ['mitm'],
    'attack_os_fingerprinting': ['os fingerprinting attack'],
    'attack_password': ['password attacks'],
    'attack_port_scanning': ['port scanning attack'],
    'attack_ransomware': ['ransomware attack'],
    'attack_sql_injection': ['sql injection'],
    'attack_uploading': ['uploading attack'],
    'attack_vulnerability_scanner': ['vulnerability scanner attack'],
    'attack_xss': ['xss attacks']
}



# Map string labels to integers
label_to_int = {label: i for i, label in enumerate(label_mapping)}
label_to_int["unknown"] = 255  # Assign a valid ubyte value for unknown labels

# Print the label-to-integer mapping
for label, int_value in label_to_int.items():
    print(f"'{label}': {int_value}")

def get_label_from_filename(filepath):
    """Assign a label to a file based on its name or parent folder."""
    filename = os.path.basename(filepath).lower()  # Extract filename
    foldername = os.path.basename(os.path.dirname(filepath)).lower()  # Extract parent folder

    for label, keywords in label_mapping.items():
        # Check if any keyword is in the filename or foldername
        if any(keyword in filename or keyword in foldername for keyword in keywords):
            return label
    return "unknown"


def pad_udp_header(packet):
    """Pad UDP headers to 20 bytes to match TCP headers for deep learning model, without overwriting the payload."""
    if UDP in packet:
        udp_layer = packet[UDP]
        udp_header_len = len(raw(udp_layer))  # Get the length of the raw UDP header

        # UDP headers are typically 8 bytes, so we pad it up to 20 bytes
        if udp_header_len < 20:
            padding_needed = 20 - udp_header_len
            packet[UDP].payload = Raw(b'\x00' * padding_needed) / udp_layer.payload
    return packet

def anonymize_mac(mac_address):
    """Anonymize a MAC address using SHA-256 and return the first 8 characters of the hash."""
    return hashlib.sha256(mac_address.encode()).hexdigest()[:12]


def zero_mac(mac_address):
    """Return a '00:00:00:00:00:00' string to represent a zeroed MAC address."""
    return "00:00:00:00:00:00"


def zero_ip(ip_address):
    """Return a '0.0.0.0' string to represent a zeroed IP."""
    return "0.0.0.0"


def anonymize_ip(ip_address):
    """Anonymize an IPv4 address using SHA-256 and return the first 8 characters of the hash."""
    if '.' in ip_address:  # IPv4 check
        return hashlib.sha256(ip_address.encode()).hexdigest()[:8]
    else:
        return "unknown"


def create_session_key(packet):
    # ARP (Ethernet level)
    if ARP in packet:
        return ('ARP', packet[ARP].psrc, packet[ARP].pdst, packet[ARP].hwsrc, packet[ARP].hwdst)

    # ICMP (IP, no TCP/UDP)
    if ICMP in packet and IP in packet:
        return ('ICMP', packet[IP].src, packet[IP].dst, packet[ICMP].type, packet[ICMP].code)

    # TCP or UDP
    if IP in packet:
        proto = TCP if TCP in packet else UDP if UDP in packet else None
        if proto:
            ips = sorted([packet[IP].src, packet[IP].dst])
            ports = sorted([packet[proto].sport, packet[proto].dport])
            return (ips[0], ips[1], ports[0], ports[1], proto.name)

    return None



def extract_packet_data(packet, mac_strategy='remove', ip_strategy='anonymize',
                        zero_ports=False, pad_udp=False, arp_mac_fallback='zero'):
    packet_data = raw(packet)
    eth_present = Ether in packet
    eth_header_len = 14 if eth_present else 0

    # Handle ARP packets explicitly:
    if ARP in packet:
        if not eth_present:
            raise ValueError("ARP packet explicitly requires Ethernet header.")

        effective_mac_strategy = arp_mac_fallback if mac_strategy == 'remove' else mac_strategy
        if effective_mac_strategy not in ['anonymize', 'zero']:
            raise ValueError(f"Invalid ARP MAC fallback '{arp_mac_fallback}'")

        # Anonymize or zero MAC explicitly (Ethernet & ARP payload)
        if effective_mac_strategy == 'anonymize':
            eth_src_mac_bin = bytes.fromhex(anonymize_mac(packet[Ether].src))
            eth_dst_mac_bin = bytes.fromhex(anonymize_mac(packet[Ether].dst))
            arp_src_mac_bin = bytes.fromhex(anonymize_mac(packet[ARP].hwsrc))
            arp_dst_mac_bin = bytes.fromhex(anonymize_mac(packet[ARP].hwdst))
        elif effective_mac_strategy == 'zero':
            zero_mac_bin = b'\x00\x00\x00\x00\x00\x00'
            eth_src_mac_bin = eth_dst_mac_bin = arp_src_mac_bin = arp_dst_mac_bin = zero_mac_bin

        # Ethernet MAC replacement explicitly
        packet_data = eth_dst_mac_bin + eth_src_mac_bin + packet_data[12:]

        # Replace MAC in ARP payload explicitly
        arp_src_mac_offset = eth_header_len + 8
        arp_dst_mac_offset = eth_header_len + 18
        packet_data = (
            packet_data[:arp_src_mac_offset] + arp_src_mac_bin +
            packet_data[arp_src_mac_offset + 6:arp_dst_mac_offset] + arp_dst_mac_bin +
            packet_data[arp_dst_mac_offset + 6:]
        )

        # Handle ARP IP explicitly in payload
        arp_src_ip_offset = eth_header_len + 14
        arp_dst_ip_offset = eth_header_len + 24

        if ip_strategy == 'anonymize':
            src_ip_bin = bytes.fromhex(anonymize_ip(packet[ARP].psrc))
            dst_ip_bin = bytes.fromhex(anonymize_ip(packet[ARP].pdst))
        elif ip_strategy == 'zero':
            src_ip_bin = dst_ip_bin = b'\x00\x00\x00\x00'
        elif ip_strategy == 'remove':
            src_ip_bin = dst_ip_bin = b'\xff\xff\xff\xff'
        else:
            raise ValueError(f"Invalid IP strategy '{ip_strategy}'")

        packet_data = (
            packet_data[:arp_src_ip_offset] + src_ip_bin +
            packet_data[arp_src_ip_offset + 4:arp_dst_ip_offset] + dst_ip_bin +
            packet_data[arp_dst_ip_offset + 4:]
        )

        return packet_data  # ARP packet explicitly handled

    # Handle Ethernet for IP packets (ICMP, TCP, UDP)
    if eth_present:
        if mac_strategy == 'remove':
            packet_data = packet_data[14:]
            eth_present = False
            eth_header_len = 0
        elif mac_strategy in ('anonymize', 'zero'):
            if mac_strategy == 'anonymize':
                eth_src_mac_bin = bytes.fromhex(anonymize_mac(packet[Ether].src))
                eth_dst_mac_bin = bytes.fromhex(anonymize_mac(packet[Ether].dst))
            elif mac_strategy == 'zero':
                eth_src_mac_bin = eth_dst_mac_bin = b'\x00\x00\x00\x00\x00\x00'
            packet_data = eth_dst_mac_bin + eth_src_mac_bin + packet_data[12:]

    # IP addresses explicitly
    if IP in packet:
        ip_header_offset = eth_header_len if eth_present else 0
        ip_src_offset = ip_header_offset + 12
        ip_dst_offset = ip_header_offset + 16

        if ip_strategy == 'anonymize':
            src_ip_bin = bytes.fromhex(anonymize_ip(packet[IP].src))
            dst_ip_bin = bytes.fromhex(anonymize_ip(packet[IP].dst))
        elif ip_strategy == 'zero':
            src_ip_bin = dst_ip_bin = b'\x00\x00\x00\x00'
        elif ip_strategy == 'remove':
            src_ip_bin = dst_ip_bin = b'\xff\xff\xff\xff'

        packet_data = (
            packet_data[:ip_src_offset] + src_ip_bin +
            packet_data[ip_src_offset + 4:ip_dst_offset] + dst_ip_bin +
            packet_data[ip_dst_offset + 4:]
        )

        # Explicit zeroing ports for TCP/UDP if asked
        if zero_ports and (TCP in packet or UDP in packet):
            transport_offset = eth_header_len + packet[IP].ihl * 4
            src_port_offset = transport_offset
            dst_port_offset = transport_offset + 2
            packet_data = (
                packet_data[:src_port_offset] + b'\x00\x00' +
                packet_data[src_port_offset + 2:dst_port_offset] + b'\x00\x00' +
                packet_data[dst_port_offset + 2:]
            )

    # Explicit optional UDP padding
    if pad_udp and UDP in packet:
        packet = pad_udp_header(packet)
        packet_data = raw(packet)
        if mac_strategy == 'remove' and Ether in packet:
            packet_data = packet_data[14:]

    return packet_data





def is_irrelevant_packet(packet):
    if ARP in packet:
        return False  # ARP explicitly included
    if ICMP in packet:
        return False  # ICMP explicitly included
    if IP in packet:
        if TCP in packet and len(packet[TCP].payload) == 0 and packet[TCP].flags & 0x13:
            return True  # TCP empty handshake packets
        if UDP in packet and (packet[UDP].dport == 53 or packet[UDP].sport == 53):
            return True  # DNS packet filtering
        return False
    return True  # Not IP, ICMP, ARP

def extract_sessions(pcap_file, length=SESSION_LENGTH):
    """Extract and process sessions efficiently without memory overflow."""
    sessions = {}
    unique_sessions = {}
    original_lengths = []
    truncated = padded = used_full_length = 0

    with PcapReader(pcap_file) as packets:
        for packet in packets:
            if is_irrelevant_packet(packet):
                continue

            session_key = create_session_key(packet)
            if session_key is None:
                continue

            packet_data = extract_packet_data(
                packet,
                mac_strategy=MAC_STRATEGY,
                ip_strategy=IP_STRATEGY,
                zero_ports=ZERO_PORTS
            )

            # Append data to session
            sessions.setdefault(session_key, bytearray()).extend(packet_data)

    # De-duplicate sessions
    for session_key, session_data in sessions.items():
        if not session_data:
            continue
        session_hash = hashlib.sha256(session_data).hexdigest()
        if session_hash not in unique_sessions:
            unique_sessions[session_hash] = (session_key, session_data)
            original_lengths.append(len(session_data))

    final_sessions = {k: v[1] for k, v in unique_sessions.items()}

    # Adjust session length
    for session_key, session_data in final_sessions.items():
        session_length = len(session_data)
        if session_length > length:
            truncated += 1
            final_sessions[session_key] = session_data[:length]
        elif session_length < length:
            padded += 1
            final_sessions[session_key] = session_data.ljust(length, b'\x00')
        else:
            used_full_length += 1

    average_length = sum(original_lengths) / len(original_lengths) if original_lengths else 0

    print(f"Sessions Total: {len(final_sessions)}, Truncated: {truncated}, Padded: {padded}, "
          f"Used Full Length: {used_full_length}, Average Original Length: {average_length:.2f}")

    stats = {
        'Sessions Total': len(final_sessions),
        'Truncated': truncated,
        'Padded': padded,
        'Used Full Length': used_full_length,
        'Average Original Length': average_length
    }

    return final_sessions, stats


def extract_sessions_and_label(pcap_file, length=SESSION_LENGTH):
    label_str = get_label_from_filename(os.path.basename(pcap_file))
    label_int = label_to_int.get(label_str, 255)
    sessions, stats = extract_sessions(pcap_file, length)
    labels = [label_int] * len(sessions)
    return sessions, labels, stats


def convert_sessions_to_matrices(sessions):
    """Generator to process session data into 28x28 matrices one at a time."""
    for session_data in sessions.values():
        yield np.array(list(session_data), dtype=np.uint8).reshape(28, 28)


def save_to_idx3(matrices, filename):
    mode = 'ab' if os.path.exists(filename) else 'wb'
    with open(filename, mode) as file:
        if mode == 'wb':
            file.write(struct.pack('>IIII', 2051, 0, 28, 28))  # Header
        for matrix in matrices:
            file.write(matrix.astype(np.uint8).tobytes())
            del matrix  # Free memory


def save_to_idx1(labels, filename):
    mode = 'ab' if os.path.exists(filename) else 'wb'
    with open(filename, mode) as file:
        if mode == 'wb':
            file.write(struct.pack('>II', 2049, 0))  # Header
        for label in labels:
            file.write(struct.pack('>B', label))
            del label  # Free memory


def update_idx3_header(filename, num_items):
    with open(filename, 'r+b') as file:
        file.seek(4)
        file.write(struct.pack('>I', num_items))


def update_idx1_header(filename, num_items):
    with open(filename, 'r+b') as file:
        file.seek(4)
        file.write(struct.pack('>I', num_items))


def process_single_file(pcap_file, idx3_path, idx1_path):
    try:
        sessions, labels, stats = extract_sessions_and_label(pcap_file, length=SESSION_LENGTH)

        # Process and save each session to .idx3 file immediately
        for matrix in convert_sessions_to_matrices(sessions):
            save_to_idx3([matrix], idx3_path)

        # Save labels in .idx1 file immediately
        save_to_idx1(labels, idx1_path)

        # Explicitly free memory
        del sessions, labels
        gc.collect()

    except Exception as e:
        print(f"⚠️ Error processing {pcap_file}: {e}")

        # Ensure stats is always assigned
        stats = {
            'Sessions Total': 0,
            'Truncated': 0,
            'Padded': 0,
            'Used Full Length': 0,
            'Average Original Length': 0
        }

    return stats  # ✅ Now stats is always returned


def main():
    directories = PCAP_DIRECTORIES

    pcap_files = [
        os.path.join(root, filename)
        for dir_path in directories
        for root, _, files in os.walk(dir_path)  # 🔥 Recursively search subdirectories
        for filename in files
        if filename.endswith('.pcap')
    ]

    idx3_path = OUTPUT_IDX3
    idx1_path = OUTPUT_IDX1

    # Remove old files before starting fresh
    if os.path.exists(idx3_path):
        os.remove(idx3_path)
    if os.path.exists(idx1_path):
        os.remove(idx1_path)

    total_sessions = total_truncated = total_padded = total_full_length = total_original_length = 0

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_single_file, pcap_file, idx3_path, idx1_path): pcap_file for pcap_file in pcap_files}

        for future in as_completed(futures):
            stats = future.result()

            # Accumulate stats
            total_sessions += stats['Sessions Total']
            total_truncated += stats['Truncated']
            total_padded += stats['Padded']
            total_full_length += stats['Used Full Length']
            total_original_length += stats['Average Original Length'] * stats['Sessions Total']

            # Free memory
            del stats
            gc.collect()

    final_average_length = total_original_length / total_sessions if total_sessions else 0

    print(f"Final Stats: Sessions Total: {total_sessions}, Truncated: {total_truncated}, "
          f"Padded: {total_padded}, Used Full Length: {total_full_length}, "
          f"Average Original Length: {final_average_length:.2f}")

    update_idx3_header(idx3_path, total_sessions)
    update_idx1_header(idx1_path, total_sessions)


if __name__ == "__main__":
    main()
