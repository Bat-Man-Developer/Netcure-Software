from datetime import datetime
import pyshark
import subprocess
import socket
import os
import csv

class NetworkData:
    def __init__(self):
        self.wifi_interface_name = self.get_wifi_interface_name()
        self.personal_device_ip_address = self.get_ip_address()

    def get_wifi_interface_name(self):
        try:
            result = subprocess.run(["netsh", "interface", "show", "interface"], capture_output=True, text=True)
            if result.returncode == 0:
                output_lines = result.stdout.splitlines()
                wifi_interface = None
                for line in output_lines:
                    if "WiFi" in line or "Wi-Fi" in line:
                        wifi_interface = line.split()[-1]
                        break
                if wifi_interface:
                    return wifi_interface
                else:
                    print("WiFi interface not found. ==> ")
                    return 0
            else:
                print("Error: Unable to view WiFi network interfaces. ==> ")
                return 0
        except Exception as e:
            print(f"An unexpected error occurred: {str(e)} ==> ")
            return "none"

    def get_ip_address(self):
        try:
            hostname = socket.gethostname()
            ip_address = socket.gethostbyname(hostname)
            return ip_address
        except Exception as e:
            print(f"An error occurred while trying to retrieve the Personal Device IP address: {str(e)}  ==> ")
            return "none"

    def save_pcap_file(self, pcap_file):
        timeout = 5
        try:
            capture_dir = "c:/Xampp/htdocs/net-cure-website/network_packets"
            if not os.path.exists(capture_dir):
                os.makedirs(capture_dir)

            capture = pyshark.LiveCapture(interface=self.wifi_interface_name,
                                           output_file=os.path.join(capture_dir, "network_packets.pcap"))
            capture.sniff(timeout=timeout)
        except Exception as e:
            print("An unexpected error occurred:", e)

    def save_csv_file(self, pcap_file, csv_file):
        try:
            cap = pyshark.FileCapture(pcap_file)
            with open(csv_file, 'a') as f:
                for packet in cap:
                    if 'ip' in packet.layers[1].layer_name:
                        src = packet.ip.src
                        dst = packet.ip.dst
                        length = packet.length
                        timestamp = datetime.now()
                        proto = self.get_protocol_name(packet.ip.proto)

                        if "udp" in packet.layers[2].layer_name:
                            srcport = packet.udp.srcport
                            dstport = packet.udp.dstport
                        else:
                            continue

                        f.write(f'{src},{dst},{proto},{srcport},{dstport},{length},{timestamp}\n')
        finally:
            cap.close()
            print("Uploaded Network Data Successfully.|")

    def get_protocol_name(self, proto_num):
        protocols = {
            "1": "ICMP", "2": "IGMP", "4": "IPIP", "6": "TCP", "8": "EGP",
            "9": "IGRP", "17": "UDP", "33": "DCCP", "41": "IPv6", "46": "RSVP",
            "47": "GRE", "50": "ESP", "51": "AH", "83": "VINES", "88": "EIGRP",
            "89": "OSPF", "103": "PIM", "112": "VRRP", "115": "L2TP", "132": "SCTP"
        }
        return protocols.get(str(proto_num), proto_num)

    def getTime(self):
        current_time = datetime.now().strftime("%H:%M:%S")
        return current_time

    def read_csv_file(self, file_path, column_index):
        data = []
        with open(file_path, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            header = next(csv_reader)  # Skip the header row
            column_name = header[column_index]
            for row in csv_reader:
                if row[column_index] != "":
                    data.append(row[column_index])
        return data

    def get_source_addresses(self, file_path):
        return self.read_csv_file(file_path, 0)

    def get_destination_addresses(self, file_path):
        return self.read_csv_file(file_path, 1)

    def get_protocols(self, file_path):
        return self.read_csv_file(file_path, 2)

    def get_source_ports(self, file_path):
        return self.read_csv_file(file_path, 3)

    def get_destination_ports(self, file_path):
        return self.read_csv_file(file_path, 4)

    def get_lengths(self, file_path):
        return self.read_csv_file(file_path, 5)

    def get_timestamps(self, file_path):
        return self.read_csv_file(file_path, 6)

# Usage example:
network_data = NetworkData()

pcap_file = 'C:/Xampp/htdocs/net-cure-website/network_packets/network_packets.pcap'
csv_file = 'C:/Xampp/htdocs/net-cure-website/network_packets/network_packets.csv'

network_data.save_pcap_file(pcap_file)
network_data.save_csv_file(pcap_file, csv_file)

current_time = network_data.getTime()
print("Time: ", current_time, "|")

source_addresses = network_data.get_source_addresses(csv_file)
print(source_addresses, "|")

destination_addresses = network_data.get_destination_addresses(csv_file)
print(destination_addresses, "|")

protocols = network_data.get_protocols(csv_file)
print(protocols, "|")

source_ports = network_data.get_source_ports(csv_file)
print(source_ports, "|")

destination_ports = network_data.get_destination_ports(csv_file)
print(destination_ports, "|")

lengths = network_data.get_lengths(csv_file)
print(lengths, "|")

timestamps = network_data.get_timestamps(csv_file)
print(timestamps)