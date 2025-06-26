#!/usr/bin/env python3

"""
Script to extract body content from packet data files (QMDL format or standard format)
"""

import os
import re

def extract_packet_bodies(input_file, output_file, log_id=None):
    """
    Extract packet body content from the input file and write them to the output file.
    
    Args:
        input_file (str): Path to the input file containing packet data
        output_file (str): Path to the output file to write packet bodies
        log_id (int, optional): Specific log ID to extract. If None, extracts all packets.
    """
    # Pattern to match different header formats and extract the body
    # This handles both QMDL format and standard format
    qmdl_pattern = r'Header: QcDiagLogHeader.*?log_id=(\d+).*?,Body: (b\'.*?\')'
    standard_pattern = r'Header: (0x[0-9a-fA-F]+) , Body: (b\'.*?\')'
    
    try:
        # Read the entire input file
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find all matching packets
        qmdl_matches = re.findall(qmdl_pattern, content)
        standard_matches = re.findall(standard_pattern, content)
        
        extracted_packets = []
        
        # Process QMDL format matches
        for log_id_str, body in qmdl_matches:
            if log_id is None or (log_id is not None and int(log_id_str) == log_id):
                extracted_packets.append(body)
        
        # Process standard format matches
        for header, body in standard_matches:
            if log_id is None or (log_id is not None and header.lower() == f"0x{log_id:x}"):
                extracted_packets.append(body)
        
        if not extracted_packets:
            if log_id:
                print(f"No packets with log_id={log_id} found in {input_file}")
            else:
                print(f"No packets found in {input_file}")
            return False
        
        # Write the extracted packets to the output file
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, packet in enumerate(extracted_packets, 1):
                f.write(f"Body: {packet}\n")
        
        if log_id:
            print(f"Successfully extracted {len(extracted_packets)} packets with log_id={log_id} to {output_file}")
        else:
            print(f"Successfully extracted {len(extracted_packets)} packets to {output_file}")
        return True
    
    except Exception as e:
        print(f"Error processing files: {str(e)}")
        return False

def main():
    # Define file paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Get user input for file paths and filtering
    input_default = os.path.join(current_dir, "..", "data", "25 July 525p", "data_with_qmdl_b881_v2.txt")
    output_default = os.path.join(current_dir, "..", "data", "extracted_bodies.txt")
    
    print(f"Default input file: {input_default}")
    input_file = input(f"Enter input file path (or press Enter for default): ").strip() or input_default
    
    print(f"Default output file: {output_default}")
    output_file = input(f"Enter output file path (or press Enter for default): ").strip() or output_default
    
    # Ask if user wants to filter by log ID
    filter_choice = input("Do you want to filter by log ID? (y/n): ").strip().lower()
    log_id = None
    if filter_choice == 'y':
        try:
            log_id = int(input("Enter log ID to filter (e.g., 47233 for QMDL b881): ").strip())
        except ValueError:
            print("Invalid log ID, extracting all packets.")
    
    # Extract packets
    extract_packet_bodies(input_file, output_file, log_id)

if __name__ == "__main__":
    main()
    
    # Examples of usage:
    # To extract all packets:
    # extract_packet_bodies("input.txt", "output.txt")
    
    # To extract only packets with log_id 47233 (b881 in QMDL format):
    # extract_packet_bodies("input.txt", "output.txt", 47233)
    
    # To extract only packets with header 0xb881 (standard format):
    # extract_packet_bodies("input.txt", "output.txt", 0xb881)