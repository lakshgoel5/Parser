#!/usr/bin/env python3

"""
Script to extract 0xb881 packets from required.txt and save them to b881_packets.txt
"""

import os
import re

def extract_b881_packets(input_file, output_file):
    """
    Extract all 0xb881 packets from the input file and write them to the output file.
    
    Args:
        input_file (str): Path to the input file containing packet data
        output_file (str): Path to the output file to write b881 packets
    """
    # Pattern to match 0xb881 packets
    pattern = r'Header: 0xb881 , Body: (b\'.*?\')'
    
    try:
        # Read the entire input file
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find all b881 packets
        b881_packets = re.findall(pattern, content)
        
        if not b881_packets:
            print(f"No 0xb881 packets found in {input_file}")
            return False
        
        # Write the extracted packets to the output file
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, packet in enumerate(b881_packets, 1):
                f.write(f"Body: {packet}\n")
        
        print(f"Successfully extracted {len(b881_packets)} 0xb881 packets to {output_file}")
        return True
    
    except Exception as e:
        print(f"Error processing files: {str(e)}")
        return False

def main():
    # Define file paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(current_dir, "required.txt")
    output_file = os.path.join(current_dir, "b881_packets.txt")
    
    # Extract packets
    extract_b881_packets(input_file, output_file)

if __name__ == "__main__":
    main()