#!/usr/bin/env python3

"""
Script to split bytes from b881 packets into separate files.
Each file will contain integer values for a specific byte position.
Also includes frequency counts of each value at the end of the files.
"""

import re
import os
from collections import Counter

def parse_and_split_binary_data(input_file, output_dir="split_bytesb881", output_format="integer"):
    """
    Parse binary data from input file and split bytes of each packet
    into separate files (byte_0.txt, byte_1.txt, etc.)
    Also add frequency counts at the end of each file
    
    Args:
        input_file (str): Path to the input file containing b881 packet data
        output_dir (str): Directory to save the output files
        output_format (str): "integer" for .txt files with integer values,
                            "binary" for .bin files
    """
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize lists to store bytes for each position (we'll dynamically expand this)
    byte_columns = []
    
    # Pattern to match lines with Body content
    pattern = r"Body:\s*(b\'.*?\')"
    
    line_count = 0
    max_bytes = 0  # Track the maximum number of bytes in any packet
    
    try:
        with open(input_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Match the pattern
                match = re.match(pattern, line)
                if match:
                    # Extract the body string
                    body_str = match.group(1)
                    
                    # Convert escape sequences to actual bytes
                    try:
                        # Use eval to properly decode the byte string
                        body_bytes = eval(body_str)
                        
                        # Update max_bytes if this packet has more bytes
                        max_bytes = max(max_bytes, len(body_bytes))
                        
                        # Make sure byte_columns is large enough
                        while len(byte_columns) < len(body_bytes):
                            byte_columns.append([])
                        
                        # Store each byte in its corresponding column
                        for i in range(len(body_bytes)):
                            byte_columns[i].append(body_bytes[i])
                        
                        line_count += 1
                        
                        if line_count % 100 == 0:
                            print(f"Processed {line_count} packets...")
                            
                    except Exception as e:
                        print(f"Error processing line: {line}")
                        print(f"Error: {e}")
                        continue
    
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found!")
        return
    
    print(f"Found packets with up to {max_bytes} bytes")
    
    # Write each byte column to separate files with frequency counts
    for i in range(len(byte_columns)):
        if output_format == "integer":
            output_file = os.path.join(output_dir, f"byte_{i}.txt")
            try:
                # Calculate frequency counts
                value_counts = Counter(byte_columns[i])
                
                with open(output_file, 'w') as f:
                    # Write integers, one per line
                    for byte_val in byte_columns[i]:
                        f.write(f"{byte_val}\n")
                    
                    # Add separator before frequency counts
                    f.write("\n" + "-" * 50 + "\n")
                    f.write("FREQUENCY COUNTS:\n")
                    f.write("-" * 50 + "\n")
                    
                    # Add frequency counts sorted by value
                    for value, count in sorted(value_counts.items()):
                        f.write(f"Value {value}: {count} occurrences ({(count/len(byte_columns[i])*100):.2f}%)\n")
                    
                    # Add most common values
                    f.write("\n" + "-" * 50 + "\n")
                    f.write("MOST COMMON VALUES:\n")
                    f.write("-" * 50 + "\n")
                    for value, count in value_counts.most_common(10):
                        f.write(f"Value {value}: {count} occurrences ({(count/len(byte_columns[i])*100):.2f}%)\n")
                
                print(f"Created {output_file} with {len(byte_columns[i])} integer values and frequency counts")
            except Exception as e:
                print(f"Error writing {output_file}: {e}")
        else:  # binary format
            # For binary format, we'll create two files - one binary and one with frequency counts
            output_file_bin = os.path.join(output_dir, f"byte_{i}.bin")
            output_file_count = os.path.join(output_dir, f"byte_{i}_counts.txt")
            
            try:
                # Write binary data
                with open(output_file_bin, 'wb') as f:
                    f.write(bytes(byte_columns[i]))
                
                # Calculate and write frequency counts to a separate file
                value_counts = Counter(byte_columns[i])
                with open(output_file_count, 'w') as f:
                    f.write(f"FREQUENCY COUNTS FOR BYTE POSITION {i}:\n")
                    f.write("-" * 50 + "\n")
                    
                    # Add frequency counts sorted by value
                    for value, count in sorted(value_counts.items()):
                        f.write(f"Value {value}: {count} occurrences ({(count/len(byte_columns[i])*100):.2f}%)\n")
                    
                    # Add most common values
                    f.write("\n" + "-" * 50 + "\n")
                    f.write("MOST COMMON VALUES:\n")
                    f.write("-" * 50 + "\n")
                    for value, count in value_counts.most_common(10):
                        f.write(f"Value {value}: {count} occurrences ({(count/len(byte_columns[i])*100):.2f}%)\n")
                
                print(f"Created {output_file_bin} with {len(byte_columns[i])} bytes")
                print(f"Created {output_file_count} with frequency counts")
            except Exception as e:
                print(f"Error writing files for byte position {i}: {e}")
    
    print(f"\nProcessing complete!")
    print(f"Total packets processed: {line_count}")
    print(f"Output files created in '{output_dir}' directory")

def main():
    """Main function to run the script"""
    # Define file paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(current_dir, "b881_packets.txt")
    output_dir = os.path.join(current_dir, "split_bytesb881")
    
    # Use integer format for output files
    output_format = "integer"  # Change to "binary" for .bin files
    
    print("B881 Packet Byte Splitter")
    print("=" * 50)
    print(f"Input file: {input_file}")
    print(f"Output directory: {output_dir}")
    print(f"Output format: {output_format}")
    print()
    
    # Process the file
    parse_and_split_binary_data(input_file, output_dir, output_format)

if __name__ == "__main__":
    main()