#!/usr/bin/env python3
import re
import os

def parse_and_split_binary_data(input_file, output_dir="output_bytes", output_format="binary"):
    """
    Parse binary data from input file and split first 20 bytes of each body
    into separate files (byte_0.bin, byte_1.bin, etc.)
    
    Args:
        output_format: "binary" for .bin files, "integer" for .txt files with integer values
    """
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize lists to store bytes for each position (0-19)
    byte_columns = [[] for _ in range(20)]
    
    # Pattern to match lines with Header and Body
    pattern = r"Header:\s*0x[0-9a-fA-F]+\s*,\s*Body:\s*b'(.*)'"
    
    line_count = 0
    
    try:
        with open(input_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Match the pattern
                match = re.match(pattern, line)
                if match:
                    # Extract the body string (everything between b' and ')
                    body_str = match.group(1)
                    
                    # Convert escape sequences to actual bytes
                    try:
                        # Use eval to properly decode the byte string
                        body_bytes = eval(f"b'{body_str}'")
                        
                        # Take first 20 bytes (or less if body is shorter)
                        bytes_to_process = min(20, len(body_bytes))
                        
                        # Store each byte in its corresponding column
                        for i in range(bytes_to_process):
                            byte_columns[i].append(body_bytes[i])
                        
                        # If body has less than 20 bytes, pad with zeros or skip
                        # Here we'll pad with zeros for consistency
                        for i in range(bytes_to_process, 20):
                            byte_columns[i].append(0)
                        
                        line_count += 1
                        
                        if line_count % 1000 == 0:
                            print(f"Processed {line_count} lines...")
                            
                    except Exception as e:
                        print(f"Error processing line: {line}")
                        print(f"Error: {e}")
                        continue
    
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found!")
        return
    
    # Write each byte column to separate files
    for i in range(20):
        if output_format == "integer":
            output_file = os.path.join(output_dir, f"byte_{i}.txt")
            try:
                with open(output_file, 'w') as f:
                    # Write integers, one per line
                    for byte_val in byte_columns[i]:
                        f.write(f"{byte_val}\n")
                print(f"Created {output_file} with {len(byte_columns[i])} integer values")
            except Exception as e:
                print(f"Error writing {output_file}: {e}")
        else:  # binary format
            output_file = os.path.join(output_dir, f"byte_{i}.bin")
            try:
                with open(output_file, 'wb') as f:
                    # Convert list of integers to bytes
                    f.write(bytes(byte_columns[i]))
                print(f"Created {output_file} with {len(byte_columns[i])} bytes")
            except Exception as e:
                print(f"Error writing {output_file}: {e}")
    
    print(f"\nProcessing complete!")
    print(f"Total lines processed: {line_count}")
    print(f"Output files created in '{output_dir}' directory")

def main():
    # Configuration
    input_filename = "b881_packets.txt"  # Change this to your input file name
    output_directory = "split_bytesb881"   # Change this to your desired output directory
    
    # Choose output format: "binary" or "integer"
    output_format = "integer"  # Change to "binary" for .bin files
    
    print("Binary Data Splitter")
    print("=" * 50)
    print(f"Input file: {input_filename}")
    print(f"Output directory: {output_directory}")
    print(f"Output format: {output_format}")
    print()
    
    # Process the file
    parse_and_split_binary_data(input_filename, output_directory, output_format)

if __name__ == "__main__":
    main()