def get_packet_body_lengths(file_path):
    """
    Calculate the length (number of bytes) of each packet body in the input data file.
    
    Args:
        file_path (str): Path to the input data file
    
    Returns:
        list: A list of tuples containing (packet_index, body_length)
    """
    lengths = []
    with open(file_path, 'r') as file:
        for i, line in enumerate(file, 1):
            # Skip empty lines
            if not line.strip():
                continue
                
            # Split the line into header and body parts
            try:
                # Find the position of " Body: b'" 
                body_start = line.find(" Body: b'")
                
                if body_start == -1:
                    print(f"Warning: Line {i} does not match expected format")
                    continue
                    
                # Extract the body part (which is in Python bytes literal format)
                body_literal = line[body_start + 8:].strip()
                  # Remove the trailing single quote if it exists
                if body_literal.endswith("'"):
                    body_literal = body_literal[:-1]
                
                # Manually count the bytes in the body_literal instead of using eval
                # This approach avoids issues with escape sequences and line continuation characters
                
                # Split the string into individual hex characters (represented as \xXX)
                # Each byte in a Python bytes literal is represented as \xXX where XX is a hex value
                body_length = 0
                j = 0
                while j < len(body_literal):
                    if body_literal[j:j+2] == '\\x' and j+4 <= len(body_literal):
                        # This is a hex escape sequence like \xAB
                        body_length += 1
                        j += 4  # Skip the entire \xAB sequence
                    elif body_literal[j:j+2] == '\\\\' or body_literal[j:j+2] == "\\'":
                        # This is an escaped backslash or single quote
                        body_length += 1
                        j += 2
                    elif body_literal[j:j+1] == '\\' and j+1 < len(body_literal):
                        # This is some other escape sequence like \n, \t, \r
                        body_length += 1
                        j += 2
                    else:
                        # This is a regular character
                        body_length += 1
                        j += 1
                
                lengths.append((i, body_length))
            except Exception as e:
                print(f"Error processing line {i}: {e}")
    
    return lengths

def print_packet_lengths(file_path):
    """
    Print the length of each packet body in the input data file.
    
    Args:
        file_path (str): Path to the input data file
    """
    lengths = get_packet_body_lengths(file_path)
    
    print(f"{'Packet #':<10} {'Length (bytes)':<15}")
    print("-" * 25)
    
    for packet_num, length in lengths:
        print(f"{packet_num:<10} {length:<15}")
    
    if lengths:
        avg_length = sum(length for _, length in lengths) / len(lengths)
        print(f"\nTotal packets: {len(lengths)}")
        print(f"Average length: {avg_length:.2f} bytes")

# Example usage
if __name__ == "__main__":
    # Change this to the appropriate path if needed
    input_file = "input_data.txt"
    print_packet_lengths(input_file)