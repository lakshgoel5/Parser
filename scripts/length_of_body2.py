import re

def parse_and_calculate_body_lengths(filename):
    """
    Read data from file and calculate body lengths
    """
    try:
        with open(filename, 'r') as file:
            content = file.read()
        
        # Find all lines that match the pattern "Header: 0x..., Body: b'...'"
        pattern = r"Header:\s*(0x[0-9a-fA-F]+)\s*,\s*Body:\s*b'([^']+)'"
        matches = re.findall(pattern, content)
        
        print(f"Found {len(matches)} entries\n")
        print("Header\t\tBody Length")
        print("-" * 30)
        
        total_length = 0
        for i, (header, body_str) in enumerate(matches, 1):
            # Decode the byte string representation
            # Replace escape sequences with actual bytes
            body_bytes = bytes(body_str, 'utf-8').decode('unicode_escape').encode('latin1')
            body_length = len(body_bytes)
            
            print(f"{header}\t{body_length}")
            total_length += body_length
        
        print("-" * 30)
        print(f"Total bodies: {len(matches)}")
        print(f"Total length: {total_length} bytes")
        print(f"Average length: {total_length/len(matches):.1f} bytes")
        
        return matches
        
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return []
    except Exception as e:
        print(f"Error processing file: {e}")
        return []

def get_individual_body_length(body_str):
    """
    Helper function to get length of a single body string
    """
    try:
        body_bytes = bytes(body_str, 'utf-8').decode('unicode_escape').encode('latin1')
        return len(body_bytes)
    except Exception as e:
        print(f"Error processing body string: {e}")
        return 0

# Main execution
if __name__ == "__main__":
    filename = "input_data.txt"
    
    # Parse file and calculate lengths
    matches = parse_and_calculate_body_lengths(filename)
    
    # Example of how to get length of individual body if needed
    if matches:
        print(f"\nExample - First body length: {get_individual_body_length(matches[0][1])} bytes")