def extract_data_after_column(input_file, output_file, column_index=120):
    """
    Extract data from each line starting at specified column index and save to output file
    
    Args:
        input_file (str): Path to the input file
        output_file (str): Path to the output file
        column_index (int): Column index to start extraction from (0-based)
    """
    try:
        with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
            for line in infile:
                # Check if the line is long enough
                if len(line) > column_index:
                    # Extract data starting from column_index
                    extracted_data = line[column_index:]
                    outfile.write(extracted_data)
                else:
                    # If line is shorter than column_index, write an empty line
                    outfile.write('\n')
        
        print(f"Data extraction complete. Output saved to {output_file}")
    
    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
if __name__ == "__main__":
    input_file = input("Enter the path to the input file: ")
    output_file = input("Enter the path to the output file: ")
    
    # Note: In Python, string indices are 0-based, so column 120 is actually index 119
    # But to match the requirement of "after column 120", we use 120
    extract_data_after_column(input_file, output_file, 119)