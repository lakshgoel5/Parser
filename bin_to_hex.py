def view_hex(filename, max_bytes=300):
    with open(filename, 'rb') as f:
        data = f.read(max_bytes)
    
    # Print hex values
    hex_str = ' '.join(f'{b:02x}' for b in data)
    print(f"First {len(data)} bytes of {filename}:")
    print(hex_str)
    
    # Print as integers
    print("As integers:", list(data))

# View your file
view_hex('split_bytes/byte_9.bin')