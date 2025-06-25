#!/usr/bin/env python3
"""
Data Extractor for Entropy Analysis
Extracts byte sequences at different offsets and granularities from binary data.
"""

import struct
from typing import List, Dict, Tuple
import numpy as np

class DataExtractor:
    def __init__(self, data_list: List[bytes]):
        """
        Initialize with list of binary data packets.
        
        Args:
            data_list: List of bytes objects containing the binary data
        """
        self.data_list = data_list
        self.sequences = {}
        
    def extract_sequences(self, byte_ranges: List[int] = [1, 2, 4, 8, 16, 32]) -> Dict:
        """
        Extract sequences for different byte ranges and offsets.
        
        Args:
            byte_ranges: List of byte ranges to extract (1, 2, 4, 8, 16, 32 bytes)
            
        Returns:
            Dictionary containing extracted sequences organized by byte_range and offset
        """
        results = {}
        
        for byte_range in byte_ranges:
            results[byte_range] = {}
            
            # Determine maximum possible offset for this byte range
            min_data_length = min(len(data) for data in self.data_list)
            max_offset = max(0, min_data_length - byte_range)
            
            if max_offset < 0:
                print(f"Warning: Data too short for {byte_range}-byte extraction")
                continue
                
            # Extract sequences for each possible offset
            for offset in range(max_offset + 1):
                sequence = []
                
                for packet_idx, data in enumerate(self.data_list):
                    if len(data) >= offset + byte_range:
                        # Extract bytes at this offset
                        byte_chunk = data[offset:offset + byte_range]
                        
                        # Convert to appropriate integer value based on byte range
                        if byte_range == 1:
                            value = struct.unpack('B', byte_chunk)[0]  # 8-bit unsigned
                        elif byte_range == 2:
                            value = struct.unpack('<H', byte_chunk)[0]  # 16-bit little-endian
                        elif byte_range == 4:
                            value = struct.unpack('<I', byte_chunk)[0]  # 32-bit little-endian
                        elif byte_range == 8:
                            value = struct.unpack('<Q', byte_chunk)[0]  # 64-bit little-endian
                        elif byte_range == 16:
                            # For 16 bytes, we'll use the first 8 bytes as a 64-bit value
                            # and combine with a hash of the remaining 8 bytes
                            first_8 = struct.unpack('<Q', byte_chunk[:8])[0]
                            second_8 = struct.unpack('<Q', byte_chunk[8:16])[0]
                            value = first_8 ^ second_8  # XOR combination
                        
                        sequence.append(value)
                
                results[byte_range][offset] = sequence
                
        self.sequences = results
        return results
    
    def get_sequence_stats(self) -> Dict:
        """
        Calculate statistics for extracted sequences.
        
        Returns:
            Dictionary containing statistics for each sequence
        """
        stats = {}
        
        for byte_range, offsets in self.sequences.items():
            stats[byte_range] = {}
            
            for offset, sequence in offsets.items():
                if sequence:
                    seq_array = np.array(sequence)
                    stats[byte_range][offset] = {
                        'length': len(sequence),
                        'min': int(np.min(seq_array)),
                        'max': int(np.max(seq_array)),
                        'mean': float(np.mean(seq_array)),
                        'std': float(np.std(seq_array)),
                        'unique_values': len(np.unique(seq_array)),
                        'entropy': self._calculate_entropy(sequence)
                    }
                    
        return stats
    
    def _calculate_entropy(self, sequence: List[int]) -> float:
        """
        Calculate Shannon entropy of a sequence.
        
        Args:
            sequence: List of integer values
            
        Returns:
            Shannon entropy value
        """
        if not sequence:
            return 0.0
            
        # Calculate value frequencies
        unique, counts = np.unique(sequence, return_counts=True)
        probabilities = counts / len(sequence)
        
        # Calculate Shannon entropy
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        return float(entropy)
    
    def save_sequences(self, filename: str):
        """
        Save extracted sequences to a file for later analysis.
        
        Args:
            filename: Output filename
        """
        import pickle
        
        with open(filename, 'wb') as f:
            pickle.dump({
                'sequences': self.sequences,
                'stats': self.get_sequence_stats()
            }, f)
        
        print(f"Sequences saved to {filename}")

def parse_data_from_string(data_strings: List[str]) -> List[bytes]:
    """
    Parse binary data from string representations.
    
    Args:
        data_strings: List of string representations of binary data
        
    Returns:
        List of bytes objects
    """
    parsed_data = []
    
    for data_str in data_strings:
        # Remove the 'b'' wrapper and parse as bytes
        if data_str.startswith("b'") and data_str.endswith("'"):
            # Use eval to properly parse the byte string
            try:
                byte_data = eval(data_str)
                parsed_data.append(byte_data)
            except Exception as e:
                print(f"Error parsing data string: {e}")
                continue
        else:
            print(f"Invalid data format: {data_str[:50]}...")
            
    return parsed_data

def read_data_from_file(filename: str) -> List[bytes]:
    """
    Read binary data from a file where each line contains a bytes representation.
    
    Args:
        filename: Path to the file containing the data
        
    Returns:
        List of bytes objects
    """
    parsed_data = []
    
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            
        print(f"Reading {len(lines)} lines from {filename}")
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
                
            # Handle different possible formats
            try:
                # Try to parse as a byte string representation
                if line.startswith("b'") or line.startswith('b"'):
                    byte_data = eval(line)
                    parsed_data.append(byte_data)
                elif line.startswith("Body: b'"):
                    # Handle format like "Body: b'...'"
                    body_part = line[6:]  # Remove "Body: " prefix
                    byte_data = eval(body_part)
                    parsed_data.append(byte_data)
                elif line.startswith("'") or line.startswith('"'):
                    # Handle quoted strings
                    byte_data = eval(f"b{line}")
                    parsed_data.append(byte_data)
                else:
                    # Try to parse as hex string
                    if all(c in '0123456789abcdefABCDEF\\x ' for c in line):
                        # Convert hex representation to bytes
                        hex_clean = line.replace('\\x', '').replace(' ', '')
                        if len(hex_clean) % 2 == 0:
                            byte_data = bytes.fromhex(hex_clean)
                            parsed_data.append(byte_data)
                        else:
                            print(f"Line {line_num}: Invalid hex format - {line[:50]}...")
                    else:
                        print(f"Line {line_num}: Unknown format - {line[:50]}...")
                        
            except Exception as e:
                print(f"Line {line_num}: Error parsing '{line[:50]}...' - {e}")
                continue
                
        print(f"Successfully parsed {len(parsed_data)} data entries")
        return parsed_data
        
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found")
        return []
    except Exception as e:
        print(f"Error reading file '{filename}': {e}")
        return []

# Example usage
if __name__ == "__main__":
    # Option 1: Read from file
    print("Reading data from file...")
    file_data = read_data_from_file("C:\\Users\\laksh\\OneDrive - IIT Delhi\\Desktop\\SURA\\Parser\\data\\b881_packets.txt")  # Change to your filename

    if file_data:
        # Create extractor and extract sequences
        extractor = DataExtractor(file_data)
        sequences = extractor.extract_sequences()
        
        # Print statistics
        stats = extractor.get_sequence_stats()
        
        print(f"\nExtraction Results for {len(file_data)} packets:")
        for byte_range, offsets in stats.items():
            print(f"\n{byte_range}-byte sequences:")
            for offset, stat in offsets.items():
                print(f"  Offset {offset}: {stat['length']} values, "
                      f"entropy={stat['entropy']:.3f}, "
                      f"unique={stat['unique_values']}")
        
        # Save results
        extractor.save_sequences("extracted_sequences.pkl")
    else:
        print("No data found in file. Using sample data...")
        
        # Option 2: Use sample data (fallback)
        sample_data = [
            b'\x01\x00\x03\x00\x00\x00\x00\x01\x00\x00\x00\x003\x93F\x00\x00\x00\x00\x00\xefy\x03\x00\x00\x00\x00\x00\xdb\xa2\x01\x00\x00\x00\x00\x00<\x90\x05\x00\x00\x00\x00\x00\xe8Z\x02\x00\x00\x00\x00\x00\x18\x00\x00\x00q9\x00\x00\xbc\x02\x00\x00\xd2=\x00\x000d\x02\x00\xf1\x16\x00\x00\x03\x00\x00\x00\xc1;\x00\x00\x00\x00\x00\x00\xfa\x00\x00\x00'
        ] * 7  # Simulate multiple packets
        
        # Create extractor and extract sequences
        extractor = DataExtractor(sample_data)
        sequences = extractor.extract_sequences()
        
        # Print statistics
        stats = extractor.get_sequence_stats()
        
        print("Extraction Results:")
        for byte_range, offsets in stats.items():
            print(f"\n{byte_range}-byte sequences:")
            for offset, stat in offsets.items():
                print(f"  Offset {offset}: {stat['length']} values, "
                      f"entropy={stat['entropy']:.3f}, "
                      f"unique={stat['unique_values']}")
        
        # Save results
        extractor.save_sequences("extracted_sequences.pkl")