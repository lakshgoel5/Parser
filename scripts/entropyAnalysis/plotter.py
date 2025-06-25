#!/usr/bin/env python3
"""
Enhanced Entropy Plotter for Binary Data Analysis
Creates visualizations similar to the reference image showing different sequence patterns.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import pickle
from scipy import stats
import pandas as pd
from collections import Counter

class EnhancedEntropyPlotter:
    def __init__(self, sequences: Dict = None, stats: Dict = None):
        """
        Initialize plotter with extracted sequences.
        
        Args:
            sequences: Dictionary of extracted sequences from DataExtractor
            stats: Dictionary of sequence statistics
        """
        self.sequences = sequences or {}
        self.stats = stats or {}
        
        # Set up plotting style for better visualization
        plt.style.use('default')
        self.colors = ['#2E8B57', '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        
    def classify_sequence_pattern(self, sequence: List[int]) -> str:
        """
        Classify sequence into pattern types: Identifier, Sequential, Random, etc.
        
        Args:
            sequence: List of values to classify
            
        Returns:
            Pattern type as string
        """
        if len(sequence) < 3:
            return "Unknown"
            
        # Check for constant (identifier-like) pattern
        unique_vals = len(set(sequence))
        if unique_vals == 1:
            return "Constant"
        elif unique_vals <= 3:
            return "Identifier"
            
        # Check for sequential pattern
        diffs = np.diff(sequence)
        if len(set(diffs)) == 1 and diffs[0] != 0:  # Constant difference
            return "Sequential"
        elif np.std(diffs) < np.std(sequence) * 0.1:  # Nearly constant difference
            return "Near-Sequential"
            
        # Check for high entropy (random-like)
        entropy = self._calculate_entropy(sequence)
        if entropy > 0.8 * np.log2(min(len(sequence), 256)):  # High entropy
            return "Random"
        elif entropy < 0.3 * np.log2(min(len(sequence), 256)):  # Low entropy
            return "Low-Entropy"
        else:
            return "Mixed"
    
    def _calculate_entropy(self, sequence: List[int]) -> float:
        """Calculate Shannon entropy of a sequence."""
        if not sequence:
            return 0
        counter = Counter(sequence)
        probs = np.array(list(counter.values())) / len(sequence)
        return -np.sum(probs * np.log2(probs + 1e-10))
    
    def plot_pattern_analysis(self, byte_range: int, max_offsets: int = 6, 
                            figsize: Tuple[int, int] = (15, 10)):
        """
        Create pattern analysis plots similar to the reference image.
        
        Args:
            byte_range: Byte range to plot (1, 2, 4, 8, 16)
            max_offsets: Maximum number of offsets to plot
            figsize: Figure size tuple
        """
        if byte_range not in self.sequences:
            print(f"No data available for {byte_range}-byte sequences")
            return
            
        offsets_data = self.sequences[byte_range]
        offsets_to_plot = list(offsets_data.keys())[:max_offsets]
        
        # Create subplots
        fig, axes = plt.subplots(len(offsets_to_plot), 1, figsize=figsize, 
                                sharex=True)
        if len(offsets_to_plot) == 1:
            axes = [axes]
            
        fig.suptitle(f'Pattern Analysis for {byte_range}-byte Sequences', 
                     fontsize=16, fontweight='bold')
        
        for i, offset in enumerate(offsets_to_plot):
            sequence = offsets_data[offset]
            packet_indices = range(len(sequence))
            
            # Classify pattern
            pattern_type = self.classify_sequence_pattern(sequence)
            
            # Choose color and style based on pattern
            if pattern_type in ["Identifier", "Constant"]:
                color = '#2E8B57'  # Green for identifiers
                marker = 's'  # Square markers
                linestyle = '-'
            elif pattern_type in ["Sequential", "Near-Sequential"]:
                color = '#FF6B6B'  # Red for sequential
                marker = 'o'  # Circle markers
                linestyle = '--'
            elif pattern_type == "Random":
                color = '#4ECDC4'  # Teal for random
                marker = '.'  # Dot markers
                linestyle = ':'
            else:
                color = '#45B7D1'  # Blue for mixed/other
                marker = '^'  # Triangle markers
                linestyle = '-.'
            
            axes[i].plot(packet_indices, sequence, color=color, marker=marker,
                        linestyle=linestyle, markersize=4, linewidth=1.5, 
                        alpha=0.8, label=pattern_type)
            
            # Formatting
            axes[i].set_ylabel(f'Offset {offset}\nValue', fontsize=10)
            axes[i].grid(True, alpha=0.3)
            axes[i].legend(loc='upper right')
            
            # Add statistics box
            if byte_range in self.stats and offset in self.stats[byte_range]:
                entropy = self.stats[byte_range][offset]['entropy']
                unique_vals = self.stats[byte_range][offset]['unique_values']
                mean_val = self.stats[byte_range][offset]['mean']
                
                stats_text = f'Pattern: {pattern_type}\nEntropy: {entropy:.3f}\nUnique: {unique_vals}\nMean: {mean_val:.1f}'
                axes[i].text(0.02, 0.98, stats_text, 
                           transform=axes[i].transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                   alpha=0.8, edgecolor='gray'))
        
        axes[-1].set_xlabel('Packet #', fontsize=12)
        plt.tight_layout()
        plt.show()
    
    def plot_multi_pattern_comparison(self, byte_range: int, 
                                    figsize: Tuple[int, int] = (16, 8)):
        """
        Create a comparison plot showing multiple patterns on the same graph.
        
        Args:
            byte_range: Byte range to plot
            figsize: Figure size tuple
        """
        if byte_range not in self.sequences:
            print(f"No data available for {byte_range}-byte sequences")
            return
            
        offsets_data = self.sequences[byte_range]
        
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # Plot each offset with different styling
        for i, (offset, sequence) in enumerate(offsets_data.items()):
            if i >= 6:  # Limit to 6 offsets for clarity
                break
                
            packet_indices = range(len(sequence))
            pattern_type = self.classify_sequence_pattern(sequence)
            
            # Normalize values for better comparison (optional)
            normalized_seq = np.array(sequence)
            if np.max(sequence) > 0:
                normalized_seq = normalized_seq / np.max(sequence) * 10 + i * 12
            else:
                normalized_seq = normalized_seq + i * 12
            
            # Style based on pattern
            if pattern_type in ["Identifier", "Constant"]:
                color = self.colors[0]
                marker = 's'
                linestyle = '-'
            elif pattern_type in ["Sequential", "Near-Sequential"]:
                color = self.colors[1]
                marker = 'o'
                linestyle = '--'
            elif pattern_type == "Random":
                color = self.colors[2]
                marker = '.'
                linestyle = ':'
            else:
                color = self.colors[i % len(self.colors)]
                marker = '^'
                linestyle = '-.'
            
            ax.plot(packet_indices, normalized_seq, color=color, marker=marker,
                   linestyle=linestyle, markersize=4, linewidth=1.5, alpha=0.8,
                   label=f'Offset {offset} ({pattern_type})')
        
        ax.set_xlabel('Packet #', fontsize=12)
        ax.set_ylabel('Normalized Value', fontsize=12)
        ax.set_title(f'Multi-Pattern Comparison for {byte_range}-byte Sequences', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.show()
    
    def plot_pattern_heatmap(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Create a heatmap showing pattern types across byte ranges and offsets.
        
        Args:
            figsize: Figure size tuple
        """
        if not self.sequences:
            print("No sequences available for heatmap")
            return
            
        # Prepare data for heatmap
        pattern_data = {}
        for byte_range, offsets in self.sequences.items():
            pattern_data[f'{byte_range}-byte'] = {}
            for offset, sequence in offsets.items():
                pattern_type = self.classify_sequence_pattern(sequence)
                pattern_data[f'{byte_range}-byte'][f'Offset {offset}'] = pattern_type
        
        # Convert to DataFrame
        df = pd.DataFrame(pattern_data).T
        
        # Create mapping for pattern types to numbers
        pattern_types = set()
        for col in df.columns:
            pattern_types.update(df[col].values)
        
        # Convert all pattern types to strings to avoid sorting issues
        pattern_types_str = [str(p) for p in pattern_types]
        pattern_map = {pattern: i for i, pattern in enumerate(sorted(pattern_types_str))}
        
        # Convert patterns to numbers, ensuring we convert to strings first
        df_numeric = df.applymap(lambda x: pattern_map.get(str(x), 0))
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(df_numeric, annot=df, fmt='', cmap='Set3', 
                   cbar_kws={'label': 'Pattern Type'}, ax=ax)
        ax.set_title('Pattern Types Across Byte Ranges and Offsets', 
                    fontsize=14, fontweight='bold')
        ax.set_ylabel('Byte Range')
        ax.set_xlabel('Offset')
        
        plt.tight_layout()
        plt.show()
    
    def plot_entropy_vs_patterns(self, figsize: Tuple[int, int] = (12, 6)):
        """
        Plot entropy values grouped by pattern types.
        
        Args:
            figsize: Figure size tuple
        """
        if not self.stats or not self.sequences:
            print("No data available for entropy vs patterns plot")
            return
            
        # Collect data
        data = []
        for byte_range, offsets in self.sequences.items():
            for offset, sequence in offsets.items():
                pattern_type = self.classify_sequence_pattern(sequence)
                if byte_range in self.stats and offset in self.stats[byte_range]:
                    entropy = self.stats[byte_range][offset]['entropy']
                    data.append({
                        'byte_range': f'{byte_range}-byte',
                        'offset': offset,
                        'pattern_type': pattern_type,
                        'entropy': entropy
                    })
        
        df = pd.DataFrame(data)
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Convert pattern_type to string to avoid type comparison issues
        df['pattern_type'] = df['pattern_type'].astype(str)
        
        # Box plot of entropy by pattern type
        df.boxplot(column='entropy', by='pattern_type', ax=ax1)
        ax1.set_title('Entropy Distribution by Pattern Type')
        ax1.set_xlabel('Pattern Type')
        ax1.set_ylabel('Entropy')
        
        # Scatter plot of entropy vs byte range, colored by pattern
        for pattern in df['pattern_type'].unique():
            pattern_data = df[df['pattern_type'] == pattern]
            ax2.scatter(pattern_data['byte_range'], pattern_data['entropy'],
                       label=pattern, alpha=0.7, s=50)
        
        ax2.set_title('Entropy by Byte Range and Pattern')
        ax2.set_xlabel('Byte Range')
        ax2.set_ylabel('Entropy')
        ax2.legend()
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def plot_individual_bytes(self, byte_position_start: int, byte_position_end: int = None, 
                         figsize: Tuple[int, int] = (15, 10)):
        """
        Plot specific individual byte positions from the data.
        
        Args:
            byte_position_start: Start byte position to plot (0-based index)
            byte_position_end: End byte position to plot (inclusive)
            figsize: Figure size tuple
        """
        if 1 not in self.sequences:  # We need 1-byte sequences
            print("No 1-byte sequences available")
            return
            
        if byte_position_end is None:
            byte_position_end = byte_position_start
            
        byte_positions = range(byte_position_start, byte_position_end + 1)
        num_positions = len(byte_positions)
        
        # Create subplots
        fig, axes = plt.subplots(num_positions, 1, figsize=figsize, sharex=True)
        if num_positions == 1:
            axes = [axes]
            
        fig.suptitle(f'Analysis of Individual Bytes (Positions {byte_position_start}-{byte_position_end})', 
                    fontsize=16, fontweight='bold')
        
        for i, pos in enumerate(byte_positions):
            if pos not in self.sequences[1]:
                print(f"No data for byte position {pos}")
                continue
                
            sequence = self.sequences[1][pos]
            packet_indices = range(len(sequence))
            
            # Classify pattern
            pattern_type = self.classify_sequence_pattern(sequence)
            
            # Choose color and style based on pattern
            if pattern_type in ["Identifier", "Constant"]:
                color = '#2E8B57'  # Green for identifiers
                marker = 's'  # Square markers
            elif pattern_type in ["Sequential", "Near-Sequential"]:
                color = '#FF6B6B'  # Red for sequential
                marker = 'o'  # Circle markers
            elif pattern_type == "Random":
                color = '#4ECDC4'  # Teal for random
                marker = '.'  # Dot markers
            else:
                color = '#45B7D1'  # Blue for mixed/other
                marker = '^'  # Triangle markers
            
            axes[i].plot(packet_indices, sequence, color=color, marker=marker,
                        markersize=4, linewidth=1.5, alpha=0.8, label=pattern_type)
            
            # Formatting
            axes[i].set_ylabel(f'Byte {pos}\nValue', fontsize=10)
            axes[i].grid(True, alpha=0.3)
            axes[i].legend(loc='upper right')
            
            # Add statistics box
            if 1 in self.stats and pos in self.stats[1]:
                entropy = self.stats[1][pos]['entropy']
                unique_vals = self.stats[1][pos]['unique_values']
                mean_val = self.stats[1][pos]['mean']
                
                stats_text = f'Pattern: {pattern_type}\nEntropy: {entropy:.3f}\nUnique: {unique_vals}\nMean: {mean_val:.1f}'
                axes[i].text(0.02, 0.98, stats_text, 
                           transform=axes[i].transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                   alpha=0.8, edgecolor='gray'))
        
        axes[-1].set_xlabel('Packet #', fontsize=12)
        plt.tight_layout()
        plt.show()

    def plot_multi_byte_chunks(self, byte_range: int, start_offset: int = 0, num_chunks: int = 4,
                             figsize: Tuple[int, int] = (15, 10)):
        """
        Plot specific multi-byte chunks from the data.
        
        Args:
            byte_range: Size of each chunk in bytes (2, 4, 8, 16, 32)
            start_offset: Starting offset for the first chunk
            num_chunks: Number of consecutive chunks to plot
            figsize: Figure size tuple
        """
        if byte_range not in self.sequences:
            print(f"No {byte_range}-byte sequences available")
            return
            
        # Calculate offsets to plot
        offsets_to_plot = [start_offset + i * byte_range for i in range(num_chunks)]
        valid_offsets = [offset for offset in offsets_to_plot if offset in self.sequences[byte_range]]
        
        if not valid_offsets:
            print(f"No valid offsets found starting from {start_offset} for {byte_range}-byte chunks")
            return
            
        # Create subplots
        fig, axes = plt.subplots(len(valid_offsets), 1, figsize=figsize, sharex=True)
        if len(valid_offsets) == 1:
            axes = [axes]
            
        fig.suptitle(f'Analysis of {byte_range}-byte Chunks (Starting at offset {start_offset})', 
                    fontsize=16, fontweight='bold')
        
        for i, offset in enumerate(valid_offsets):
            sequence = self.sequences[byte_range][offset]
            packet_indices = range(len(sequence))
            
            # Classify pattern
            pattern_type = self.classify_sequence_pattern(sequence)
            
            # Choose color and style based on pattern
            if pattern_type in ["Identifier", "Constant"]:
                color = '#2E8B57'  # Green for identifiers
                marker = 's'  # Square markers
                linestyle = '-'
            elif pattern_type in ["Sequential", "Near-Sequential"]:
                color = '#FF6B6B'  # Red for sequential
                marker = 'o'  # Circle markers
                linestyle = '--'
            elif pattern_type == "Random":
                color = '#4ECDC4'  # Teal for random
                marker = '.'  # Dot markers
                linestyle = ':'
            else:
                color = '#45B7D1'  # Blue for mixed/other
                marker = '^'  # Triangle markers
                linestyle = '-.'
            
            axes[i].plot(packet_indices, sequence, color=color, marker=marker,
                        linestyle=linestyle, markersize=4, linewidth=1.5, 
                        alpha=0.8, label=pattern_type)
            
            # Create a label that shows the byte range this chunk represents
            byte_start = offset
            byte_end = offset + byte_range - 1
            chunk_label = f'Bytes {byte_start}-{byte_end}'
            
            # Formatting
            axes[i].set_ylabel(f'{chunk_label}\nValue', fontsize=10)
            axes[i].grid(True, alpha=0.3)
            axes[i].legend(loc='upper right')
            
            # Add statistics box
            if byte_range in self.stats and offset in self.stats[byte_range]:
                entropy = self.stats[byte_range][offset]['entropy']
                unique_vals = self.stats[byte_range][offset]['unique_values']
                mean_val = self.stats[byte_range][offset]['mean']
                
                stats_text = f'Pattern: {pattern_type}\nEntropy: {entropy:.3f}\nUnique: {unique_vals}\nMean: {mean_val:.1f}'
                axes[i].text(0.02, 0.98, stats_text, 
                           transform=axes[i].transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                   alpha=0.8, edgecolor='gray'))
        
        axes[-1].set_xlabel('Packet #', fontsize=12)
        plt.tight_layout()
        plt.show()
    
# Enhanced example usage
def create_sample_data_with_patterns():
    """Create sample data with different recognizable patterns."""
    packets = []
    
    for i in range(100):  # 100 packets
        packet = bytearray(50)  # 50 bytes per packet
        
        # Identifier pattern (constant values at specific offsets)
        packet[0] = 0x01  # Always 1
        packet[1] = 0xFF  # Always 255
        
        # Sequential pattern
        packet[2] = i % 256  # Incrementing counter
        packet[3] = (i * 2) % 256  # Incrementing by 2
        
        # Random pattern
        packet[4] = np.random.randint(0, 256)
        packet[5] = np.random.randint(0, 256)
        
        # Mixed pattern (some structure + noise)
        packet[6] = (i % 16) + np.random.randint(0, 4)
        packet[7] = (i // 10) % 256
        
        # Fill remaining bytes with low-entropy data
        for j in range(8, 50):
            packet[j] = np.random.choice([0x00, 0x01, 0x02], p=[0.7, 0.2, 0.1])
        
        packets.append(bytes(packet))
    
    return packets

if __name__ == "__main__":
    from data_extractor import DataExtractor, read_data_from_file
    
    # Read actual data from the specified file
    print("Reading data from file...")
    data_path = r"C:\Users\laksh\OneDrive - IIT Delhi\Desktop\SURA\Parser\data\b881_packets.txt"
    file_data = read_data_from_file(data_path)
    
    if not file_data:
        print(f"Error: Could not read data from {data_path}")
        exit(1)
        
    print(f"Successfully loaded {len(file_data)} packets from file")
    
    # Extract sequences with specified byte ranges (1, 2, 4, 8, 16, 32)
    print("Extracting sequences from data...")
    extractor = DataExtractor(file_data)
    sequences = extractor.extract_sequences(byte_ranges=[1, 2, 4, 8, 16, 32])
    
    # Generate statistics for the sequences
    print("Generating sequence statistics...")
    stats = extractor.get_sequence_stats()
    
    # Create enhanced plotter
    plotter = EnhancedEntropyPlotter(sequences, stats)
    
    # Generate plots for all requested byte ranges
    print("Generating pattern analysis plots...")
    
    # Create an interactive menu for viewing specific byte positions/chunks
    import matplotlib
    matplotlib.use('TkAgg')  # Use a non-blocking backend
    
    # Analyze each byte range first to get a summary
    byte_ranges = [1, 2, 4, 8, 16, 32]
    for byte_range in byte_ranges:
        if byte_range in sequences:
            print(f"\nAnalyzing {byte_range}-byte sequences...")
            available_offsets = sorted(list(sequences[byte_range].keys()))
            max_offset = max(available_offsets) if available_offsets else 0
            print(f"  Available offsets: 0-{max_offset}")
        else:
            print(f"No data for {byte_range}-byte sequences")
            
    print("\n=== Interactive Plotting Menu ===")
    print("1. Plot individual bytes (1-byte)")
    print("2. Plot 2-byte chunks (bytes 0-1, 2-3, etc.)")
    print("3. Plot 4-byte chunks (bytes 0-3, 4-7, etc.)")
    print("4. Plot 8-byte chunks (bytes 0-7, 8-15, etc.)")
    print("5. Plot 16-byte chunks (bytes 0-15, 16-31, etc.)")
    print("6. Plot 32-byte chunks (bytes 0-31, 32-63, etc.)")
    print("7. Plot overview analysis")
    print("8. Exit")
    
    while True:
        try:
            choice = int(input("\nEnter your choice (1-8): "))
            
            if choice == 8:
                break
                
            elif choice == 7:
                # Generate overview plots
                print("\nGenerating overview plots...")
                try:
                    plotter.plot_pattern_heatmap()
                    plotter.plot_entropy_vs_patterns()
                except Exception as e:
                    print(f"Error generating overview plots: {e}")
                    print("Try viewing individual byte sequences instead.")
                
            elif choice == 1:
                # Plot individual bytes
                start_pos = int(input("Enter start byte position: "))
                end_pos = int(input("Enter end byte position (or same as start for single): "))
                plotter.plot_individual_bytes(start_pos, end_pos)
                
            elif choice in [2, 3, 4, 5, 6]:
                # Plot multi-byte chunks
                byte_range = [2, 4, 8, 16, 32][choice-2]
                start_offset = int(input(f"Enter start offset for {byte_range}-byte chunks: "))
                num_chunks = int(input("Enter number of chunks to plot: "))
                plotter.plot_multi_byte_chunks(byte_range, start_offset, num_chunks)
                
            else:
                print("Invalid choice. Please enter a number between 1 and 8.")
                
        except ValueError:
            print("Please enter a valid number.")
        except Exception as e:
            print(f"Error: {e}")
    
    # Only generate overview plots at the end if we're not exiting from the menu
    # This code is skipped if we exited via the menu, avoiding the error
    
    # Save extracted sequences and stats for future use
    print("\nSaving extracted sequences and stats...")
    with open('extracted_sequences.pkl', 'wb') as f:
        pickle.dump({'sequences': sequences, 'stats': stats}, f)
    print("Data saved to extracted_sequences.pkl")