from collections import Counter

def count_frequency():
    # Open the file and read lines
    with open('length.txt', 'r') as file:
        # Read all lines and convert each to integer
        numbers = [int(line.strip()) for line in file if line.strip()]
    
    # Count frequency of each number
    frequency = Counter(numbers)
    
    # Sort the frequencies by number value
    sorted_freq = sorted(frequency.items())
    
    # Print results
    print("Number\tFrequency")
    print("-----------------")
    for number, count in sorted_freq:
        print(f"{number}\t{count}")
    
    # Print total count of numbers
    print(f"\nTotal numbers: {len(numbers)}")
    
if __name__ == "__main__":
    count_frequency()
