import pandas as pd
import os
import re
from glob import glob

# Find all split files matching the pattern
pattern = 'dataset_translated_*_*.csv'
split_files = glob(pattern)

if not split_files:
    print("No split files found matching the pattern 'dataset_translated_*_*.csv'")
    exit(1)

print(f"Found {len(split_files)} split files")

# Extract start index from filename and sort
def extract_start_index(filename):
    # Extract the first number from filename (e.g., "dataset_translated_1_1000.csv" -> 1)
    match = re.search(r'dataset_translated_(\d+)_\d+\.csv', filename)
    if match:
        return int(match.group(1))
    return 0

# Sort files by start index to ensure correct order
split_files.sort(key=extract_start_index)

print("\nFiles to merge (in order):")
for file in split_files:
    print(f"  - {file}")

# Read and concatenate all split files
dataframes = []
total_rows = 0

for file in split_files:
    df = pd.read_csv(file)
    dataframes.append(df)
    total_rows += len(df)
    print(f"Loaded {file}: {len(df)} rows")

# Merge all dataframes
merged_df = pd.concat(dataframes, ignore_index=True)

print(f"\nTotal rows in merged dataset: {len(merged_df)}")
print(f"Expected total rows: {total_rows}")

# Save merged file
output_file = 'dataset_ABSA2_merged.csv'
merged_df.to_csv(output_file, index=False)
print(f"\nMerged file saved as: {output_file}")

print("\nMerging completed!")

