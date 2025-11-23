import pandas as pd
import os

# Read the CSV file
input_file = 'dataset_translated_6000.csv'
df = pd.read_csv(input_file)

# Get total number of rows
total_rows = len(df)
print(f"Total rows in dataset: {total_rows}")

# Define chunk size (1000 rows per file)
chunk_size = 1000

# Split into 6 files
for i in range(6):
    start_idx = i * chunk_size
    end_idx = (i + 1) * chunk_size
    
    # Extract the chunk (inclusive of both start and end indices)
    chunk = df.iloc[start_idx:end_idx]
    
    # Create output filename
    output_file = f'dataset_translated_{start_idx + 1}_{end_idx}.csv'
    
    # Save to CSV
    chunk.to_csv(output_file, index=False)
    print(f"Created {output_file} with {len(chunk)} rows (rows {start_idx + 1} to {end_idx})")

print("\nSplitting completed!")

