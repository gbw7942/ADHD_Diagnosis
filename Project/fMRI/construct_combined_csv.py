import os
import pandas as pd

def combine_csv_files(input_dir, output_file):
    """
    Combine all CSV files in the specified directory into one CSV file.
    
    Args:
        input_dir (str): Directory containing CSV files.
        output_file (str): Path to save the combined CSV file.
    """
    # List all CSV files in the directory
    csv_files = [file for file in os.listdir(input_dir) if file.endswith('.csv')]
    
    if not csv_files:
        print("No CSV files found in the directory.")
        return
    
    combined_df = pd.DataFrame()
    
    for file in csv_files:
        file_path = os.path.join(input_dir, file)
        df = pd.read_csv(file_path)
        combined_df = pd.concat([combined_df, df], ignore_index=True)
    
    combined_df.to_csv(output_file, index=False)
    print(f"Combined {len(csv_files)} CSV files into {output_file}")

# Example usage
combine_csv_files("../Data","./data/megafile.csv")
