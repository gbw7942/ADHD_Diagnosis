import os
import pandas as pd
import re

def process_task_rest_files(input_dir, mega_file, output_file):
    """
    Search for task-rest_run files, extract ScanDir_ID, map to DX from megafile, and save to CSV.
    
    Args:
        input_dir (str): Directory containing the files.
        mega_file (str): Path to the megafile CSV.
        output_file (str): Path to save the output CSV.
    """
    # List to store image filenames and ScanDir_IDs
    Image = []
    ScanDir_ID = []
    
    # Regex to match task-rest_run files and extract ScanDir_ID (preserve all numbers, including leading zeros)
    pattern = r'sub-(\d+)_ses-\d+_task-rest.*\.nii\.gz'
    
    # Search files in directory
    for file in os.listdir(input_dir):
        if 'task-rest_run' in file:
            match = re.search(pattern, file)
            if match:
                Image.append(file)
                ScanDir_ID.append(match.group(1).lstrip('0'))
    
    # Load the megafile CSV
    columns = ['ScanDir ID', "DX"]
    mega_df = pd.read_csv(mega_file, usecols=columns, sep='\t')
    
    # Create a mapping of ScanDir_ID to DX
    id_to_dx = pd.Series(mega_df.DX.values, index=mega_df['ScanDir ID'].astype(str)).to_dict()
    
    # Map DX values for each ScanDir_ID
    DX_values = [id_to_dx.get(f'{sid}', 'N/A') for sid in ScanDir_ID]
    
    # Create a DataFrame for output
    output_df = pd.DataFrame({
        'Image': Image,
        'ScanDir_ID': ScanDir_ID,
        'DX': DX_values
    })
    
    # Save to CSV
    output_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

# Example usage
input_directory = '/root/autodl-tmp/CNNLSTM/Project/Data'
megafile_path = '/root/autodl-tmp/CNNLSTM/archive/Diagnosing-ADHD-With-ConvLSTM/References/adhd200_preprocessed_phenotypics.tsv'
output_csv = 'finalist.csv'
process_task_rest_files(input_directory, megafile_path, output_csv)
