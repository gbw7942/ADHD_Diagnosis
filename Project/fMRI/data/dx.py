import csv
from collections import Counter

# Replace 'data.tsv' with your file path
filename = 'adhd200_preprocessed_phenotypics.tsv'

# Open the file and read the TSV using DictReader
with open(filename, newline='') as tsvfile:
    reader = csv.DictReader(tsvfile, delimiter='\t')
    dx_counts = Counter()
    for row in reader:
        dx_value = row['DX']
        dx_counts[dx_value] += 1

print("Distribution of DX values:")
for value, count in dx_counts.items():
    print(f"DX = {value}: {count}")
