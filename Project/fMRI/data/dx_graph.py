import matplotlib.pyplot as plt

# Original data: DX and counts (unsorted order)
dx_values_unsorted = [0, 1, 3, 2]
counts_unsorted = [585, 212, 137, 13]

# To sort the data by DX value:
dx_values, counts = zip(*sorted(zip(dx_values_unsorted, counts_unsorted)))

# Create the bar chart
plt.figure(figsize=(8, 6))
plt.bar(dx_values, counts, color='skyblue')
plt.xlabel('DX')
plt.ylabel('Count')
plt.title('DX Distribution')
plt.xticks(dx_values)  # ensure ticks are at each DX value

# Display the plot
plt.show()
