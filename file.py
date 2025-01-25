import itertools

def valid_array(arr):
    # Check if a_i (i>=3) is the difference between two previous numbers
    for i in range(2, len(arr)):
        valid = False
        for j in range(i):
            for k in range(j+1, i):
                if abs(arr[j] - arr[k]) == arr[i]:
                    valid = True
                    break
            if valid:
                break
        if not valid:
            return False
    return True

def generate_arrays():
    # Generate all permutations of [1, 2, 3, ..., 10]
    all_permutations = itertools.permutations(range(1, 17))
    valid_arrays = []

    # Filter valid arrays based on the given conditions
    for arr in all_permutations:
        if arr[0] == 16 and arr[9] == 15 and valid_array(arr):
            valid_arrays.append(arr)

    return valid_arrays

# Generate and print valid arrays
valid_arrays = generate_arrays()
valid_arrays_count = len(valid_arrays)
print(f"Number of valid arrays: {valid_arrays_count}")
print(f"Valid arrays: {valid_arrays}")
