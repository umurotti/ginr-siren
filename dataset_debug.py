from collections import defaultdict
import torch
from tqdm import tqdm


# Assuming loader_trn is your data loader
loader_trn = ... # Initialize your data loader here

# Dictionary to count occurrences of each shape
shape_counts = defaultdict(int)

# Iterating over the data loader
for it, (xt, _) in enumerate(tqdm(loader_trn, total=len(loader_trn))):
    # Assuming xt is your data tensor, adjust if your data structure is different
    # Convert the shape to a string to use it as a dictionary key
    shape_str = str(tuple(xt.shape))
    shape_counts[shape_str] += 1

# Printing the shapes and their counts
for shape, count in shape_counts.items():
    print(f'Shape: {shape}, Count: {count}')

# If you need to find shapes with only 1 occurrence (unique shapes)
unique_shapes = {shape: count for shape, count in shape_counts.items() if count == 1}
print(f'Unique shapes (count == 1): {unique_shapes}')
