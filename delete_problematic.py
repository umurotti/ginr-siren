import os
import shutil

# Path to the folder containing files you want to delete
folder_path = './data/final/full/train'

# Path to the text file containing problematic file names
problematic_file_path = './data/final/full/plane_problematic_shapes.txt'

# Reading problematic names into a set for faster lookup
with open(problematic_file_path, 'r') as file:
    problematic_names = set(file.read().splitlines())

# Looping through each file and directory in the folder
for item in os.listdir(folder_path):
    # Extracting the name of the item without its extension(s)
    name_without_extension = os.path.splitext(os.path.splitext(item)[0])[0]
    
    # Checking if the item name (without extension) is in the list of problematic names
    if name_without_extension in problematic_names:
        # Constructing the full path to the item
        item_path = os.path.join(folder_path, item)
        
        # Checking if it's a file or a directory and deleting accordingly
        if os.path.isfile(item_path):
            os.remove(item_path)
            print(f'Deleted file: {item}')
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)
            print(f'Deleted directory: {item}')