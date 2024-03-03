import os

def list_npy_files_to_file(root_folder, output_file_name):
    # List to store file paths
    npy_files = []
    
    # Walk through the directory
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith(".obj.npy"):
                # Construct the full path and add it to the list
                npy_files.append(os.path.join(root, file))

    # Sort the list for consistent ordering
    npy_files.sort()

    # Writing to the file
    with open(output_file_name, 'w') as file:
        file.write("experiment_name: overfit_relu\n")
        file.write("dataset.folder: [\n")
        for npy_file in npy_files:
            file.write(f'    {npy_file},\n')
        file.write("  ]\n")

# Replace 'your_folder_path' and 'your_output_file_name.txt' with your actual folder path and desired output file name
list_npy_files_to_file('./data/final/full/train', './config/final/onceki_ckpt/train_config_generator.yaml')