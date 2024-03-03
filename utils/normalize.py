import numpy as np

# Assuming you have a NumPy array named 'my_array'
my_array = np.array([[1, 2, 3, 4, 5, 6],
                    [7, 8, 9, 10, 11, 12],
                    [13, 14, 15, 16, 17, 18]])

# Specify the file name
file_name = "/home/umaru/Downloads/dragon_recon/dragon_vrip_normalize.xyz"
path = '/home/umaru/Downloads/dragon_recon/dragon_vrip_res4.xyz'
point_cloud = np.genfromtxt(path)
coords = point_cloud[:, :3]
normals = point_cloud[:, 3:]

# Reshape point cloud such that it lies in bounding box of (-1, 1) (distorts geometry, but makes for high
# sample efficiency)
coords -= np.mean(coords, axis=0, keepdims=True)
keep_aspect_ratio = True
if keep_aspect_ratio:
    coord_max = np.amax(coords)
    coord_min = np.amin(coords)
else:
    coord_max = np.amax(coords, axis=0, keepdims=True)
    coord_min = np.amin(coords, axis=0, keepdims=True)

coords = (coords - coord_min) / (coord_max - coord_min)
coords -= 0.5
coords *= 2.

my_array = np.concatenate([coords,normals],axis=1)

# Write each row of the array as a separate line in the text file
with open(file_name, 'w') as file:
    for row in my_array:
        # Convert each element in the row to a string and join them with a space
        line = ' '.join(map(str, row))
        # Write the line to the file
        file.write(line + '\n')

