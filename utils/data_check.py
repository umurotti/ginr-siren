import time

import numpy
import open3d as o3d
import numpy as np
from skimage import measure
# part 1 for checking generated data

# Generate some sample data (replace this with your actual data)
shape_path = '/home/umaru/praktikum/changed_version/ginr-ipc/data/shapenet/overfit/2a966a7e0b07a5239a6e43b878d5b335.obj.npy'
data = np.load(shape_path)

points_uniform=128**3
num_points = data[:-points_uniform,3].shape[0]
points = data[:-points_uniform,:3]
bool_values = data[:-points_uniform,3].astype(bool)


# Create Open3D point cloud
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points)

# Set colors based on boolean values
colors = np.zeros((num_points, 3))
colors[bool_values] = [1, 0, 0]  # Set red color for True
colors[~bool_values] = [0, 0, 1]  # Set blue color for False
point_cloud.colors = o3d.utility.Vector3dVector(colors)

# Visualize the point cloud
#o3d.visualization.draw_geometries([point_cloud])


# part 2 for comparing surface

import open3d as o3d
import numpy as np
import igl

path = '/home/umaru/praktikum/changed_version/HyperDiffusion/data/02691156_manifold/2a966a7e0b07a5239a6e43b878d5b335.obj'
# Step 1: Read in the Mesh
mesh = o3d.io.read_triangle_mesh(path)

vertices = mesh.vertices
vertices -= np.mean(vertices, axis=0, keepdims=True)

v_max = np.amax(vertices)
v_min = np.amin(vertices)
vertices *= 0.5 * 0.95 / (max(abs(v_min), abs(v_max)))

mesh.vertices = o3d.utility.Vector3dVector(vertices)

voxel_size = 1/1024
start = time.time()
width =0.15
height=0.025
depth =0.2
origin= [0.1,-0.04,-0.1]
start =time.time()
voxel = o3d.geometry.VoxelGrid.create_dense(origin=np.array(origin,dtype=np.float).reshape(3,1),color=np.array([1.0,0.0,0.0],dtype=np.float).reshape(3,1),voxel_size=voxel_size,width=width,height=height,depth=depth)
end =time.time()
print(end-start)

# Visualize the voxel grid
#voxels = voxel.get_voxels()  # returns list of voxels
#indices_index = np.stack(list(vx.grid_index for vx in voxels))
iq = [int(width/voxel_size),int(height/voxel_size),int(depth/voxel_size)]

#overall_index = np.arange(0, iq[0]*iq[1]*iq[2], 1)
#samples = np.zeros(iq[0]*iq[1]*iq[2], 3)

# transform first 3 columns
# to be the x, y, z index
#start =time.time()
#samples[:, 2] = overall_index % iq[0]
#samples[:, 1] = (overall_index.long() / iq[1]) % iq[0]
#samples[:, 0] = ((overall_index.long() / iq[2]) / iq[1]) % iq[0]

#end =time.time()
#print(end-start)


start =time.time()
query_points = np.zeros((iq[0],iq[1],iq[2],3))
# Assuming query_points is a 3D NumPy array
i, j, k = np.indices(query_points.shape[:-1])
query_points[:, :, :, :] = np.stack([i, j, k], axis=-1)

query_points = (query_points*voxel_size + numpy.asarray(origin)).astype(np.float32)

end =time.time()
print(end-start)

#point_cloud = o3d.geometry.PointCloud()
#point_cloud.points = o3d.utility.Vector3dVector(query_points.reshape(-1,3))



#mesh.compute_vertex_normals()
#o3d.visualization.draw_geometries([mesh,voxel],mesh_show_wireframe=True)

res=128

num_points = res**3
min_bound = np.array([-0.5, -0.5, -0.5])
max_bound = np.array([0.5, 0.5, 0.5])
xyz_range = np.linspace(min_bound, max_bound, num=res)

grid = np.meshgrid(*xyz_range.T)

#query_points = np.stack(grid, axis=-1).astype(np.float32)
#point_cloud = o3d.geometry.PointCloud()
#point_cloud.points = o3d.utility.Vector3dVector(query_points)

#points_uniform = points_uniform.reshape((-1, 3))

mesh_old = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
scene = o3d.t.geometry.RaycastingScene()
_ = scene.add_triangles(mesh_old)  # we do not need the geometry ID for mesh

start = time.time()
query_points = query_points.reshape(iq[0],iq[1],iq[2],3)
signed_distance = scene.compute_signed_distance(query_points).numpy()
#occupancy = scene.compute_occupancy(query_points).numpy()

end =time.time()
print(end-start)


inside_surface_values = igl.fast_winding_number_for_meshes(
    np.asarray(mesh.vertices), np.asarray(mesh.triangles), query_points.reshape(-1,3).astype(np.double)
)

thresh = 0.5

occupancies_winding = np.piecewise(
    inside_surface_values,
    [inside_surface_values < thresh, inside_surface_values >= thresh],
    [-1, 1],
)

bool_values = occupancies_winding.astype(bool)

#sdf_volume = np.concatenate([query_points, occupancy[:, :, :, None]], axis=-1)
#sdf_data = sdf_volume.reshape((-1, 4))

#bool_values = sdf_data[:,3].astype(bool)
#colors = np.zeros((num_points, 3))
#colors[bool_values] = [1, 0, 0]  # Set red color for True
#colors[~bool_values] = [0, 0, 1]  # Set blue color for False

#point_cloud = o3d.geometry.PointCloud()
#point_cloud.points = o3d.utility.Vector3dVector(query_points)
#point_cloud.colors = o3d.utility.Vector3dVector(colors)
#o3d.visualization.draw_geometries([point_cloud])

occupancy = occupancies_winding.reshape(iq[0],iq[1],iq[2])
a = occupancy * np.abs(signed_distance)
end =time.time()
print(end-start)
vertices, faces, _, _ = measure.marching_cubes(a, level=-0.001)
sample_mesh = o3d.geometry.TriangleMesh()
end =time.time()
print(end-start)

sample_mesh.vertices = o3d.utility.Vector3dVector(vertices)
sample_mesh.triangles = o3d.utility.Vector3iVector(faces)
# Compute normals for the mesh

sample_mesh.compute_vertex_normals()
end = time.time()
print("using_time: "+str(end-start))
o3d.visualization.draw_geometries([sample_mesh],mesh_show_back_face=True,mesh_show_wireframe=False)


# Step 2: Sample Points on the Mesh
num_points = 100000  # Adjust the number of points as neede
query = mesh.sample_points_poisson_disk(number_of_points=num_points)
query_points = np.asarray(query.points)

# Create Open3D point cloud
#point_cloud = o3d.geometry.PointCloud()
#point_cloud.points = o3d.utility.Vector3dVector(points)

# Visualize the point cloud
#o3d.visualization.draw_geometries([points])




# part 3 for calculating density
import open3d as o3d
import numpy as np
from scipy.spatial import cKDTree

# Assume you already have a point cloud 'point_cloud' with sampled points near the surface

# Convert Open3D point cloud to NumPy array
points = np.asarray(point_cloud.points)

# Set the radius for density estimation
radius = 0.01  # Adjust this value based on your requirements

# Build a k-d tree for efficient nearest neighbor search
kdtree = cKDTree(points)

# Calculate point density for each point
point_densities = kdtree.query_ball_point(query_points, r=radius)
point_density_values = [len(density) for density in point_densities]

# Add point density as colors to the point cloud
density_color = np.array(point_density_values) / max(point_density_values)
zeros = np.zeros(density_color.shape)
query.colors = o3d.utility.Vector3dVector(np.column_stack([density_color, zeros, 1 - density_color]))

print(np.array(point_density_values).mean())
# Visualize the point cloud with density information
o3d.visualization.draw_geometries([query])
