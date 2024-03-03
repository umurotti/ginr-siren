import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

# Load mesh and convert to open3d.t.geometry.TriangleMesh
armadillo_data = o3d.data.ArmadilloMesh()
mesh = o3d.io.read_triangle_mesh(armadillo_data.path)
mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

# Create a scene and add the triangle mesh
scene = o3d.t.geometry.RaycastingScene()
_ = scene.add_triangles(mesh)  # we do not need the geometry ID for mesh


query_point = o3d.core.Tensor([[10, 10, 10]], dtype=o3d.core.Dtype.Float32)

# Compute distance of the query point from the surface
unsigned_distance = scene.compute_distance(query_point)
signed_distance = scene.compute_signed_distance(query_point)
occupancy = scene.compute_occupancy(query_point)


print("unsigned distance", unsigned_distance.numpy())
print("signed_distance", signed_distance.numpy())
print("occupancy", occupancy.numpy())


min_bound = mesh.vertex.positions.min(0).numpy()
max_bound = mesh.vertex.positions.max(0).numpy()

N = 256
query_points = np.random.uniform(low=min_bound, high=max_bound,
                                 size=[N, 3]).astype(np.float32)

# Compute the signed distance for N random points
signed_distance = scene.compute_signed_distance(query_points)


xyz_range = np.linspace(min_bound, max_bound, num=32)

# query_points is a [32,32,32,3] array ..
query_points = np.stack(np.meshgrid(*xyz_range.T), axis=-1).astype(np.float32)

# signed distance is a [32,32,32] array
signed_distance = scene.compute_signed_distance(query_points)

# We can visualize a slice of the distance field directly with matplotlib
plt.imshow(signed_distance.numpy()[:, :, 15])


