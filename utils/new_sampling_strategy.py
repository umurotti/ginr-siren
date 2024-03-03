import os
import time

import open3d as o3d
import numpy as np
import igl
import matplotlib.pyplot as plt


strategy = 'sdf'
#strategy = 'occ'


folder = '/home/umaru/praktikum/changed_version/ginr-ipc/data/shapenet/sdf_test/obj'
save_folder = '/home/umaru/praktikum/changed_version/ginr-ipc/data/shapenet/sdf_test/npy'



#strategy = 'sdf'
#strategy = 'occ'
strategy = 'occ_grid'

folder = './data/shapenet_mesh/manifold'
save_folder = './data/shapenet/sampling_results'

filter = './data/shapenet/single_filter.txt'

files = []
filter_list = []

with open(filter, "r") as text:
    for line in text:
        file_name = line.strip()
        filter_list.append(file_name)

for file in os.listdir(folder):
    if file in filter_list:
        files.append(file)

i=0


# Step 1: Read in the Mesh
for file in files:
    mesh = o3d.io.read_triangle_mesh(os.path.join(folder,file))

    vertices = mesh.vertices
    vertices -= np.mean(vertices, axis=0, keepdims=True)

    v_max = np.amax(vertices)
    v_min = np.amin(vertices)
    vertices *= 0.5 * 0.95 / (max(abs(v_min), abs(v_max)))

    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    #self.obj = obj

    #total_points =  1000000
    n_points_uniform = 300 # int(total_points * 0.5)
    n_points_surface = 400 # total_points

    #n_points_surface = 1000000
    #n_points_uniform = 1000000


    if strategy == 'occ':
        points_uniform = np.random.uniform(
            -0.5, 0.5, size=(n_points_uniform, 3)
        )

        min_bound = np.array([-0.5,-0.5,-0.5])
        max_bound = np.array([0.5, 0.5, 0.5])
        xyz_range = np.linspace(min_bound, max_bound, num=128)

        grid = np.meshgrid(*xyz_range.T)
        points_uniform = np.stack(grid, axis=-1).astype(np.float32)
        points_uniform = points_uniform.reshape((-1,3))

        start = time.time()
        print('sampling near surface...')
        points_surface = np.asarray(mesh.sample_points_poisson_disk(number_of_points=n_points_surface).points)
        end = time.time()
        print('sampling takes: '+str(end-start))


        res = 0.0001
        points_surface_1 = points_surface + 0.001 * np.random.randn(n_points_surface, 3)
        points_surface_2 = points_surface + 0.01 * np.random.randn(n_points_surface, 3)
        points_surface_3 = points_surface + 0.0001 * np.random.randn(n_points_surface, 3)
        #points_surface_4 = points_surface + 0.00001 * np.random.randn(n_points_surface, 3)


        points = np.concatenate([points_surface_1,points_surface_2,points_surface_3, points_uniform], axis=0)
        labels = np.zeros(points.shape[0])
        labels[:-n_points_uniform] = 1 # 1 means near surface



        inside_surface_values = igl.fast_winding_number_for_meshes(
            np.asarray(mesh.vertices), np.asarray(mesh.triangles), points
        )

        thresh = 0.5


        occupancies_winding = np.piecewise(
            inside_surface_values,
            [inside_surface_values < thresh, inside_surface_values >= thresh],
            [0, 1],
        )

        occupancies = occupancies_winding[..., None]



        #print(points.shape, occupancies.shape, occupancies.sum())

        point_cloud = points
        point_cloud = np.hstack((point_cloud, occupancies, labels[:,None]))
        print(point_cloud.shape, points.shape, occupancies.shape)
        print(i)
        i=i+1
        save_path = os.path.join(save_folder,file)
        np.save(save_path,point_cloud)

    elif strategy == 'occ_grid':
        mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        scene = o3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(mesh)  # we do not need the geometry ID for mesh

        min_bound = mesh.vertex.positions.min(0).numpy()
        max_bound = mesh.vertex.positions.max(0).numpy()

        #min_bound = np.array([-0.5,-0.5,-0.5])
        #max_bound = np.array([0.5, 0.5, 0.5])
        
        min_bound = np.array([min_bound,min_bound,min_bound])
        max_bound = np.array([max_bound, max_bound, max_bound])
        
        xyz_range = np.linspace(min_bound, max_bound, num=256)

        # query_points is a [32,32,32,3] array ..
        grid = np.meshgrid(*xyz_range.T)
        query_points = np.stack(grid, axis=-1).astype(np.float32)
        
        
        # signed distance is a [32,32,32] array
        signed_distance = scene.compute_signed_distance(query_points).numpy()
        occupancy = scene.compute_occupancy(query_points).numpy()
        
        print(str(signed_distance.shape)+"_"+str(occupancy.shape))

        sdf_volume = np.concatenate([query_points,signed_distance[:,:,:,None],occupancy[:,:,:,None]],axis=-1)
        print(sdf_volume.shape)

        sdf_data= sdf_volume.reshape((-1,5))
        
        save_path = os.path.join(save_folder,file + '_' + strategy)
        np.save(save_path, occupancy)
        
        plt.imshow(signed_distance[16, : , :])
        
    else:
        start = time.time()
        print('sampling near surface...')
        res = 32

        num_points = res ** 3
        min_bound = np.array([-0.5, -0.5, -0.5])
        max_bound = np.array([0.5, 0.5, 0.5])
        xyz_range = np.linspace(min_bound, max_bound, num=res)

        grid = np.meshgrid(*xyz_range.T)

        points_uniform = np.stack(grid, axis=-1).astype(np.float32).reshape(-1,3)

        n_points_uniform =res**3
        n_points_surface = 250000

        points_surface = np.asarray(mesh.sample_points_poisson_disk(number_of_points=n_points_surface).points,dtype=np.float32)
        points_surface_1 = points_surface + 0.001 * np.random.randn(n_points_surface, 3)
        points_surface_2 = points_surface + 0.01 * np.random.randn(n_points_surface, 3)

        query_points = np.concatenate([points_surface_1,points_surface_2, points_uniform], axis=0,dtype=np.float32)
        labels = np.zeros(query_points.shape[0])
        labels[:-n_points_uniform] = 1  #1 means near surface

        # points_uniform = points_uniform.reshape((-1, 3))

        mesh_old = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        scene = o3d.t.geometry.RaycastingScene()

        _ = scene.add_triangles(mesh_old)  # we do not need the geometry ID for mesh
        signed_distance = scene.compute_signed_distance(query_points).numpy().reshape(-1,1)


        # occupancy = scene.compute_occupancy(query_points).numpy()
        inside_surface_values = igl.fast_winding_number_for_meshes(
            np.asarray(mesh.vertices), np.asarray(mesh.triangles), query_points.reshape(-1, 3).astype(np.double)
        )

        thresh = 0.5


        occupancies_winding = np.piecewise(
            inside_surface_values,
            [inside_surface_values < thresh, inside_surface_values >= thresh],
            [1, -1],
        )

        # for vis:

        #occupancies_winding_bool = np.piecewise(
        #    inside_surface_values,
        #    [inside_surface_values < thresh, inside_surface_values >= thresh],
        #    [1, 0],
        #)
        #bool_values = occupancies_winding_bool.astype(bool)
        # Create Open3D point cloud
        #point_cloud = o3d.geometry.PointCloud()
        #point_cloud.points = o3d.utility.Vector3dVector(query_points)

        # Set colors based on boolean values
        #colors = np.zeros((query_points.shape[0], 3))
        #colors[bool_values] = [1, 0, 0]  # Set red color for True
        #colors[~bool_values] = [0, 0, 1]  # Set blue color for False
        #point_cloud.colors = o3d.utility.Vector3dVector(colors)
        #o3d.visualization.draw_geometries([point_cloud],)

        #occupancy = np.abs(occupancies_winding)

        print(i)
        i=i+1
        point_cloud = np.concatenate([query_points.reshape(-1, 3),occupancies_winding[:,None],signed_distance,labels[:,None]],axis=-1)
        print(point_cloud.shape)
        print(i)
        save_path = os.path.join(save_folder,file)
        np.save(save_path,point_cloud)
        end = time.time()
        print('sampling takes: '+str(end-start))

        # for visualization

        #vertices, faces, _, _ = measure.marching_cubes(a, level=-0.003)
        #sample_mesh = o3d.geometry.TriangleMesh()

        #sample_mesh.vertices = o3d.utility.Vector3dVector(vertices)
        #sample_mesh.triangles = o3d.utility.Vector3iVector(faces)
        # Compute normals for the mesh

        #sample_mesh.compute_vertex_normals()
        #o3d.visualization.draw_geometries([sample_mesh], mesh_show_back_face=True, mesh_show_wireframe=False)

        
        # We can visualize a slice of the distance field directly with matplotlib
        #plt.imshow(signed_distance.numpy()[16, : , :])


