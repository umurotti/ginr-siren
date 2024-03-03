from skimage import measure
import numpy as np
import open3d as o3d
import plyfile

def reconstruct_shape(meshes,epoch=-1,it=0,mode='train'):
    for k in range(len(meshes)):
        # try writing to the ply file
        verts = meshes[k]['vertices']
        faces = meshes[k]['faces']

        sample_mesh = o3d.geometry.TriangleMesh()

        sample_mesh.vertices = o3d.utility.Vector3dVector(vertices)
        sample_mesh.triangles = o3d.utility.Vector3iVector(faces)
        # Compute normals for the mesh
        sample_mesh.compute_vertex_normals()
        #d.visualization.draw_geometries([sample_mesh])


        #filter out small blobs:
        print("Cluster connected triangles")
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            triangle_clusters, cluster_n_triangles, cluster_area = (sample_mesh.cluster_connected_triangles())

        triangle_clusters = np.asarray(triangle_clusters)
        cluster_n_triangles = np.asarray(cluster_n_triangles)
        cluster_area = np.asarray(cluster_area)

        print("Show mesh with small clusters removed")

        triangles_to_remove = cluster_n_triangles[triangle_clusters] < 20
        sample_mesh.remove_triangles_by_mask(triangles_to_remove)
        #o3d.visualization.draw_geometries([mesh_0])

        print('filter with average with 5 iterations')
        iter = int(epoch/40)
        mesh_out = sample_mesh.filter_smooth_simple(number_of_iterations=5)
        mesh_out.compute_vertex_normals()
        #o3d.visualization.draw_geometries([mesh_out])


        verts,faces = np.asarray(mesh_out.vertices), np.asarray(mesh_out.triangles)




        voxel_grid_origin = [-0.5] * 3
        mesh_points = np.zeros_like(verts)
        mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
        mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
        mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

        num_verts = verts.shape[0]
        num_faces = faces.shape[0]

        verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

        for i in range(0, num_verts):
            verts_tuple[i] = tuple(mesh_points[i, :])

        faces_building = []
        for i in range(0, num_faces):
            faces_building.append(((faces[i, :].tolist(),)))
        faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

        el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
        el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

        ply_data = plyfile.PlyData([el_verts, el_faces])
        # logging.debug("saving mesh to %s" % (ply_filename_out))
        ply_data.write("./data/shapenet/sampling_results/" + str(epoch) + "_" +str(mode)+"_"+ str(it*len(meshes)+k) + "_poly.ply")

np_file = './data/shapenet/sampling_results/1dbcb49dfbfd0844a480511cbe2c4655_manifold.obj_occ_grid.npy'
threshold = 0.5
res_path = ""



occupancies = np.load(np_file)
print(occupancies.shape)

vertices, faces, _, _ = measure.marching_cubes(occupancies, level=threshold)

print(vertices, faces)

meshes=[]
tmp={}
tmp['vertices'], tmp['faces']=vertices,faces
meshes.append(tmp)
tmp = {}
reconstruct_shape(meshes)