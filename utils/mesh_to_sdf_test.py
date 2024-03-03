from mesh_to_sdf import mesh_to_voxels
import time
import trimesh
import skimage
import argparse

from mesh_to_sdf import sample_sdf_near_surface,get_surface_point_cloud,scale_to_unit_sphere
import pyrender
import numpy as np
import os

'''
path = '/home/umaru/praktikum/changed_version/HyperDiffusion/data/02691156_manifold/1a6ad7a24bb89733f412783097373bdc.obj'
mesh = trimesh.load(path)
start = time.time()
voxels = mesh_to_voxels(mesh, 32, pad=True,sign_method='depth')

vertices, faces, normals, _ = skimage.measure.marching_cubes(voxels, level=0.01)
mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
end =time.time()
print(end-start)
#mesh.show()
folder = '/home/umaru/praktikum/changed_version/ginr-ipc/data/shapenet/sdf_test/obj'
save_folder = '/home/umaru/praktikum/changed_version/ginr-ipc/data/shapenet/sdf_test/npy'
'''


parser = argparse.ArgumentParser()
parser.add_argument('--mode', help='Specify the path.', required=True)
parser.add_argument('--input_folder', help='Specify the configuration file.', required=True)
parser.add_argument('--output_folder', help='Specify the configuration file.', required=True)

args = parser.parse_args()

mode=args.mode
input_folder=args.input_folder
output_folder=args.output_folder


#mode = 'siren_sdf'
#input_folder = '/home/umaru/praktikum/changed_version/HyperDiffusion/data/02691156_manifold/'
#output_folder = './data'


filter = '/home/umaru/praktikum/changed_version/ginr-ipc/data/shapenet/shape_filter_small.txt'

files = []
filter_list = []

with open(filter, "r") as text:
    for line in text:
        file_name = line.strip()
        filter_list.append(file_name)

for file in os.listdir(input_folder):
    if file in filter_list:
        files.append(file)

i=0

# Step 1: Read in the Mesh
for file in files:
    start = time.time()
    #path = "/home/umaru/Downloads/dragon_recon/dragon_vrip.ply"
    #mesh= trimesh.load(path)
    mesh = trimesh.load(os.path.join(input_folder,file))

    if mode == 'sdf' or mode =='occ':
        points, sdf, grads = sample_sdf_near_surface(mesh, number_of_points=20000,sign_method='depth',return_gradients=True)#1000w
        point_cloud = np.concatenate([points.reshape(-1, 3), sdf.reshape(-1,1),grads.reshape(-1,3)], axis=-1)


        print(point_cloud.shape)
        #save_path = os.path.join(save_folder, 'test')
        print(i)
        i = i + 1
        save_path = os.path.join(output_folder, file)
        np.save(save_path, point_cloud)
        end =time.time()
        print(end-start)

    elif mode == 'siren_sdf':
        mesh = scale_to_unit_sphere(mesh)
        pcd = get_surface_point_cloud(mesh,surface_point_method='scan',calculate_normals=True)
        points,normal = pcd.get_random_surface_points(count=300000)
        #print(points.shape)
        #print(normal.shape)
        result = np.concatenate([points,normal],axis=-1)
        #print(result.shape)
        save_path = os.path.join(output_folder, file)
        np.save(save_path, result)
        #print(np.count_nonzero(sdf>0.0))
        #print(np.count_nonzero(sdf<0.0))
        #print(np.mean(sdf))

    else:
        print("wrong mode")
    # for visualization    
    '''
    colors = np.zeros(points.shape)
    colors[sdf < 0.00, 2] = 1
    colors[sdf > 0.00, 0] = 1
    cloud = pyrender.Mesh.from_points(points,colors)
    scene = pyrender.Scene()
    scene.add(cloud)


    #scaled_normal = 0.01 * grads/np.linalg.norm(grads,axis=-1,keepdims=True)
    #line_set=[]
        # Create LineSet for the point and its normal
   #for points,normal in zip(points,scaled_normal):

    #   line = pyrender.Primitive(positions=np.array([points, points+normal]),mode=1)
    #line_set.positions = points
    #line_set.normals = scaled_normal
    # Add the LineSet to the scene
    #    line_set.append(line)
        #break
    #node = pyrender.Mesh(primitives=line_set)
    # Add the Node to the scene
    #scene.add_node(node)
    #scene.add(node)
    viewer = pyrender.Viewer(scene, use_raymond_lighting=True, point_size=3)
    '''

