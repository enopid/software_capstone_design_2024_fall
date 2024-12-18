import argparse
import os
import shutil
import sys

import numpy as np
import torch
import igl
import trimesh
import robust_laplacian
import tqdm
from sklearn.cluster import KMeans

#class/train의 구조를 train.class의 구조로 변경
def split_train_test(args):
    print('------split train test------')

    dir_list = os.listdir(args.data_path)
    train_path = os.path.join(args.data_path, 'train')
    test_path = os.path.join(args.data_path, 'test')
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)
    with tqdm.tqdm(total=len(dir_list)) as t:
        for d in dir_list:
            if not os.path.isdir(os.path.join(args.data_path, d)):
                continue
            for i, subset_name in enumerate(['train', 'test']):
                subset_mesh_path = os.path.join(args.data_path, d, subset_name)
                sub_mesh = [f for f in filter(lambda x: os.path.splitext(x)[1] in ['.obj', '.ply'],
                                            sorted(os.listdir(subset_mesh_path)))]
                os.makedirs(os.path.join(args.data_path, subset_name, d))
                for m in sub_mesh:
                    shutil.copy(os.path.join(subset_mesh_path, m), os.path.join(args.data_path, subset_name, d, m))
            shutil.rmtree(os.path.join(args.data_path, d))

            t.set_description(f"split dataset")
            t.update()


# mesh normalization, vertex in [-0.5, 0.5]와 동시에 회전과 uniscaling
def normalize_meshes(args):
    print('------normalize meshes------')
    for _, subset_name in enumerate(['train', 'test']):

        subset_mesh_path = os.path.join(args.data_path, subset_name)

        norm_mesh_path = os.path.join(args.data_path, subset_name + '_norm')
        if not os.path.exists(norm_mesh_path):
            os.makedirs(norm_mesh_path)
        for d in os.listdir(subset_mesh_path):
            if not os.path.isdir(os.path.join(subset_mesh_path, d)):
                continue
            if not os.path.exists(os.path.join(norm_mesh_path, d)):
                os.makedirs(os.path.join(norm_mesh_path, d))
            with tqdm.tqdm(total=len(os.listdir(os.path.join(subset_mesh_path, d)))) as t:
                for file in os.listdir(os.path.join(subset_mesh_path, d)):
                    file.strip()
                    if os.path.splitext(file)[1] not in ['.obj']:
                        continue

                    mesh = trimesh.load(os.path.join(subset_mesh_path, d, file), process=False)

                    if subset_name == 'test':
                        vertices = mesh.vertices - mesh.vertices.min(axis=0)
                        vertices = vertices / vertices.max()
                        mesh.vertices = vertices
                        mesh.export(os.path.join(norm_mesh_path, d, os.path.splitext(file)[0] + '.obj'))
                    else:

                        if args.augment_orient:
                            rotations_ratio = np.random.choice([0, 1, 2, 3], size=3, replace=False)
                            scales_ratio = np.random.normal(1, 0.1, size=(15, 3))
                        else:
                            rotations_ratio = [0]
                            scales_ratio = np.random.normal(1, 0.1, size=(45, 3))

                        for i in range(1):#len(rotations_ratio)):
                            # trimesh.copy() is deepcopy. copy(include_cache=False):If True, will shallow copy cached data to new mesh
                            mesh_tans_rotation = mesh.copy()
                            rotation = trimesh.transformations.rotation_matrix((np.pi / 2) * rotations_ratio[i],
                                                                            [0, 1, 0])
                            mesh_tans_rotation.apply_transform(rotation)

                            for j in range(1):#len(scales_ratio)):
                                mesh_trans_scale = mesh_tans_rotation.copy()
                                mesh_trans_scale.vertices = mesh_trans_scale.vertices * scales_ratio[j]

                                vertices = mesh_trans_scale.vertices - mesh_trans_scale.vertices.min(axis=0)
                                vertices = vertices / vertices.max()
                                mesh_trans_scale.vertices = vertices
                                mesh_trans_scale.export(
                                    os.path.join(norm_mesh_path, d,
                                                os.path.splitext(file)[0] + '_R{0}_S{1}.obj'.format(i, j)))

                    t.set_description(f"normalize / {subset_name:5} / {d:9}")
                    t.update()


def generate_cot_eigen_vectors(args):
    print('------generate cot eigen vectors------')
    for _, subset_name in enumerate(['train', 'test']):

        subset_mesh_path = os.path.join(args.data_path, subset_name + '_norm')

        eigen_vectors_path = os.path.join(args.data_path, 'eigen_vectors', subset_name)
        eigen_values_path = os.path.join(args.data_path, 'eigen_values', subset_name)
        if not os.path.exists(eigen_vectors_path):
            os.makedirs(eigen_vectors_path)
        if not os.path.exists(eigen_values_path):
            os.makedirs(eigen_values_path)

        for d in sorted(os.listdir(subset_mesh_path)):
            if not os.path.isdir(os.path.join(subset_mesh_path, d)):
                continue
            if not os.path.exists(os.path.join(eigen_vectors_path, d)):
                os.makedirs(os.path.join(eigen_vectors_path, d))
            if not os.path.exists(os.path.join(eigen_values_path, d)):
                os.makedirs(os.path.join(eigen_values_path, d))
            with tqdm.tqdm(total=len(os.listdir(os.path.join(subset_mesh_path, d)))) as t:
                for file in sorted(os.listdir(os.path.join(subset_mesh_path, d))):
                    file.strip()
                    if os.path.splitext(file)[1] not in ['.obj']:
                        continue
                    mesh = trimesh.load(os.path.join(subset_mesh_path, d, file), process=False)
                    mesh: trimesh.Trimesh

                    V = np.array(mesh.vertices)  # 정점 좌표 (n x 3 배열)
                    F = np.array(mesh.faces)     # 면 정보 (m x 3 배열)
                    cot = -igl.cotmatrix(V,F).toarray()

                    if np.sum(cot != cot.T) > 0:  # non Symmetry using roboust Lap (SGP 2020)
                        L, M = robust_laplacian.mesh_laplacian(np.asarray(mesh.vertices), np.asarray(mesh.faces))
                        cot = torch.from_numpy(L.toarray()).float().to(args.device)
                    else:
                        cot = torch.from_numpy(cot).float()
                        cot=cot.to(args.device)
                    

                    eigen_values, eigen_vectors = torch.linalg.eigh(cot)
                    ind = torch.argsort(eigen_values)[:]

                    np.save(os.path.join(eigen_vectors_path, d, os.path.splitext(file)[0] + '_eigen.npy'), eigen_vectors[:, ind].cpu().numpy())
                    np.save(os.path.join(eigen_values_path, d, os.path.splitext(file)[0] + '_eigenValues.npy'), eigen_values[ind].cpu().numpy())

                    t.set_description(f"generate eigen vec / {subset_name:5} / {d:9}")
                    t.update()


def generate_gaussian_curvature(args):
    print('------generate gaussian curvature------')
    for _, subset_name in enumerate(['train', 'test']):

        subset_mesh_path = os.path.join(args.data_path, subset_name + '_norm')

        gaussian_curvatures_path = os.path.join(args.data_path, 'gaussian_curvatures', subset_name)
        if not os.path.exists(gaussian_curvatures_path):
            os.makedirs(gaussian_curvatures_path)
        for d in os.listdir(subset_mesh_path):
            if not os.path.isdir(os.path.join(subset_mesh_path, d)):
                continue
            if not os.path.exists(os.path.join(gaussian_curvatures_path, d)):
                os.makedirs(os.path.join(gaussian_curvatures_path, d))
            with tqdm.tqdm(total=len(os.listdir(os.path.join(subset_mesh_path, d)))) as t:
                for file in sorted(os.listdir(os.path.join(subset_mesh_path, d))):
                    file.strip()
                    if os.path.splitext(file)[1] not in ['.obj']:
                        continue

                    # mesh_curvature_igl
                    # mesh_vertices, mesh_faces = igl.read_triangle_mesh(os.path.join(subset_mesh_path, d, file))
                    mesh = trimesh.load(os.path.join(subset_mesh_path, d, file), process=False)
                    mesh_gaussian_curvature = trimesh.curvature.discrete_gaussian_curvature_measure(mesh, mesh.vertices, 0)
                    if np.isnan(mesh_gaussian_curvature).sum() > 0 or np.isinf(mesh_gaussian_curvature).sum() > 0:
                        print('gaussian_curvature errors, exit')
                        sys.exit()

                    np.save(
                        os.path.join(gaussian_curvatures_path, d, os.path.splitext(file)[0] + '_gaussian_curvature.npy'),
                        mesh_gaussian_curvature)

                    t.set_description(f"generate gaussian curvature / {subset_name:5} / {d:9}")
                    t.update()


def generate_dihedral_angles(args):
    print('------generate Vertex_Face 3innerProducts------')
    for _, subset_name in enumerate(['train', 'test']):

        subset_mesh_path = os.path.join(args.data_path, subset_name + '_norm')

        V_dihedral_angles_path = os.path.join(args.data_path, 'V_dihedral_angles', subset_name)
        if not os.path.exists(V_dihedral_angles_path):
            os.makedirs(V_dihedral_angles_path)
        for d in os.listdir(subset_mesh_path):
            if not os.path.isdir(os.path.join(subset_mesh_path, d)):
                continue
            if not os.path.exists(os.path.join(V_dihedral_angles_path, d)):
                os.makedirs(os.path.join(V_dihedral_angles_path, d))
            with tqdm.tqdm(total=len(os.listdir(os.path.join(subset_mesh_path, d)))) as t:
                for file in sorted(os.listdir(os.path.join(subset_mesh_path, d))):
                    file.strip()
                    if os.path.splitext(file)[1] not in ['.obj']:
                        continue

                    mesh = trimesh.load(os.path.join(subset_mesh_path, d, file), process=False)
                    mesh: trimesh.Trimesh

                    vertex_faces_adjacency_matrix = np.zeros((mesh.vertices.shape[0], mesh.faces.shape[0]))
                    for vertex, faces in enumerate(mesh.vertex_faces):
                        for i, face in enumerate(faces):
                            if face == -1:
                                break
                            vertex_faces_adjacency_matrix[vertex, face] = 1

                    dihedral_angle = list()
                    for i in range(mesh.faces.shape[0]):
                        dihedral_angle.append(list())

                    face_adjacency = mesh.face_adjacency

                    for adj_faces in face_adjacency:
                        dihedral_angle[adj_faces[0]].append(
                            np.abs(np.dot(mesh.face_normals[adj_faces[0]], mesh.face_normals[adj_faces[1]])))
                        dihedral_angle[adj_faces[1]].append(
                            np.abs(np.dot(mesh.face_normals[adj_faces[0]], mesh.face_normals[adj_faces[1]])))

                    # process the non-watertight mesh which include some faces which dont have three neighbors.
                    for i, l in enumerate(dihedral_angle):
                        if (len(l)) == 3:
                            continue
                        l.append(1)
                        if (len(l)) == 3:
                            continue
                        l.append(1)
                        if (len(l)) == 3:
                            continue
                        l.append(1)
                        if (len(l)) != 3:
                            print(i, 'Padding Failed')
                    face_dihedral_angle = np.array(dihedral_angle).reshape(-1, 3)

                    V_dihedral_angles = np.dot(vertex_faces_adjacency_matrix, face_dihedral_angle)

                    np.save(os.path.join(V_dihedral_angles_path, d, os.path.splitext(file)[0] + '_V_dihedralAngles.npy'),
                            V_dihedral_angles)

                    t.set_description(f"generate dihedral angle / {subset_name:5} / {d:9}")
                    t.update()


def HKS(args):
    print('------generate HKS------')
    for _, subset_name in enumerate(['train', 'test']):

        subset_mesh_path = os.path.join(args.data_path, subset_name + '_norm')
        eigen_vectors_path = os.path.join(args.data_path, 'eigen_vectors', subset_name)
        eigen_values_path = os.path.join(args.data_path, 'eigen_values', subset_name)

        HKS_path = os.path.join(args.data_path, 'HKS', subset_name)
        if not os.path.exists(HKS_path):
            os.makedirs(HKS_path)
        for d in sorted(os.listdir(subset_mesh_path)):
            if not os.path.isdir(os.path.join(subset_mesh_path, d)):
                continue
            if not os.path.exists(os.path.join(HKS_path, d)):
                os.makedirs(os.path.join(HKS_path, d))
            with tqdm.tqdm(total=len(os.listdir(os.path.join(subset_mesh_path, d)))) as t:
                for file in sorted(os.listdir(os.path.join(subset_mesh_path, d))):
                    file.strip()
                    if os.path.splitext(file)[1] not in ['.obj']:
                        continue

                    eigen_vector = \
                        np.load(os.path.join(eigen_vectors_path, d, os.path.splitext(file)[0] + '_eigen.npy')).astype(
                            np.float128)
                    eigen_values = \
                        np.load(os.path.join(eigen_values_path, d, os.path.splitext(file)[0] + '_eigenValues.npy')).astype(
                            np.float128)

                    t_min = 4 * np.log(10) / eigen_values.max()
                    t_max = 4 * np.log(10) / np.sort(eigen_values)[1]
                    ts = np.linspace(t_min, t_max, num=100, dtype=np.float128)
                    exp_value = (-eigen_values[None, :, None] * ts.flatten()[None, None, :])
                    hkss = (eigen_vector[:, :, None] ** 2) * np.exp(exp_value)
                    hks = torch.tensor(np.sum(hkss, axis=1).astype(np.float64)).float()
                    hks_cat = ((hks[:, 1] - hks[:, 1].min()) / (hks[:, 1].max() - hks[:, 1].min())).unsqueeze(1)
                    for i, k in enumerate([2, 3, 4, 5, 8, 10, 15, 20]):
                        hks_norm = ((hks[:, k] - hks[:, k].min()) / (hks[:, k].max() - hks[:, k].min())).unsqueeze(1)
                        hks_cat = torch.cat((hks_cat, hks_norm), dim=1)
                    if torch.isnan(hks_cat).sum() > 0 or torch.isinf(hks_cat).sum() > 0:
                        print('hks errors, exit')
                        sys.exit()
                    np.save(os.path.join(HKS_path, d, os.path.splitext(file)[0] + '_hks.npy'), hks_cat.numpy())

                    t.set_description(f"generate HKS / {subset_name:5} / {d:9}")
                    t.update()

def generate_eigen_clustering(args):
    print('------generate eigen clustering------')
    for _, subset_name in enumerate(['train', 'test']):

        subset_mesh_path = os.path.join(args.data_path, subset_name + '_norm')
        eigen_vectors_path = os.path.join(args.data_path, 'eigen_vectors', subset_name)
        eigen_values_path = os.path.join(args.data_path, 'eigen_values', subset_name)

        eigen_clustering_path = os.path.join(args.data_path, 'Eigen_clustering', subset_name)
        if not os.path.exists(eigen_clustering_path):
            os.makedirs(eigen_clustering_path)

        for d in sorted(os.listdir(subset_mesh_path)):
            if not os.path.isdir(os.path.join(subset_mesh_path, d)):
                continue
            if not os.path.exists(os.path.join(eigen_clustering_path, d)):
                os.makedirs(os.path.join(eigen_clustering_path, d))
            with tqdm.tqdm(total=len(os.listdir(os.path.join(subset_mesh_path, d)))) as t:
                for file in sorted(os.listdir(os.path.join(subset_mesh_path, d))):
                    file.strip()
                    if os.path.splitext(file)[1] not in ['.obj']:
                        continue

                    eigen_vector = \
                        np.load(os.path.join(eigen_vectors_path, d, os.path.splitext(file)[0] + '_eigen.npy')).astype(
                            np.float128)
                    eigen_values = \
                        np.load(os.path.join(eigen_values_path, d, os.path.splitext(file)[0] + '_eigenValues.npy')).astype(
                            np.float128)

                    layer=3
                    ratio=0.5

                    cluster=[]
                    v=eigen_vector.shape[0]
                    assert int(v*(ratio**layer))>0
                    tmp=0
                    cluster_label=np.array(range(v))
                    for i in range(layer):
                        k=int(v*(ratio**(i+1)))
                        kmeans = KMeans(n_clusters=k)
                        kmeans.fit(eigen_vector[:,:k])
                        tmp+=kmeans.n_iter_
                        
                        output = np.zeros((np.max(kmeans.labels_)+1,eigen_vector.shape[1]))
                        np.add.at(output, kmeans.labels_, eigen_vector)
                        
                        counts = np.zeros_like(output)
                        np.add.at(counts, kmeans.labels_, 1)
                        eigen_vector=np.divide(output, counts, out=np.zeros_like(output), where=counts != 0)

                        cluster_label=kmeans.labels_[cluster_label]
                        cluster.append(cluster_label)

                    cluster=np.array(cluster)
                    cluster=np.transpose(cluster)
                    assert cluster.shape==(v,3)

                    np.save(os.path.join(eigen_clustering_path, d, os.path.splitext(file)[0] + '_eigenclustering.npy'), cluster)

                    t.set_description(f"generate Eigen Clustering / {subset_name:5} / {d:9}")
                    t.update()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../data/shrec_16')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--augment_orient', action='store_true')
    args = parser.parse_args()

    split_train_test(args)
    normalize_meshes(args)
    generate_cot_eigen_vectors(args)
    generate_gaussian_curvature(args)
    generate_dihedral_angles(args)
    HKS(args)
    #generate_eigen_clustering(args)
