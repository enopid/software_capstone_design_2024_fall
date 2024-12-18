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

def generate_face_cot_eigen_vectors(args):
    print('------generate cot eigen vectors------')
    for _, subset_name in enumerate(['train', 'test']):

        subset_mesh_path = os.path.join(args.data_path, subset_name + '_norm')
        eigen_vectors_path = os.path.join(args.data_path, 'eigen_vectors', subset_name)

        face_eigen_vectors_path = os.path.join(args.data_path, 'face_eigen_vectors', subset_name)
        if not os.path.exists(face_eigen_vectors_path):
            os.makedirs(face_eigen_vectors_path)

        for d in sorted(os.listdir(subset_mesh_path)):
            if not os.path.isdir(os.path.join(subset_mesh_path, d)):
                continue
            if not os.path.exists(os.path.join(face_eigen_vectors_path, d)):
                os.makedirs(os.path.join(face_eigen_vectors_path, d))
            with tqdm.tqdm(total=len(os.listdir(os.path.join(subset_mesh_path, d)))) as t:
                for file in sorted(os.listdir(os.path.join(subset_mesh_path, d))):
                    file.strip()
                    if os.path.splitext(file)[1] not in ['.obj']:
                        continue
                    mesh = trimesh.load(os.path.join(subset_mesh_path, d, file), process=False)
                    mesh: trimesh.Trimesh
                    
                    V = np.array(mesh.vertices)  # 정점 좌표 (n x 3 배열)
                    F = np.array(mesh.faces)     # 면 정보 (m x 3 배열)

                    eigen_vector = \
                        np.load(os.path.join(eigen_vectors_path, d, os.path.splitext(file)[0] + '_eigen.npy')).astype(
                            np.float128)
                    
                    face_eigen_vectors = []
                    for face in F:
                        # Get Eigenvectors of the vertices in this face
                        vertex_eigenvectors = eigen_vector[face]  # Shape: (3, N_eigenvectors)
                        mean_eigenvector = vertex_eigenvectors.mean(axis=0)  # Mean across the 3 vertices
                        face_eigen_vectors.append(mean_eigenvector)

                    face_eigen_vectors = np.array(face_eigen_vectors)
                    

                    np.save(os.path.join(face_eigen_vectors_path, d, os.path.splitext(file)[0] + 'face_eigen.npy'), face_eigen_vectors)

                    t.set_description(f"generate face eigen vec / {subset_name:5} / {d:9}")
                    t.update()

def generate_face_areas(args):
    print('------generate cot eigen vectors------')
    for _, subset_name in enumerate(['train', 'test']):

        subset_mesh_path = os.path.join(args.data_path, subset_name + '_norm')

        face_areas_path = os.path.join(args.data_path, 'areas', subset_name)
        if not os.path.exists(face_areas_path):
            os.makedirs(face_areas_path)

        for d in sorted(os.listdir(subset_mesh_path)):
            if not os.path.isdir(os.path.join(subset_mesh_path, d)):
                continue
            if not os.path.exists(os.path.join(face_areas_path, d)):
                os.makedirs(os.path.join(face_areas_path, d))
            with tqdm.tqdm(total=len(os.listdir(os.path.join(subset_mesh_path, d)))) as t:
                for file in sorted(os.listdir(os.path.join(subset_mesh_path, d))):
                    file.strip()
                    if os.path.splitext(file)[1] not in ['.obj']:
                        continue
                    mesh = trimesh.load(os.path.join(subset_mesh_path, d, file), process=False)
                    mesh: trimesh.Trimesh

                    face_areas = np.array(mesh.area_faces)
                    

                    np.save(os.path.join(face_areas_path, d, os.path.splitext(file)[0] + 'face_eigen.npy'), face_areas)

                    t.set_description(f"generate face area / {subset_name:5} / {d:9}")
                    t.update()

def generate_face_gaussian_curvature(args):
    print('------generate gaussian curvature------')
    for _, subset_name in enumerate(['train', 'test']):

        subset_mesh_path = os.path.join(args.data_path, subset_name + '_norm')

        face_gaussian_curvatures_path = os.path.join(args.data_path, 'face_gaussian_curvatures', subset_name)
        if not os.path.exists(face_gaussian_curvatures_path):
            os.makedirs(face_gaussian_curvatures_path)
        for d in os.listdir(subset_mesh_path):
            if not os.path.isdir(os.path.join(subset_mesh_path, d)):
                continue
            if not os.path.exists(os.path.join(face_gaussian_curvatures_path, d)):
                os.makedirs(os.path.join(face_gaussian_curvatures_path, d))
            with tqdm.tqdm(total=len(os.listdir(os.path.join(subset_mesh_path, d)))) as t:
                for file in sorted(os.listdir(os.path.join(subset_mesh_path, d))):
                    file.strip()
                    if os.path.splitext(file)[1] not in ['.obj']:
                        continue
                    mesh = trimesh.load(os.path.join(subset_mesh_path, d, file), process=False)
                    mesh: trimesh.Trimesh
                    
                    F = np.array(mesh.faces)     # 면 정보 (m x 3 배열)
                    vertex_normals = mesh.vertex_normals
                    face_normals = mesh.face_normals
                    mesh_gaussian_curvature = np.vstack([
                        (vertex_normals[F[:, 0]] * face_normals).sum(axis=1),
                        (vertex_normals[F[:, 1]] * face_normals).sum(axis=1),
                        (vertex_normals[F[:, 2]] * face_normals).sum(axis=1),
                    ])
                    mesh_gaussian_curvature=np.sort(mesh_gaussian_curvature, axis=0)

                    np.save(
                        os.path.join(face_gaussian_curvatures_path, d, os.path.splitext(file)[0] + 'face_gaussian_curvature.npy'),
                        mesh_gaussian_curvature)

                    t.set_description(f"generate face gaussian curvature / {subset_name:5} / {d:9}")
                    t.update()


def generate_face_dihedral_angles(args):
    print('------generate Vertex_Face 3innerProducts------')
    for _, subset_name in enumerate(['train', 'test']):

        subset_mesh_path = os.path.join(args.data_path, subset_name + '_norm')

        face_V_dihedral_angles_path = os.path.join(args.data_path, 'face_V_dihedral_angles', subset_name)
        if not os.path.exists(face_V_dihedral_angles_path):
            os.makedirs(face_V_dihedral_angles_path)
        for d in os.listdir(subset_mesh_path):
            if not os.path.isdir(os.path.join(subset_mesh_path, d)):
                continue
            if not os.path.exists(os.path.join(face_V_dihedral_angles_path, d)):
                os.makedirs(os.path.join(face_V_dihedral_angles_path, d))
            with tqdm.tqdm(total=len(os.listdir(os.path.join(subset_mesh_path, d)))) as t:
                for file in sorted(os.listdir(os.path.join(subset_mesh_path, d))):
                    file.strip()
                    if os.path.splitext(file)[1] not in ['.obj']:
                        continue

                    mesh = trimesh.load(os.path.join(subset_mesh_path, d, file), process=False)
                    mesh: trimesh.Trimesh

                    V_dihedral_angles = np.sort(mesh.face_angles, axis=1).T

                    np.save(os.path.join(face_V_dihedral_angles_path, d, os.path.splitext(file)[0] + '_V_dihedralAngles.npy'),
                            V_dihedral_angles)

                    t.set_description(f"generate face dihedral angle / {subset_name:5} / {d:9}")
                    t.update()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../data/shrec_16')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--augment_orient', action='store_true')
    args = parser.parse_args()

    generate_face_cot_eigen_vectors(args)
    generate_face_areas(args)
    generate_face_gaussian_curvature(args)
    generate_face_dihedral_angles(args)
