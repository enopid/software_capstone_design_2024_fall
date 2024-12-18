import gc
import os
import sys
import tqdm

import numpy as np
import trimesh

import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Dataset
from torch_geometric.data import Data

class FaceClsVITDataset(InMemoryDataset):
    def __init__(self, args, set_type='train', is_train=True):
        # init
        self.args = args
        self.set_type = set_type
        self.is_train = is_train

        if not os.path.exists(self.args.data_path):
            raise NameError(f"data_path 잘못입력한듯 {self.args.data_path} 요 폴더 확실히 있음?")
        
        self._set_dataset()
        super(FaceClsVITDataset, self).__init__(self.data_path)
        self.load(self.processed_file_names)

    def _set_dataset(self):        
        if self.args.dataset=="shrec_16":
            self.data_path=os.path.join(self.args.data_path, self.args.dataset)
        elif self.args.dataset=="cubes":
            self.data_path=os.path.join(self.args.data_path, self.args.dataset)
        elif self.args.dataset=="Manifold40":
            self.data_path=os.path.join(self.args.data_path, self.args.dataset)
        else:
            raise NameError(f"dataset 이름 잘못입력함 {self.args.dataset}")
    
    @property
    def raw_file_names(self):
        r"""The name of the files in the :obj:`self.raw_dir` folder that must
        be present in order to skip downloading.
        """
        folder_names=["eigen_values", "face_eigen_vectors"]
        folder_names=[os.path.join(self.raw_dir, f) for f in folder_names]
        # 최종적으로 저장될 파일의 전체 경로
        return folder_names
    
    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, 'face_processed', self.set_type)
    
    @property
    def processed_file_names(self):
        r"""The name of the files in the :obj:`self.processed_dir` folder that
        must be present in order to skip processing.
        """
        
        # 최종적으로 저장될 파일의 전체 경로
        return os.path.join(self.processed_dir, "data.pt")

    def download(self) -> None:
        r"""Downloads the dataset to the :obj:`self.raw_dir` folder."""

        #일단은 skip
        return 0
        raise FileExistsError("{self.args.dataset}} 압축파일 없는 뎁쇼 ")

    def process(self) -> None:
        r"""Processes the dataset to the :obj:`self.processed_dir` folder."""
        datas = list()

        class_num = torch.zeros(self.args.num_classes)
        gt_dict = {d: int(i) for i, d in
                   enumerate(sorted(os.listdir(os.path.join(self.raw_dir, self.set_type + '_norm'))))}

        # data path
        mesh_path = os.path.join(self.raw_dir, self.set_type + '_norm')
        face_vector_path = os.path.join(self.raw_dir, 'face_eigen_vectors', self.set_type)
        gaussian_curvature_path = os.path.join(self.raw_dir, 'face_gaussian_curvatures', self.set_type)
        V_dihedral_angles_path = os.path.join(self.raw_dir, 'face_V_dihedral_angles', self.set_type)

        dirs_list = [f for f in sorted(os.listdir(mesh_path))]
        with tqdm.tqdm(total=len(dirs_list)) as t:
            for d in dirs_list:
                if not os.path.isdir(os.path.join(mesh_path, d)):
                    continue
                file_list = sorted(os.listdir(os.path.join(mesh_path, d)))
                for file in file_list:
                    file.strip()
                    if self.is_train and 'Manifold40' in self.raw_dir.split('/'):
                        if file.split('_')[-1] not in ['S0.obj', 'S1.obj', 'S2.obj', 'S3.obj']:
                            continue

                    # load mesh
                    mesh = trimesh.load(os.path.join(mesh_path, d, file), process=False, force='mesh')
                    mesh: trimesh.Trimesh

                    eigen_vector = torch.from_numpy(
                        np.load(os.path.join(face_vector_path, d, os.path.splitext(file)[0] + 'face_eigen.npy')).astype(np.float32)).half()
                    if torch.isnan(eigen_vector).sum() > 0 or torch.isinf(eigen_vector).sum() > 0:
                        print('eigen_vector errors, exit')
                        sys.exit()

                    # normalize the gaussian curvature
                    gaussian_curvature = torch.from_numpy(np.load(os.path.join(gaussian_curvature_path, d,
                                                                                        os.path.splitext(file)[
                                                                                            0] + 'face_gaussian_curvature.npy'))).half()
                    if torch.isnan(gaussian_curvature).sum() > 0 or torch.isinf(gaussian_curvature).sum() > 0:
                        print('gaussian_curvature errors, exit')
                        sys.exit()

                    V_dihedral_angles = torch.from_numpy(
                        np.load(
                            os.path.join(V_dihedral_angles_path, d, os.path.splitext(file)[0] + '_V_dihedralAngles.npy'))).half()
                    if torch.isnan(V_dihedral_angles).sum() > 0 or torch.isinf(V_dihedral_angles).sum() > 0:
                        print('vf_dihedral_angle errors, exit')
                        sys.exit()

                    F = mesh.faces
                    V = mesh.vertices

                    face_center = V[F.flatten()].reshape(-1, 3, 3).mean(axis=1)
                    # the input features (1+3+3+3+3)
                    features = torch.cat(
                        (torch.from_numpy(np.array(mesh.area_faces)).unsqueeze(1).half(),
                        torch.from_numpy(np.array(mesh.face_normals)).half(),
                        torch.from_numpy(np.array(face_center)).half(),
                        gaussian_curvature.T, 
                        V_dihedral_angles.T
                        ), dim=1)

                    x=features.half()
                    
                    faces = mesh.faces  # Shape: (M, 3)
                    num_faces = faces.shape[0]

                    # Create a dictionary to store edges and their connected faces
                    edge_to_faces = {}

                    # Iterate through each face and its edges
                    for face_idx, face in enumerate(faces):
                        edges = [
                            tuple(sorted((face[0], face[1]))),
                            tuple(sorted((face[1], face[2]))),
                            tuple(sorted((face[2], face[0])))
                        ]
                        for edge in edges:
                            if edge not in edge_to_faces:
                                edge_to_faces[edge] = []
                            edge_to_faces[edge].append(face_idx)

                    # Create connections between faces that share an edge
                    connections = set()
                    for face_indices in edge_to_faces.values():
                        if len(face_indices) > 1:
                            for i in range(len(face_indices)):
                                for j in range(i + 1, len(face_indices)):
                                    connections.add((face_indices[i], face_indices[j]))

                    # Convert connections to edge_index
                    edge_index = torch.tensor(list(connections), dtype=torch.long).t()  # Shape: (2, E)

                    y=torch.tensor([gt_dict[d]])
                    data=Data(x,edge_index=edge_index,y=y)
                    data.meshpath=mesh_path
                    data.eigen_vectors=np.load(os.path.join(face_vector_path, d, os.path.splitext(file)[0] + 'face_eigen.npy')).astype(np.float16)
                    datas.append(data)

                    del mesh, eigen_vector, gaussian_curvature, V_dihedral_angles, features, data
                    gc.collect()
                t.set_description(f"processing {self.set_type} dataset")
                t.update()
        
        self.save(datas,self.processed_file_names)
    
    def save(self, datas, path):
        data, slices=self.collate(datas)
        self._data, self.slices=(data, slices)
        torch.save((data,slices),path)
    
    def load(self,path):
        self._data, self.slices=torch.load(path,weights_only=False)