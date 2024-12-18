from collections.abc import Mapping
from typing import List, Optional, Sequence, Union

import torch.utils.data
from torch.utils.data.dataloader import default_collate

from torch_geometric.data import Batch, Dataset
from torch_geometric.data.data import BaseData
from torch_geometric.data.datapipes import DatasetAdapter

from torch_geometric.data import Data

from sklearn.cluster import KMeans, MiniBatchKMeans
import numpy as np
from joblib import Parallel, delayed

import torch_geometric.nn.pool

class Collater:
    def __init__(self, follow_batch, exclude_keys):
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

    def __call__(self, batch):
        elem = batch[0]
        if isinstance(elem, BaseData):
            return Batch.from_data_list(batch, self.follow_batch,
                                        self.exclude_keys)
        elif isinstance(elem, torch.Tensor):
            return default_collate(batch)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float)
        elif isinstance(elem, int):
            return torch.tensor(batch)
        elif isinstance(elem, str):
            return batch
        elif isinstance(elem, Mapping):
            return {key: self([data[key] for data in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):
            return type(elem)(*(self(s) for s in zip(*batch)))
        elif isinstance(elem, Sequence) and not isinstance(elem, str):
            return [self(s) for s in zip(*batch)]

        raise TypeError(f'DataLoader found invalid type: {type(elem)}')

    def collate(self, batch):  # pragma: no cover
        # TODO Deprecated, remove soon.
        return self(batch)

class Eigen_Clustering_Collator:
    def __init__(self, follow_batch, exclude_keys, ratio=0.5, layer=3):
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys
        self.collator = Collater(follow_batch, exclude_keys)
        
        self.layer=layer
        self.ratio=ratio

    def __call__(self, batch):
        #modified_batch=Parallel(n_jobs=-1, backend='threading', batch_size=1)(delayed(self.Eigen_Clustering_Error_check)(data) for data in batch)
        modified_batch=[self.Eigen_Clustering_Error_check(data) for data in batch]
        return self.collator(modified_batch)
    
    def Eigen_Clustering_Error_check(self, data):
        try:
            return self.Eigen_Clustering(data)
        except Exception as e:
            print(f"Error: {e}")  # 로그로 예외 확인
            return None  # 오류 발생 시 None 반환

    def Eigen_Clustering(self, data):
        modified_data=data
        eigen_vector=data.eigen_vectors
        v=eigen_vector.shape[0]
        cluster=np.zeros((3,v))

        cluster_label=np.array(range(v))
        for i in range(self.layer):
            k=int(v*(self.ratio**(i+1)))
            kmeans = KMeans(n_clusters=k, init="k-means++" ,n_init=1)
            kmeans.fit(eigen_vector[:,:k])

            eigen_vector=kmeans.cluster_centers_

            cluster_label=kmeans.labels_[cluster_label]
            cluster[i,:]=cluster_label
            
        modified_data.x=torch.cat((data.x,torch.from_numpy(np.transpose(cluster)).float()), dim=1)

        return modified_data
    
    def collate(self, batch):  # pragma: no cover
        # TODO Deprecated, remove soon.
        return self(batch)




class DataLoader(torch.utils.data.DataLoader):
    r"""A data loader which merges data objects from a
    :class:`torch_geometric.data.Dataset` to a mini-batch.
    Data objects can be either of type :class:`~torch_geometric.data.Data` or
    :class:`~torch_geometric.data.HeteroData`.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch. (default: :obj:`False`)
        follow_batch (List[str], optional): Creates assignment batch
            vectors for each key in the list. (default: :obj:`None`)
        exclude_keys (List[str], optional): Will exclude each key in the
            list. (default: :obj:`None`)
            
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`.
    """
    def __init__(
        self,
        dataset: Union[Dataset, Sequence[BaseData], DatasetAdapter],
        batch_size: int = 1,
        shuffle: bool = False,
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
        eigen_pooling=None,
        **kwargs,
    ):
        # Remove for PyTorch Lightning:
        kwargs.pop('collate_fn', None)

        # Save for PyTorch Lightning < 1.6:
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys
        
        if eigen_pooling!=None:
            ratio, layer=eigen_pooling
            collator=Eigen_Clustering_Collator(follow_batch, exclude_keys, ratio=ratio,layer=layer)
        else:
            collator=Collater(follow_batch, exclude_keys)

        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=collator,
            **kwargs,
        )
            
