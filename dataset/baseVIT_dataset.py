import sys

from torch_geometric.data import Dataset
#from torch_geometric.loader import DataLoader
from dataset.VITdataloader import DataLoader

from dataset.VITdataset_cls import ClsVITDataset
from dataset.VITdataset_face_cls import FaceClsVITDataset

from dataset.VITdataset_seg import SegVITDataset


class BaseDataset:
    def __init__(self, args):
        self.args = args

    def face_classification_dataset(self):
        if "Eigen" in self.args.netvit.split("_"):
            eigen_pooling=(self.args.eigen_ratio, self.args.eigen_layer)
        else:
            eigen_pooling=None

        if self.args.mode == 'train':
            train_ds = FaceClsVITDataset(self.args, set_type='train', is_train=True)
            train_dl = DataLoader(train_ds, batch_size=self.args.batch_size, shuffle=True, pin_memory=False,
                                  num_workers=self.args.num_workers, prefetch_factor=self.args.prefetch_factor, eigen_pooling=eigen_pooling)

        test_ds = FaceClsVITDataset(self.args, set_type='test', is_train=False)
        test_dl = DataLoader(test_ds, batch_size=self.args.batch_size, shuffle=False, pin_memory=False,
                             num_workers=self.args.num_workers, prefetch_factor=self.args.prefetch_factor, eigen_pooling=eigen_pooling)
        if self.args.mode == 'train':
            print(f"train dataset size : {len(train_ds)}")
            print(f"test dataset size : {len(test_ds)}")
            return train_dl, test_dl
        else:
            print(f"test dataset size : {len(test_ds)}")
            return test_dl

    def classification_dataset(self):
        if "Eigen" in self.args.netvit.split("_"):
            eigen_pooling=(self.args.eigen_ratio, self.args.eigen_layer)
        else:
            eigen_pooling=None

        if self.args.mode == 'train':
            train_ds = ClsVITDataset(self.args, set_type='train', is_train=True)
            train_dl = DataLoader(train_ds, batch_size=self.args.batch_size, shuffle=True, pin_memory=False,
                                  num_workers=self.args.num_workers, prefetch_factor=self.args.prefetch_factor, eigen_pooling=eigen_pooling)

        test_ds = ClsVITDataset(self.args, set_type='test', is_train=False)
        test_dl = DataLoader(test_ds, batch_size=self.args.batch_size, shuffle=False, pin_memory=False,
                             num_workers=self.args.num_workers, prefetch_factor=self.args.prefetch_factor, eigen_pooling=eigen_pooling)
        if self.args.mode == 'train':
            print(f"train dataset size : {len(train_ds)}")
            print(f"test dataset size : {len(test_ds)}")
            return train_dl, test_dl
        else:
            print(f"test dataset size : {len(test_ds)}")
            return test_dl

    def segmentation_dataset(self):
        if "Eigen" in self.args.netvit.split("_"):
            eigen_pooling=(self.args.eigen_ratio, self.args.eigen_layer)
        else:
            eigen_pooling=None

        if self.args.mode == 'train':
            train_ds = SegVITDataset(self.args, set_type='train', is_train=True)
            train_dl = DataLoader(train_ds, batch_size=self.args.batch_size, shuffle=True, pin_memory=False,
                                  num_workers=self.args.num_workers, prefetch_factor=self.args.prefetch_factor, eigen_pooling=eigen_pooling)

            # Class_num is the ratio for solving the problem of data imbalance, which is used in the loss function.
            class_num_train = train_ds.class_num
            weight_train = class_num_train.max() / class_num_train

        test_ds = SegVITDataset(self.args, set_type='test', is_train=False)
        test_dl = DataLoader(test_ds, batch_size=self.args.batch_size, shuffle=False, pin_memory=False,
                             num_workers=self.args.num_workers, prefetch_factor=self.args.prefetch_factor, eigen_pooling=eigen_pooling)
        # Class_num
        class_num_test = test_ds.class_num
        weight_test = class_num_test.max() / class_num_test

        if self.args.mode == 'train':
            print(f"train dataset size : {len(train_ds)}")
            print(f"test dataset size : {len(test_ds)}")
            return train_dl, test_dl, weight_train, weight_test
        else:
            print(f"test dataset size : {len(test_ds)}")
            return test_dl, weight_test
