import gc

import torch
import torch.nn.init as init
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as geo_nn
from torch_geometric.nn import global_mean_pool, GCNConv, ASAPooling, GATv2Conv, GINConv
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import scatter

import trimesh, os


def SetNeighborLoader(data):
    train_loader = NeighborLoader(
        data,
        num_neighbors=[10,10],
        batch_size=16,
        input_nodes=data.train_mask
    )
    return train_loader

class GIN(nn.Module):
    def __init__(self, in_channels, out_channels,
                 n_layers=2):
        super(GIN, self).__init__()

        self.conv = GINConv(
            nn.Sequential(nn.Linear(in_channels, out_channels), nn.BatchNorm1d(out_channels), nn.ReLU(),
                       nn.Linear(out_channels, out_channels), nn.ReLU())
                       )

    def forward(self, x, edge_index, mask=None):
        
        x = self.conv(x, edge_index)
 
        return x

class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 n_layers=1, heads=3):
        super(GAT, self).__init__()

        self.convs = nn.ModuleList()
        self.convs.append(GATv2Conv(in_channels, hidden_channels, heads=heads))
        for i in range(n_layers-1):
            self.convs.append(GATv2Conv(hidden_channels*heads, hidden_channels, heads=heads))
        self.convs.append(GATv2Conv(hidden_channels*heads, out_channels, heads=1))

        self.bns = nn.ModuleList()
        for i in range(n_layers):
            self.bns.append(geo_nn.BatchNorm(hidden_channels*heads))
        self.bns.append(geo_nn.BatchNorm(out_channels))

    def forward(self, x, edge_index, mask=None):
        for layer_idx in range(len(self.convs)):
            x = self.bns[layer_idx](F.relu(self.convs[layer_idx](x, edge_index, mask)))

        return x

class GNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 n_layers, normalize=False):
        super(GNN, self).__init__()
        
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, normalize))
        for i in range(n_layers-1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels, normalize))
        self.convs.append(GCNConv(hidden_channels, out_channels, normalize))

        self.bns = nn.ModuleList()
        for i in range(n_layers):
            self.bns.append(geo_nn.BatchNorm(hidden_channels))
        self.bns.append(geo_nn.BatchNorm(out_channels))



    def forward(self, x, edge_index, mask=None):
        for layer_idx in range(len(self.convs)):
            x = self.bns[layer_idx](F.relu(self.convs[layer_idx](x, edge_index, mask)))

        return x

class Eigen_Pooling(nn.Module):
    def __init__(self):
        super(Eigen_Pooling, self).__init__()
    
    def forward(self, x : torch.Tensor, cluster : torch.Tensor, edge_index : torch.Tensor, batch : torch.Tensor, reduce="max"):
        max_cluster=torch.max(cluster[:,0])+1
        cluster_index=torch.squeeze(cluster[:,0],0)+batch*max_cluster
        _, inverse_index=torch.unique(cluster_index, sorted=True, return_inverse=True)
        
        batch=scatter(batch,dim=0,index=inverse_index,reduce="max")

        x=scatter(x,dim=0,index=inverse_index,reduce=reduce)
        
        edge_index=inverse_index[edge_index]
        edge_index = torch.unique(edge_index, dim=1)  # 중복 제거

        mask = edge_index[0] != edge_index[1]
        edge_index = edge_index[:, mask]


        cluster=scatter(cluster[:,1:],dim=0,index=inverse_index,reduce="max")

        return x, cluster, edge_index, batch

class ASAP_Pooling(nn.Module):
    def __init__(self, input_dim, ratio):
        super(ASAP_Pooling, self).__init__()
        self.ASAP_Pooling=ASAPooling(input_dim, ratio)
    
    def forward(self, x, edge_index, batch):
        x, edge_index, _, batch, topk=self.ASAP_Pooling(x=x,edge_index=edge_index,batch=batch)
        return x, edge_index, batch, topk

class Net_GIN_ASAP_GlobalPooling(nn.Module):
    def __init__(self, num_class):
        super(Net_GIN_ASAP_GlobalPooling, self).__init__()
        self.num_class = num_class

        self.baseblock1=self.BaseBlock(39, 128, 0.7)
        self.baseblock2=self.BaseBlock(128, 128, 0.7)
        self.baseblock3=self.BaseBlock(128, 256, 0.7)


        self.fc = nn.Sequential(
            nn.Linear((128+128+256), 256),
            nn.Linear(256, self.num_class)
        )

    def forward(self, x, edge_index, batch):
        x1, edge_index, batch1=self.baseblock1(x, edge_index, batch)
        x2, edge_index, batch2=self.baseblock2(x1, edge_index, batch1)
        x3, edge_index, batch3=self.baseblock3(x2, edge_index, batch2)
        
        x1=geo_nn.global_add_pool(x1,batch1)
        x2=geo_nn.global_add_pool(x2,batch2)
        x3=geo_nn.global_add_pool(x3,batch3)

        x = torch.cat((x1, x2, x3), dim=1)
        x=self.fc(x)

        return x
    
    class BaseBlock(nn.Module):
        def __init__(self, in_channels, out_channels, ratio, n_layers=2):
            super().__init__()
            self.pooling = ASAP_Pooling(out_channels,ratio)
            self.gnn = GIN(in_channels, out_channels, n_layers)

        def forward(self, x, edge_index, batch):
            x = self.gnn(x, edge_index)
            x, edge_index, batch, topk = self.pooling(x, edge_index, batch)
            return x, edge_index, batch 

class Net_GAT_ASAP_GlobalPooling(nn.Module):
    def __init__(self, num_class):
        super(Net_GAT_ASAP_GlobalPooling, self).__init__()
        self.num_class = num_class

        self.baseblock1=self.BaseBlock(39, 64, 128, 0.7)
        self.baseblock2=self.BaseBlock(128, 128, 128, 0.7)
        self.baseblock3=self.BaseBlock(128, 256, 256, 0.7)


        self.fc = nn.Sequential(
            nn.Linear(256, 256),
            nn.Linear(256, self.num_class)
        )

    def forward(self, x, edge_index, batch):
        x, edge_index, batch=self.baseblock1(x, edge_index, batch)
        x, edge_index, batch=self.baseblock2(x, edge_index, batch)
        x, edge_index, batch=self.baseblock3(x, edge_index, batch)
        x=global_mean_pool(x,batch)
        x=self.fc(x)

        return x
    
    class BaseBlock(nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels, ratio, n_layers=2):
            super().__init__()
            self.pooling = ASAP_Pooling(out_channels,ratio)
            self.gnn = GAT(in_channels, hidden_channels, out_channels, n_layers)

        def forward(self, x, edge_index, batch):
            x = self.gnn(x, edge_index)
            x, edge_index, batch, topk = self.pooling(x, edge_index, batch)
            return x, edge_index, batch 

class Net_GCN_ASAP_GlobalPooling(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.num_class = num_class

        self.baseblock1=self.BaseBlock((39), 64, 128, 0.7)
        self.baseblock2=self.BaseBlock(128, 128, 128, 0.7)
        self.baseblock3=self.BaseBlock(128, 256, 256, 0.7)


        self.fc = nn.Sequential(
            nn.Linear(256, 256),
            nn.Linear(256, self.num_class)
        )

    def forward(self, x, edge_index0, batch0, save_visualiztion_path=None):
        posx0=x[:,:3]
        #wo_test (pos:3 normal:3 / eigen:20 / curv:1 / angle:3 / hks:9)
        indices = torch.arange(39)
        mask = (indices < 6) & (indices >= 26)

        x, edge_index1, batch1, topk=self.baseblock1(x, edge_index0, batch0)
        posx1=posx0[topk]

        x, edge_index2, batch2, topk=self.baseblock2(x, edge_index1, batch1)
        posx2=posx1[topk]

        x, edge_index3, batch3, topk=self.baseblock3(x, edge_index2, batch2)
        posx3=posx2[topk]

        x=global_mean_pool(x,batch3)
        x=self.fc(x)

        if save_visualiztion_path:
            save_visualization(posx0,edge_index0,batch0,save_visualiztion_path,0)
            save_visualization(posx1,edge_index1,batch1,save_visualiztion_path,1)
            save_visualization(posx2,edge_index2,batch2,save_visualiztion_path,2)
            save_visualization(posx3,edge_index3,batch3,save_visualiztion_path,3)

        return x
    
    class BaseBlock(nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels, ratio, n_layers=2):
            super().__init__()
            self.pooling = ASAP_Pooling(out_channels,ratio)
            self.gnn = GNN(in_channels, hidden_channels, out_channels, n_layers)

        def forward(self, x, edge_index, batch):
            x = self.gnn(x, edge_index)
            x, edge_index, batch, topk = self.pooling(x, edge_index, batch)
            return x, edge_index, batch, topk 
        
class Net_GCN_Layer6_ASAP_GlobalPooling(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.num_class = num_class

        self.baseblock1=self.BaseBlock((39), 64, 128, 0.7)
        self.baseblock2=self.BaseBlock(128, 128, 128, 0.7)
        self.baseblock3=self.BaseBlock(128, 256, 256, 0.7)


        self.fc = nn.Sequential(
            nn.Linear(256, 256),
            nn.Linear(256, self.num_class)
        )

    def forward(self, x, edge_index0, batch0, save_visualiztion_path=None):
        posx0=x[:,:3]
        #wo_test (pos:3 normal:3 / eigen:20 / curv:1 / angle:3 / hks:9)
        indices = torch.arange(39)
        mask = (indices < 6) & (indices >= 26)

        x, edge_index1, batch1, topk=self.baseblock1(x, edge_index0, batch0)
        posx1=posx0[topk]

        x, edge_index2, batch2, topk=self.baseblock2(x, edge_index1, batch1)
        posx2=posx1[topk]

        x, edge_index3, batch3, topk=self.baseblock3(x, edge_index2, batch2)
        posx3=posx2[topk]

        x=global_mean_pool(x,batch3)
        x=self.fc(x)

        if save_visualiztion_path:
            save_visualization(posx0,edge_index0,batch0,save_visualiztion_path,0)
            save_visualization(posx1,edge_index1,batch1,save_visualiztion_path,1)
            save_visualization(posx2,edge_index2,batch2,save_visualiztion_path,2)
            save_visualization(posx3,edge_index3,batch3,save_visualiztion_path,3)

        return x
    
    class BaseBlock(nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels, ratio, n_layers=1):
            super().__init__()
            self.pooling = ASAP_Pooling(out_channels,ratio)
            self.gnn = GNN(in_channels, hidden_channels, out_channels, n_layers)

        def forward(self, x, edge_index, batch):
            x = self.gnn(x, edge_index)
            x, edge_index, batch, topk = self.pooling(x, edge_index, batch)
            return x, edge_index, batch, topk 
     
class Net_GCN_ASAP_GlobalPooling_DO(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.num_class = num_class

        self.baseblock1=self.BaseBlock(39, 64, 128, 0.5)
        self.baseblock2=self.BaseBlock(128, 128, 128, 0.5)
        self.baseblock3=self.BaseBlock(128, 256, 256, 0.5)
        self.baseblock4=GNN(256, 256, 256, 2)

        self.fc = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_class)
        )

    def forward(self, x, edge_index0, batch0, save_visualiztion_path=None):
        posx0=x[:,:3]

        x, edge_index1, batch1, topk=self.baseblock1(x, edge_index0, batch0)
        posx1=posx0[topk]

        x, edge_index2, batch2, topk=self.baseblock2(x, edge_index1, batch1)
        posx2=posx1[topk]

        x, edge_index3, batch3, topk=self.baseblock3(x, edge_index2, batch2)
        posx3=posx2[topk]

        x, edge_index3=self.baseblock4(x, edge_index3)

        x=global_mean_pool(x,batch3)
        x=self.fc(x)

        if save_visualiztion_path:
            save_visualization(posx0,edge_index0,batch0,save_visualiztion_path,0)
            save_visualization(posx1,edge_index1,batch1,save_visualiztion_path,1)
            save_visualization(posx2,edge_index2,batch2,save_visualiztion_path,2)
            save_visualization(posx3,edge_index3,batch3,save_visualiztion_path,3)

        return x
    
    class BaseBlock(nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels, ratio, n_layers=2):
            super().__init__()
            self.pooling = ASAP_Pooling(out_channels,ratio)
            self.gnn = GNN(in_channels, hidden_channels, out_channels, n_layers)
            self.dropout=nn.Dropout(p=0.3)

        def forward(self, x, edge_index, batch):
            x = self.gnn(x, edge_index)
            x=self.dropout(x)
            x, edge_index, batch, topk = self.pooling(x, edge_index, batch)
            return x, edge_index, batch, topk 
        
class Net_GCN_Eigen_GlobalPooling_V2(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.num_class = num_class

        self.baseblock0=GNN(39, 64, 64, 2)
        self.baseblock1=self.BaseBlock(64, 128, 128)
        self.baseblock2=self.BaseBlock(128, 128, 128)
        self.baseblock3=self.BaseBlock(128, 256, 256)

        self.fc = nn.Sequential(
            nn.Linear((64+128+128+256), 256),
            nn.ReLU(),
            nn.Linear(256, self.num_class)
        )

    def forward(self, x, edge_index, batch, save_visualiztion_path=None):
        cluster=x[:,39:].long()
        x=x[:,:39]
        
        x0=self.baseblock0(x, edge_index)
        x1, cluster, edge_index, batch1=self.baseblock1(x0, cluster, edge_index, batch)
        x2, cluster, edge_index, batch2=self.baseblock2(x1, cluster, edge_index, batch1)
        x3, cluster, edge_index, batch3=self.baseblock3(x2, cluster, edge_index, batch2)
        
        x0=global_mean_pool(x0,batch)
        x1=global_mean_pool(x1,batch1)
        x2=global_mean_pool(x2,batch2)
        x3=global_mean_pool(x3,batch3)
        
        x = torch.cat((x0, x1, x2, x3), dim=1)
        x=self.fc(x)

        return x
    
    class BaseBlock(nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels, n_layers=2):
            super().__init__()
            self.pooling = Eigen_Pooling()
            self.gnn = GNN(in_channels, hidden_channels, out_channels, n_layers)

        def forward(self, x, cluster, edge_index, batch):
            x, cluster, edge_index, batch = self.pooling(x, cluster,  edge_index, batch)
            x = self.gnn(x, edge_index)
            return x, cluster, edge_index, batch

class Net_GCN_Eigen_GlobalPooling(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.num_class = num_class

        self.baseblock1=self.BaseBlock((39), 64, 128)
        self.baseblock2=self.BaseBlock(128, 128, 128)
        self.baseblock3=self.BaseBlock(128, 256, 256)
        self.baseblock4=GNN(256, 256, 256, 2)

        self.fc = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_class)
        )

    def forward(self, x, edge_index, batch, save_visualiztion_path=None):
        cluster=x[:,39:].long()
        x=x[:,:39]

        x, cluster, edge_index, batch=self.baseblock1(x, cluster, edge_index, batch)
        x, cluster, edge_index, batch=self.baseblock2(x, cluster, edge_index, batch)
        x, cluster, edge_index, batch=self.baseblock3(x, cluster, edge_index, batch)
        x=self.baseblock4(x, edge_index)

        x=global_mean_pool(x,batch)
        x=self.fc(x)

        return x
    
    class BaseBlock(nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels, n_layers=2):
            super().__init__()
            self.pooling = Eigen_Pooling()
            self.gnn = GNN(in_channels, hidden_channels, out_channels, n_layers)

        def forward(self, x, cluster, edge_index, batch):
            x = self.gnn(x, edge_index)
            x, cluster, edge_index, batch = self.pooling(x, cluster,  edge_index, batch)
            return x, cluster, edge_index, batch

class Net_GAT_Eigen_GlobalPooling(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.num_class = num_class

        self.baseblock1=self.BaseBlock(39, 64, 128)
        self.baseblock2=self.BaseBlock(128, 128, 128)
        self.baseblock3=self.BaseBlock(128, 256, 256)
        self.baseblock4=GAT(256, 256, 256, 2)

        self.fc = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_class)
        )

    def forward(self, x, edge_index, batch, save_visualiztion_path=None):
        cluster=x[:,39:].long()
        x=x[:,:39]

        x, cluster, edge_index, batch=self.baseblock1(x, cluster, edge_index, batch)
        x, cluster, edge_index, batch=self.baseblock2(x, cluster, edge_index, batch)
        x, cluster, edge_index, batch=self.baseblock3(x, cluster, edge_index, batch)
        x=self.baseblock4(x, edge_index)

        x=global_mean_pool(x,batch)
        x=self.fc(x)

        return x
    
    class BaseBlock(nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels, n_layers=2):
            super().__init__()
            self.pooling = Eigen_Pooling()
            self.gnn = GAT(in_channels, hidden_channels, out_channels, n_layers)

        def forward(self, x, cluster, edge_index, batch):
            x = self.gnn(x, edge_index)
            x, cluster, edge_index, batch = self.pooling(x, cluster,  edge_index, batch)
            return x, cluster, edge_index, batch


class Net_GAT_Eigen_GlobalPooling_V2(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.num_class = num_class

        self.baseblock1=self.BaseBlock(39, 64, 128)
        self.baseblock2=self.BaseBlock(128, 128, 128)
        self.baseblock3=self.BaseBlock(128, 256, 256)
        self.baseblock4=GAT(256, 256, 256, 2)

        self.fc = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_class)
        )

    def forward(self, x, edge_index, batch, save_visualiztion_path=None):
        cluster=x[:,39:].long()
        x=x[:,:39]

        x, cluster, edge_index, batch=self.baseblock1(x, cluster, edge_index, batch)
        x, cluster, edge_index, batch=self.baseblock2(x, cluster, edge_index, batch)
        x, cluster, edge_index, batch=self.baseblock3(x, cluster, edge_index, batch)
        x=self.baseblock4(x, edge_index)

        x=global_mean_pool(x,batch)
        x=self.fc(x)

        return x
    
    class BaseBlock(nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels, n_layers=2):
            super().__init__()
            self.pooling = Eigen_Pooling()
            self.gnn = GAT(in_channels, hidden_channels, out_channels, n_layers, heads=8)

        def forward(self, x, cluster, edge_index, batch):
            x = self.gnn(x, edge_index)
            x, cluster, edge_index, batch = self.pooling(x, cluster,  edge_index, batch)
            return x, cluster, edge_index, batch


class Net_GAT_Eigen_GlobalPooling_V3(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.num_class = num_class

        self.baseblock1=self.BaseBlock(39, 64, 128)
        self.baseblock2=self.BaseBlock(128, 128, 128)
        self.baseblock3=self.BaseBlock(128, 256, 256)
        self.baseblock4=GAT(256, 256, 256, 2, heads=8)

        self.fc = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_class)
        )

    def forward(self, x, edge_index, batch, save_visualiztion_path=None):
        cluster=x[:,39:].long()
        x=x[:,:39]

        x, cluster, edge_index, batch=self.baseblock1(x, cluster, edge_index, batch)
        x, cluster, edge_index, batch=self.baseblock2(x, cluster, edge_index, batch)
        x, cluster, edge_index, batch=self.baseblock3(x, cluster, edge_index, batch)
        x=self.baseblock4(x, edge_index)

        x=global_mean_pool(x,batch)
        x=self.fc(x)

        return x
    
    class BaseBlock(nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels, n_layers=2):
            super().__init__()
            self.pooling = Eigen_Pooling()
            self.gnn = GAT(in_channels, hidden_channels, out_channels, n_layers, heads=8)

        def forward(self, x, cluster, edge_index, batch):
            x = self.gnn(x, edge_index)
            x, cluster, edge_index, batch = self.pooling(x, cluster,  edge_index, batch)
            return x, cluster, edge_index, batch
        
class Net_GAT_Eigen_GlobalPooling_face_V1(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.num_class = num_class

        self.baseblock1=self.BaseBlock(13, 64, 128)
        self.baseblock2=self.BaseBlock(128, 128, 128)
        self.baseblock3=self.BaseBlock(128, 256, 256)
        self.baseblock4=GAT(256, 256, 256, 2)

        self.fc = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_class)
        )

    def forward(self, x, edge_index, batch, save_visualiztion_path=None):
        cluster=x[:,13:].long()
        x=x[:,:13]

        x, cluster, edge_index, batch=self.baseblock1(x, cluster, edge_index, batch)
        x, cluster, edge_index, batch=self.baseblock2(x, cluster, edge_index, batch)
        x, cluster, edge_index, batch=self.baseblock3(x, cluster, edge_index, batch)
        x=self.baseblock4(x, edge_index)

        x=global_mean_pool(x,batch)
        x=self.fc(x)

        return x
    
    class BaseBlock(nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels, n_layers=2):
            super().__init__()
            self.pooling = Eigen_Pooling()
            self.gnn = GAT(in_channels, hidden_channels, out_channels, n_layers, heads=8)

        def forward(self, x, cluster, edge_index, batch):
            x = self.gnn(x, edge_index)
            x, cluster, edge_index, batch = self.pooling(x, cluster,  edge_index, batch)
            return x, cluster, edge_index, batch


class Net_GCN_WoP_GlobalPooling(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.num_class = num_class

        self.baseblock1=self.BaseBlock(39, 64, 128, 0.7)
        self.baseblock2=self.BaseBlock(128, 128, 128, 0.7)
        self.baseblock3=self.BaseBlock(128, 256, 256, 0.7)


        self.fc = nn.Sequential(
            nn.Linear(256, 256),
            nn.Linear(256, self.num_class)
        )

    def forward(self, x, edge_index0, batch0, save_visualiztion_path=None):
        x, edge_index1, batch1=self.baseblock1(x, edge_index0, batch0)
        x, edge_index2, batch2=self.baseblock2(x, edge_index1, batch1)
        x, edge_index3, batch3=self.baseblock3(x, edge_index2, batch2)

        x=global_mean_pool(x,batch3)
        x=self.fc(x)

        return x
    
    class BaseBlock(nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels, ratio, n_layers=2):
            super().__init__()
            self.gnn = GNN(in_channels, hidden_channels, out_channels, n_layers)

        def forward(self, x, edge_index, batch):
            x = self.gnn(x, edge_index)
            return x, edge_index, batch 

class Net_GAT_WoP_GlobalPooling(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.num_class = num_class

        self.baseblock1=self.BaseBlock(39, 64, 128, 0.7)
        self.baseblock2=self.BaseBlock(128, 128, 128, 0.7)
        self.baseblock3=self.BaseBlock(128, 256, 256, 0.7)


        self.fc = nn.Sequential(
            nn.Linear(256, 256),
            nn.Linear(256, self.num_class)
        )

    def forward(self, x, edge_index0, batch0, save_visualiztion_path=None):
        x, edge_index1, batch1=self.baseblock1(x, edge_index0, batch0)
        x, edge_index2, batch2=self.baseblock2(x, edge_index1, batch1)
        x, edge_index3, batch3=self.baseblock3(x, edge_index2, batch2)

        x=global_mean_pool(x,batch3)
        x=self.fc(x)

        return x
    
    class BaseBlock(nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels, ratio, n_layers=2):
            super().__init__()
            self.gnn = GAT(in_channels, hidden_channels, out_channels, n_layers)

        def forward(self, x, edge_index, batch):
            x = self.gnn(x, edge_index)
            return x, edge_index, batch 

class Net_GCN_WoP_WoEigen(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.num_class = num_class

        self.baseblock1=self.BaseBlock(19, 64, 128, 0.7)
        self.baseblock2=self.BaseBlock(128, 128, 128, 0.7)
        self.baseblock3=self.BaseBlock(128, 256, 256, 0.7)

        self.fc = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_class)
        )

    def forward(self, x, edge_index0, batch0, save_visualiztion_path=None):
        indices = torch.arange(39)
        mask = (indices < 6) | (indices >= 26)
        x=x[:,mask]
        x, edge_index1, batch1=self.baseblock1(x, edge_index0, batch0)
        x, edge_index2, batch2=self.baseblock2(x, edge_index1, batch1)
        x, edge_index3, batch3=self.baseblock3(x, edge_index2, batch2)

        x=global_mean_pool(x,batch3)
        x=self.fc(x)

        return x
    
    class BaseBlock(nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels, ratio, n_layers=2):
            super().__init__()
            self.gnn = GNN(in_channels, hidden_channels, out_channels, n_layers)

        def forward(self, x, edge_index, batch):
            x = self.gnn(x, edge_index)
            return x, edge_index, batch 

class Net_GCN_Layer6_WoP_WoEigen(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.num_class = num_class

        self.baseblock1=self.BaseBlock(19, 64, 128, 0.7)
        self.baseblock2=self.BaseBlock(128, 128, 128, 0.7)
        self.baseblock3=self.BaseBlock(128, 256, 256, 0.7)

        self.fc = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_class)
        )

    def forward(self, x, edge_index0, batch0, save_visualiztion_path=None):
        indices = torch.arange(39)
        mask = (indices < 6) | (indices >= 26)
        x=x[:,mask]
        x, edge_index1, batch1=self.baseblock1(x, edge_index0, batch0)
        x, edge_index2, batch2=self.baseblock2(x, edge_index1, batch1)
        x, edge_index3, batch3=self.baseblock3(x, edge_index2, batch2)

        x=global_mean_pool(x,batch3)
        x=self.fc(x)

        return x
    
    class BaseBlock(nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels, ratio, n_layers=1):
            super().__init__()
            self.gnn = GNN(in_channels, hidden_channels, out_channels, n_layers)

        def forward(self, x, edge_index, batch):
            x = self.gnn(x, edge_index)
            return x, edge_index, batch 
        
class Net_GCN_Layer2_WoP_WoEigen(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.num_class = num_class

        self.baseblock1=self.BaseBlock(19, 128, 256, 0.7)

        self.fc = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_class)
        )

    def forward(self, x, edge_index0, batch0, save_visualiztion_path=None):
        indices = torch.arange(39)
        mask = (indices < 6) | (indices >= 26)
        x=x[:,mask]
        x, edge_index1, batch1=self.baseblock1(x, edge_index0, batch0)

        x=global_mean_pool(x,batch1)
        x=self.fc(x)

        return x
    
    class BaseBlock(nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels, ratio, n_layers=1):
            super().__init__()
            self.gnn = GNN(in_channels, hidden_channels, out_channels, n_layers)

        def forward(self, x, edge_index, batch):
            x = self.gnn(x, edge_index)
            return x, edge_index, batch 

class Net_GCN_Layer4_WoP_WoEigen(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.num_class = num_class

        self.baseblock1=self.BaseBlock(19, 64, 128, 0.7)
        self.baseblock2=self.BaseBlock(128, 128, 256, 0.7)
        self.baseblock3=self.BaseBlock(128, 256, 256, 0.7)

        self.fc = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_class)
        )

    def forward(self, x, edge_index0, batch0, save_visualiztion_path=None):
        indices = torch.arange(39)
        mask = (indices < 6) | (indices >= 26)
        x=x[:,mask]
        x, edge_index1, batch1=self.baseblock1(x, edge_index0, batch0)
        x, edge_index2, batch2=self.baseblock2(x, edge_index1, batch1)

        x=global_mean_pool(x,batch2)
        x=self.fc(x)

        return x
    
    class BaseBlock(nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels, ratio, n_layers=1):
            super().__init__()
            self.gnn = GNN(in_channels, hidden_channels, out_channels, n_layers)

        def forward(self, x, edge_index, batch):
            x = self.gnn(x, edge_index)
            return x, edge_index, batch 

def save_visualization(pos, edge_index, batch, path, level):
    os.makedirs(path,exist_ok=True)

    num_graphs = batch.max().item() + 1  
    pred_size=0
    for i in range(num_graphs):
        graph_mask = (batch == i)  # 해당 그래프에 속하는 노드들 선택
        graph_x = pos[graph_mask]  # 해당 그래프의 노드 피처 추출
        

        edge_mask = (graph_mask[edge_index[0]] & graph_mask[edge_index[1]])
        graph_edge_index = edge_index[:, edge_mask]

        save_wireframe_as_obj(graph_x,graph_edge_index,pred_size,os.path.join(path,f'wireframe_{i:3<0}_{level}.obj'))
        pred_size+=torch.sum(graph_mask)

def save_wireframe_as_obj(vertices, edges, pred_size, filename):
    with open(filename, 'w') as f:
        for vertex in vertices:
            f.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
        
        for i in range(len(edges[0])):
            f.write(f"l {edges[0][i]-pred_size + 1} {edges[1][i]-pred_size + 1}\n")

def init_weights(net, init_type='kaiming', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm1d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)

def Eigen_pooling_test():
    testpooling=Eigen_Pooling()
    
    x=[[2,3,4],[4,3,2],[3,5,6],[3,6,8],[7,5,3],[3,6,8],[7,5,3],[2,3,4],[4,3,2],[3,5,6],[3,6,8],[7,5,3]]
    batch=[0,0,0,0,0,1,1,1,1,2,2,2]
    cluster=[[0,0,1,1,1,0,0,1,1,0,1,2],[0,0,0,0,0,0,0,1,1,0,0,1]]
    edge_index=[[0,0,0,2,3,5,7,9,10],
                [1,2,3,3,4,6,8,10,11]]

    x=torch.tensor(x, requires_grad=True,dtype=torch.float64)
    cluster=torch.tensor(cluster,requires_grad=True,dtype=torch.long)
    cluster=cluster.transpose(0,1)
    edge_index=torch.tensor(edge_index,requires_grad=True,dtype=torch.long)
    batch=torch.tensor(batch,requires_grad=True,dtype=torch.long)

    x, cluster, edge_index, batch=testpooling(x,cluster,edge_index,batch)
    print(x)
    print(cluster)
    print(edge_index)
    print(batch)

    is_correct = torch.autograd.gradcheck(testpooling, (x,cluster,edge_index,batch))
    print("Gradient check passed:", is_correct)

def Eigen_Network_test():
    testnet=Net_GCN_Eigen_GlobalPooling(22)
    
    x=[[2,3,4],[4,3,2],[3,5,6],[3,6,8],[7,5,3],[3,6,8],[7,5,3],[2,3,4],[4,3,2],[3,5,6],[3,6,8],[7,5,3]]
    batch=[0,0,0,0,0,1,1,1,1,2,2,2]
    cluster=[[0,0,1,1,1,0,0,1,1,0,1,2],[0,0,0,0,0,0,0,1,1,0,0,1],[0,0,0,0,0,0,0,0,0,0,0,0]]
    edge_index=[[0,0,0,2,3,5,7,9,10],
                [1,2,3,3,4,6,8,10,11]]

    x=torch.tensor(x,dtype=torch.float)
    x=x.repeat(1,13)
    cluster=torch.tensor(cluster,dtype=torch.long)
    cluster=cluster.transpose(0,1)
    cluster.float()
    x=torch.cat((x,cluster),dim=1)
    edge_index=torch.tensor(edge_index,dtype=torch.long)
    batch=torch.tensor(batch,dtype=torch.long)

    x=testnet(x,cluster,edge_index,batch)
    print(x)

if __name__=="__main__":
    Eigen_pooling_test()