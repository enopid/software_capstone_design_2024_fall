import os
import argparse

import tqdm, json
import numpy as np


from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torchmetrics import Accuracy

import network.NetVIT_cls as netvit 
from network.NetVIT_cls import init_weights
from dataset.baseVIT_dataset import BaseDataset

from util.utils import same_seed
from util.utils import RunningAverage
from util.utils import save_logging

import matplotlib
#import torchprofile

# display error, if we use plt.savefig() in Linux。using matplotlib.use('Agg') to solve the problem.
matplotlib.use('Agg')


# os.environ["CUDA_VISIBLE_DEVICES"] = "4"


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(args, net, train_dl, criterion, writer, epoch):
    net.train()

    preds = []
    labels = []

    train_loss_avg = RunningAverage()
    train_acc_avg = RunningAverage()

    accuracy = Accuracy(task="multiclass", num_classes=args.num_classes, ignore_index=None).to(args.device)
    
    with tqdm.tqdm(total=len(train_dl)) as t:
        for i, batch in enumerate(train_dl):
            # load train data
            x=batch.x.to(args.device)
            edge_index=batch.edge_index.to(args.device)
            y=batch.y.to(args.device)
            batch=batch.batch.to(args.device)

            pred = net(x,edge_index, batch) # BXC
            pred_label = F.log_softmax(pred, dim=-1) 

            # computing loss
            optimizer.zero_grad()
            loss = criterion(pred, y)
            loss.backward() #여기서 SEGMENT FAULT
            optimizer.step()

            acc = accuracy(pred_label, y)

            preds.append(pred_label.cpu())
            labels.append(y.cpu())

            train_loss_avg.update(loss.cpu().item())
            train_acc_avg.update(acc.cpu().item())

            t.set_description(f"epoch - {epoch:4}")
            t.set_postfix(loss='{:05.4f}'.format(train_loss_avg()), acc='{:02.1f}%'.format(train_acc_avg()*100))
            t.update()

            torch.cuda.empty_cache()

    # tensorboard log: train
    writer.add_scalar('Accuracy/train', train_acc_avg(), epoch)
    writer.add_scalar('Loss/train', train_loss_avg(), epoch)

    return train_loss_avg(), train_acc_avg()


def test_voting(args, net, test_dl, criterion=None, writer=None, epoch=None, voting_num=10):
    net.eval()
    
    y_list=[]

    test_loss_avg = RunningAverage()
    test_acc_avg = RunningAverage()

    accuracy = Accuracy(task="multiclass", num_classes=args.num_classes, ignore_index=None).to(args.device)
    i=0
    print(f"test dataset size : {len(test_dl.dataset)}")
    pred_labels=[[] for i in range(voting_num)]
    with torch.no_grad():
        for i in range(voting_num):
            for  j, batch in enumerate(test_dl):
                x=batch.x.to(args.device)
                edge_index=batch.edge_index.to(args.device)
                y=batch.y.to(args.device)
                batch=batch.batch.to(args.device)
                
                pred = net(x,edge_index, batch) # BXC

                pred_label = F.log_softmax(pred, dim=-1) 

                pred_labels[i].append(pred_label)
                
                if i==0:
                    y_list.append(y)

                torch.cuda.empty_cache()
            print(f"voting {i+1} done")
        
        for i in range(len(y_list)):
            y=y_list[i]
            stack_tensor=torch.stack([pred[i] for pred in pred_labels])
            pred_label, _ = torch.mode(stack_tensor, dim=0)

            acc = accuracy(pred_label, y)
            test_acc_avg.update(acc.cpu().item())

    return test_acc_avg()

def test(args, net, test_dl, criterion=None, writer=None, epoch=None):
    net.eval()

    preds = []
    labels = []

    test_loss_avg = RunningAverage()
    test_acc_avg = RunningAverage()

    accuracy = Accuracy(task="multiclass", num_classes=args.num_classes, ignore_index=None).to(args.device)
    i=0
    print(f"test dataset size : {len(test_dl.dataset)}")
    with torch.no_grad():
        for batch in test_dl:
            # load train data
            x=batch.x.to(args.device)
            edge_index=batch.edge_index.to(args.device)
            y=batch.y.to(args.device)
            batch=batch.batch.to(args.device)
            
            if args.mode == 'train':
                pred = net(x,edge_index, batch) # BXC
            else:
                pred = net(x,edge_index, batch) # BXC

            pred_label = F.log_softmax(pred, dim=-1) 

            acc = accuracy(pred_label, y)

            if criterion is not None:
                loss = criterion(pred, y)
                test_loss_avg.update(loss.cpu().item())

            preds.append(pred_label.cpu())
            labels.append(y.cpu())

            test_acc_avg.update(acc.cpu().item())

            torch.cuda.empty_cache()

    if args.mode == 'train':
        # tensorboard log: test
        writer.add_scalar('Accuracy/val', test_acc_avg(), epoch)
        writer.add_scalar('Loss/val', test_loss_avg(), epoch)

    return test_loss_avg(), test_acc_avg()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'test'], default='train')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--seed', type=int, default=40938661)
    parser.add_argument('--prefetch_factor', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-3)
    parser.add_argument('--scheduler_mode', choices=['CosWarm', 'MultiStep'], default='CosWarm')
    parser.add_argument('--scheduler_T0', type=int, default=30)
    parser.add_argument('--scheduler_eta_min', type=float, default=3e-7)
    parser.add_argument('--weight_decay', type=float, default=0.3)
    parser.add_argument('--amsgrad', action='store_true')
    parser.add_argument('--loss_rate', type=float, default=1.8)
    parser.add_argument('--bandwidth', type=float, default=1.0)
    parser.add_argument('--num_classes', type=int, default=30)
    parser.add_argument('--num_inputs', nargs='+', default=[154, 64, 16], type=int, help='Multi-resolution input')
    parser.add_argument('--data_path', type=str, default='/local_datasets')
    parser.add_argument('--dataset', type=str, default='shrec16')
    parser.add_argument('--force_reload', action='store_true')
    parser.add_argument('--name', type=str, default='base')
    parser.add_argument('--netvit', type=str, default='Net_GCN_ASAP_GlobalPooling')
    parser.add_argument('--continue_training', action='store_true')
    parser.add_argument('--voting', action='store_true')
    parser.add_argument('--save_step', type=int, default=250)

    parser.add_argument('--eigen_ratio', type=float, default=0.5)
    parser.add_argument('--eigen_layer', type=int, default=3)
    
    parser.add_argument('--face_input', action='store_true')

    args = parser.parse_args()
    print(args)

    same_seed(args.seed)

    print('Load Dataset...')
    base_dataset = BaseDataset(args)
    if args.mode == 'train':
        if args.face_input:
            train_dl, test_dl = base_dataset.face_classification_dataset()
        else:
            train_dl, test_dl = base_dataset.classification_dataset()
    else:
        if args.face_input:
            test_dl = base_dataset.face_classification_dataset()
        else:
            test_dl = base_dataset.classification_dataset()

    # define the Net
    if args.netvit=='Net_GCN_ASAP_GlobalPooling':
        net = netvit.Net_GCN_ASAP_GlobalPooling(args.num_classes).to(args.device)
    elif args.netvit=='Net_GCN_ASAP_GlobalPooling_DO':
        net = netvit.Net_GCN_ASAP_GlobalPooling_DO(args.num_classes).to(args.device)
    elif args.netvit=='Net_GAT_ASAP_GlobalPooling':
        net = netvit.Net_GAT_ASAP_GlobalPooling(args.num_classes).to(args.device)
    elif args.netvit=='Net_GIN_ASAP_GlobalPooling':
        net = netvit.Net_GIN_ASAP_GlobalPooling(args.num_classes).to(args.device)
    elif args.netvit=='Net_GCN_WoP_GlobalPooling':
        net = netvit.Net_GCN_WoP_GlobalPooling(args.num_classes).to(args.device)
    elif args.netvit=='Net_GCN_WoP_WoEigen':
        net = netvit.Net_GCN_WoP_WoEigen(args.num_classes).to(args.device)
        
    elif args.netvit=='Net_GCN_Layer2_WoP_WoEigen':
        net = netvit.Net_GCN_Layer4_WoP_WoEigen(args.num_classes).to(args.device)
    elif args.netvit=='Net_GCN_Layer4_WoP_WoEigen':
        net = netvit.Net_GCN_Layer4_WoP_WoEigen(args.num_classes).to(args.device)
    elif args.netvit=='Net_GCN_Layer6_WoP_WoEigen':
        net = netvit.Net_GCN_Layer6_WoP_WoEigen(args.num_classes).to(args.device)
        
    elif args.netvit=='Net_GCN_Layer2_ASAP_GlobalPooling':
        net = netvit.Net_GCN_Layer2_ASAP_GlobalPooling(args.num_classes).to(args.device)
    elif args.netvit=='Net_GCN_Layer4_ASAP_GlobalPooling':
        net = netvit.Net_GCN_Layer4_ASAP_GlobalPooling(args.num_classes).to(args.device)
    elif args.netvit=='Net_GCN_Layer6_ASAP_GlobalPooling':
        net = netvit.Net_GCN_Layer6_ASAP_GlobalPooling(args.num_classes).to(args.device)

    elif args.netvit=='Net_GCN_Eigen_GlobalPooling':
        net = netvit.Net_GCN_Eigen_GlobalPooling(args.num_classes).to(args.device)
    elif args.netvit=='Net_GCN_Eigen_GlobalPooling_V2':
        net = netvit.Net_GCN_Eigen_GlobalPooling_V2(args.num_classes).to(args.device)

    elif args.netvit=='Net_GAT_Eigen_GlobalPooling':
        net = netvit.Net_GAT_Eigen_GlobalPooling(args.num_classes).to(args.device)
    elif args.netvit=='Net_GAT_Eigen_GlobalPooling_V2':
        net = netvit.Net_GAT_Eigen_GlobalPooling_V2(args.num_classes).to(args.device)
    elif args.netvit=='Net_GAT_WoP_GlobalPooling':
        net = netvit.Net_GAT_WoP_GlobalPooling(args.num_classes).to(args.device)

    elif args.netvit=='Net_GAT_Eigen_GlobalPooling_face_V1':
        net = netvit.Net_GAT_Eigen_GlobalPooling_face_V1(args.num_classes).to(args.device)
    
    # save the checkpoints
    if not os.path.exists(os.path.join('checkpoints', args.name)):
        os.makedirs(os.path.join('checkpoints', args.name))
        os.makedirs(os.path.join('checkpoints', args.name, args.netvit),exist_ok=True)

    # save the visualization result
    if args.mode == 'test' and (not os.path.exists(os.path.join('visualization_result', args.name))):
        os.makedirs(os.path.join('visualization_result', args.name))
        os.makedirs(os.path.join('visualization_result', args.name, args.netvit),exist_ok=True)


    if args.mode == 'train':
        # tensorboard
        writer = SummaryWriter(os.path.join('checkpoints', args.name, args.netvit, 'log_dir'))

        # Network initialization
        init_weights(net)

        optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=args.amsgrad)
        criterion = nn.CrossEntropyLoss().to(args.device)

        # select scheduler mode
        if args.scheduler_mode == 'CosWarm':
            scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.scheduler_T0, T_mult=2,
                                                                 eta_min=args.scheduler_eta_min, verbose=True)
        else:
            scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[200, 500, 1000], gamma=0.1, verbose=True)

        if args.continue_training:
            load_name="last"
            load_epoch,load_optimizer,load_scheduler=torch.load(os.path.join('checkpoints', args.name, args.netvit, f'{load_name}_TrainSetting.pth'))
            optimizer.load_state_dict(load_optimizer)
            scheduler.load_state_dict(load_scheduler)
            scheduler.last_epoch = load_epoch
            
            net.load_state_dict(torch.load(os.path.join('checkpoints', args.name, args.netvit, f'{load_name}.pth')))

            with open(os.path.join('checkpoints', args.name, args.netvit, 'best_acc.json'), 'r') as f:
                df = json.load(f)
                max_val_acc=df["Val"]["epoch_acc"]
            with open(os.path.join('checkpoints', args.name, args.netvit, 'best_loss.json'), 'r') as f:
                df = json.load(f)
                min_val_loss=df["Val"]["epoch_loss"]
        else:
            load_epoch=-1
            min_val_loss = np.inf
            max_val_acc = -np.inf

        for epoch in tqdm.trange(load_epoch+1,args.epochs):
            train_loss, train_acc = train(args, net, train_dl, criterion, writer, epoch)
            test_loss, test_acc = test(args, net, test_dl, criterion, writer, epoch)

            scheduler.step()
            writer.add_scalar('Utils/lr_scheduler', scheduler.get_last_lr()[0], global_step=epoch)

            save_logging(args, test_loss, test_acc, epoch, net, (epoch, optimizer, scheduler), train_loss, train_acc, save_name='last')
            # save best loss
            if min_val_loss > test_loss:
                min_val_loss = test_loss
                save_logging(args, test_loss, test_acc, epoch, net, (epoch, optimizer, scheduler), train_loss, train_acc, save_name='best_loss')
            # save best acc
            if max_val_acc < test_acc:
                max_val_acc = test_acc
                save_logging(args, test_loss, test_acc, epoch, net, (epoch, optimizer, scheduler), train_loss, train_acc, save_name='best_acc')
            if (epoch%args.save_step)==0:
                save_logging(args, test_loss, test_acc, epoch, net, (epoch, optimizer, scheduler), train_loss, train_acc, save_name=f'{epoch}')

        writer.close()
    else:
        net.load_state_dict(torch.load(os.path.join('checkpoints', args.name, args.netvit, 'best_acc.pth')))
        criterion = nn.CrossEntropyLoss().to(args.device)
        numparam=count_parameters(net)
        print(f"파라미터수 : {numparam}")

        if args.voting:
            test_acc = test_voting(args, net, test_dl, criterion)
            print(test_acc*100)
        else:
            test_loss, test_acc = test(args, net, test_dl, criterion)
            print(test_acc*100,test_loss)
            save_logging(args, test_loss, test_acc)
