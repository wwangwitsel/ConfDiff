import sys
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import math
from scipy.io import arff
import pandas as pd
from scipy.io import loadmat
import scipy.io as sio
from scipy import stats
from torch.utils.data import Dataset


def generate_binary_pretrain_data(uci, ds):
    if uci == 0: # image datasets: mnist, fashion, kmnist...
        if ds == 'mnist':
            train_data, train_labels, test_data, test_labels, dim = prepare_mnist_data()
        elif ds == 'kmnist':
            train_data, train_labels, test_data, test_labels, dim = prepare_kmnist_data()   
        elif ds == 'fashion':
            train_data, train_labels, test_data, test_labels, dim = prepare_fashion_data()
        elif ds == 'cifar10':
            train_data, train_labels, test_data, test_labels, dim = prepare_cifar10_data()
        elif ds == 'cifar100':
            train_data, train_labels, test_data, test_labels, dim = prepare_cifar100_data()
        print("#original train:", train_data.shape, "#original test", test_data.shape)    
        positive_pretrain_data, negative_pretrain_data, positive_test_data, negative_test_data = convert_to_binary_data(ds, train_data, train_labels, test_data, test_labels)
        positive_pretrain_label = torch.ones(positive_pretrain_data.shape[0])
        negative_pretrain_label = -torch.ones(negative_pretrain_data.shape[0])
        positive_test_label = torch.ones(positive_test_data.shape[0])
        negative_test_label = -torch.ones(negative_test_data.shape[0])
        print("#all pretrain positive:", positive_pretrain_data.shape, "#all pretrain negative:", negative_pretrain_data.shape)
        print("#all test positive:", positive_test_data.shape, "#all test negative:", negative_test_data.shape)
        #pretrain_data = torch.cat((positive_pretrain_data, negative_pretrain_data), dim=0)
        #pretrain_label = torch.cat((positive_pretrain_label, negative_pretrain_label), dim=0)
    elif uci == 1:  #upload uci multi-class datasets (.mat, .arff): usps, pendigits,opdigits,letter...

        positive_pretrain_data, negative_pretrain_data, positive_test_data, negative_test_data, num_train, num_test, dim= prepare_uci_data(ds)
        positive_pretrain_label = torch.ones(positive_pretrain_data.shape[0])
        negative_pretrain_label = -torch.ones(negative_pretrain_data.shape[0])
        positive_test_label = torch.ones(positive_test_data.shape[0])
        negative_test_label = -torch.ones(negative_test_data.shape[0])
        print("#original train:", num_train, "#original test", num_test)    
        print("#all pretrain positive:", positive_pretrain_data.shape, "#all pretrain negative:", negative_pretrain_data.shape)
        print("#all test positive:", positive_test_data.shape, "#all test negative:", negative_test_data.shape)

    return positive_pretrain_data, negative_pretrain_data, positive_pretrain_label, negative_pretrain_label, positive_test_data, negative_test_data, positive_test_label, negative_test_label, dim

def generate_pretrain_loaders(positive_pretrain_data, negative_pretrain_data, positive_pretrain_label, negative_pretrain_label, positive_test_data, negative_test_data, positive_test_label, negative_test_label, batch_size):
    pretrain_data = torch.cat((positive_pretrain_data, negative_pretrain_data), dim=0)
    pretrain_label = torch.cat((positive_pretrain_label, negative_pretrain_label), dim=0)
    pretrain_new_idx = torch.randperm(pretrain_data.shape[0])
    pretrain_data = pretrain_data[pretrain_new_idx]
    pretrain_label = pretrain_label[pretrain_new_idx]
    test_data = torch.cat((positive_test_data, negative_test_data), dim=0)
    test_label = torch.cat((positive_test_label, negative_test_label), dim=0)
    test_new_idx = torch.randperm(test_data.shape[0])
    test_data = test_data[test_new_idx]
    test_label = test_label[test_new_idx]
    
    pretrain_set = torch.utils.data.TensorDataset(pretrain_data, pretrain_label)
    test_set = torch.utils.data.TensorDataset(test_data, test_label)
    pretrain_loader = torch.utils.data.DataLoader(dataset=pretrain_set, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
    train_eval_loader = torch.utils.data.DataLoader(dataset=pretrain_set, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
    return pretrain_loader, test_loader, train_eval_loader, pretrain_data, pretrain_label

def train_test_data_gen(positive_pretrain_data, negative_pretrain_data, positive_pretrain_test_data, negative_pretrain_test_data, n, prior, batch_size):
    train_pos_num = int(2 * n * prior)
    train_neg_num = 2 * n - train_pos_num
    train_pos_idx = torch.randperm(positive_pretrain_data.shape[0])[:train_pos_num]
    train_neg_idx = torch.randperm(negative_pretrain_data.shape[0])[:train_neg_num]
    
    train_pos_data = positive_pretrain_data[train_pos_idx]
    train_pos_label = torch.ones(train_pos_num)
    train_neg_data = negative_pretrain_data[train_neg_idx]
    train_neg_label = -torch.ones(train_neg_num)
    all_train_data = torch.cat((train_pos_data, train_neg_data), dim=0)
    all_train_label = torch.cat((train_pos_label, train_neg_label), dim=0)
    
    all_data_idx = torch.randperm(2 * n)
    data1_idx = all_data_idx[:n]
    data2_idx = all_data_idx[n:]
    train_data1 = all_train_data[data1_idx, :]
    train_label1 = all_train_label[data1_idx]
    train_data2 = all_train_data[data2_idx, :]
    train_label2 = all_train_label[data2_idx]
    data1_dataset = torch.utils.data.TensorDataset(train_data1, train_label1)
    data1_loader = torch.utils.data.DataLoader(dataset=data1_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
    data2_dataset = torch.utils.data.TensorDataset(train_data2, train_label2)
    data2_loader = torch.utils.data.DataLoader(dataset=data2_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)  
    
    test_data, test_label = synth_test_dataset(prior, positive_pretrain_test_data, negative_pretrain_test_data)
    test_dataset = torch.utils.data.TensorDataset(test_data, test_label)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    return train_data1, train_data2, train_label1, train_label2, data1_loader, data2_loader, test_loader

def gen_confdiff_train_loader(train_data1, train_data2, pcomp_confidence, train_label1, train_label2, confdiff_batch_size):
    confdiff_dataset = gen_index_dataset(train_data1, train_data2, pcomp_confidence, train_label1, train_label2)
    confdiff_train_loader = torch.utils.data.DataLoader(dataset=confdiff_dataset, batch_size=confdiff_batch_size, shuffle=True, num_workers=0)
    return confdiff_train_loader
       
    
def synth_test_dataset(prior, positive_test_data, negative_test_data):
    num_p = positive_test_data.shape[0]
    num_n = negative_test_data.shape[0]
    if prior == 0.2:
        nn = num_n
        np = int(num_n*0.25)
    elif prior == 0.5:
        if num_p > num_n:
            nn = num_n
            np = num_n
        else:
            nn = num_p
            np = num_p
    elif prior == 0.8:
        np = num_p
        nn = int(num_p*0.25)
    else:
        np = num_p
        nn = num_n        
    x = torch.cat((positive_test_data[:np, :], negative_test_data[:nn, :]), dim=0)
    y = torch.cat((torch.ones(np), -torch.ones(nn)), dim=0)
    return x, y  

def convert_to_binary_data(dataname, train_data, train_labels, test_data, test_labels):
    train_index = torch.arange(train_labels.shape[0])
    test_index = torch.arange(test_labels.shape[0])
    if dataname == 'cifar10':
        positive_train_index = torch.cat((torch.cat((torch.cat((torch.cat((torch.cat((train_index[train_labels==2],train_index[train_labels==3]),dim=0),train_index[train_labels==4]), dim=0),train_index[train_labels==5]),dim=0),train_index[train_labels==6]),dim=0),train_index[train_labels==7]),dim=0)
        negative_train_index = torch.cat((torch.cat((torch.cat((train_index[train_labels==0],train_index[train_labels==1]),dim=0),train_index[train_labels==8]),dim=0),train_index[train_labels==9]),dim=0)
        positive_test_index = torch.cat((torch.cat((torch.cat((torch.cat((torch.cat((test_index[test_labels==2],test_index[test_labels==3]),dim=0),test_index[test_labels==4]), dim=0),test_index[test_labels==5]),dim=0),test_index[test_labels==6]),dim=0),test_index[test_labels==7]),dim=0)
        negative_test_index = torch.cat((torch.cat((torch.cat((test_index[test_labels==0],test_index[test_labels==1]),dim=0),test_index[test_labels==8]),dim=0),test_index[test_labels==9]),dim=0)
    elif dataname == 'mnist' or dataname == 'fashion' or dataname == 'kmnist' or dataname == 'svhn':
        positive_train_index = torch.cat((torch.cat((torch.cat((torch.cat((train_index[train_labels==0],train_index[train_labels==2]),dim=0),train_index[train_labels==4]),dim=0),train_index[train_labels==6]),dim=0),train_index[train_labels==8]),dim=0)
        negative_train_index = torch.cat((torch.cat((torch.cat((torch.cat((train_index[train_labels==1],train_index[train_labels==3]),dim=0),train_index[train_labels==5]),dim=0),train_index[train_labels==7]),dim=0),train_index[train_labels==9]),dim=0)
        positive_test_index = torch.cat((torch.cat((torch.cat((torch.cat((test_index[test_labels==0],test_index[test_labels==2]),dim=0),test_index[test_labels==4]),dim=0),test_index[test_labels==6]),dim=0),test_index[test_labels==8]),dim=0)
        negative_test_index = torch.cat((torch.cat((torch.cat((torch.cat((test_index[test_labels==1],test_index[test_labels==3]),dim=0),test_index[test_labels==5]),dim=0),test_index[test_labels==7]),dim=0),test_index[test_labels==9]),dim=0)
    else:
        positive_train_index = train_index[train_labels==1]
        negative_train_index = train_index[train_labels==-1]
        positive_test_index = test_index[test_labels==1]
        negative_test_index = test_index[test_labels==-1]
    positive_train_data = train_data[positive_train_index, :].float()
    negative_train_data = train_data[negative_train_index, :].float()
    positive_test_data = test_data[positive_test_index, :].float()
    negative_test_data = test_data[negative_test_index, :].float()
    return positive_train_data, negative_train_data, positive_test_data, negative_test_data

def prepare_mnist_data():
    ordinary_train_dataset = dsets.MNIST(root='./dataset/mnist', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = dsets.MNIST(root='./dataset/mnist', train=False, transform=transforms.ToTensor())
    train_data = ordinary_train_dataset.data.reshape(-1, 1, 28, 28)
    train_labels = ordinary_train_dataset.targets
    test_data = test_dataset.data.reshape(-1, 1, 28, 28)
    test_labels = test_dataset.targets
    dim = 28*28
    return train_data, train_labels, test_data, test_labels, dim

def prepare_kmnist_data():
    ordinary_train_dataset = dsets.KMNIST(root='./dataset/KMNIST', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = dsets.KMNIST(root='./dataset/KMNIST', train=False, transform=transforms.ToTensor())
    train_data = ordinary_train_dataset.data.reshape(-1, 1, 28, 28)
    train_labels = ordinary_train_dataset.targets
    test_data = test_dataset.data.reshape(-1, 1, 28, 28)
    test_labels = test_dataset.targets
    dim = 28*28
    return train_data, train_labels, test_data, test_labels, dim

def prepare_fashion_data():
    ordinary_train_dataset = dsets.FashionMNIST(root='./dataset/FashionMnist', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = dsets.FashionMNIST(root='./dataset/FashionMnist', train=False, transform=transforms.ToTensor())
    train_data = ordinary_train_dataset.data.reshape(-1, 1, 28, 28)
    train_labels = ordinary_train_dataset.targets
    test_data = test_dataset.data.reshape(-1, 1, 28, 28)
    test_labels = test_dataset.targets
    dim = 28*28
    return train_data, train_labels, test_data, test_labels, dim

def prepare_cifar10_data():
    train_transform = transforms.Compose(
        [transforms.ToTensor(), # transforms.RandomHorizontalFlip(), transforms.RandomCrop(32,4),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
    test_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
    ordinary_train_dataset = dsets.CIFAR10(root='./dataset', train=True, transform=train_transform, download=True)
    test_dataset = dsets.CIFAR10(root='./dataset', train=False, transform=test_transform)
    train_data = torch.from_numpy(ordinary_train_dataset.data) # because data is a numpy type
    dim0, dim1, dim2, dim3 = train_data.shape # dim3 = 3
    train_data = train_data.reshape(dim0, dim3, dim1, dim2).float()
    train_labels = ordinary_train_dataset.targets
    train_labels = torch.tensor(train_labels).float() # because train_labels is a list type
    test_data = torch.from_numpy(test_dataset.data)
    dim0, dim1, dim2, dim3 = test_data.shape # dim3 = 3
    test_data = test_data.reshape(dim0, dim3, dim1, dim2).float()
    test_labels = test_dataset.targets
    test_labels = torch.tensor(test_labels).float()
    dim = 28*28
    return train_data, train_labels, test_data, test_labels, dim

def prepare_uci_data(ds):
    dataname = "data/"+ds+".mat" 
    current_data = sio.loadmat(dataname)
    data = current_data['data']
    label = current_data['label']
    data = torch.from_numpy(data).float()
    label = torch.from_numpy(label).float()
    labels = label.argmax(dim=1)
    #labels[labels==10] = 0
    train_index = torch.arange(labels.shape[0])
    if ds=='letter':
        positive_index = torch.cat((torch.cat((torch.cat((torch.cat((torch.cat((torch.cat((torch.cat((torch.cat((torch.cat((torch.cat((torch.cat((torch.cat((train_index[labels==0],train_index[labels==1]),dim=0),train_index[labels==2]),dim=0),train_index[labels==3]),dim=0),train_index[labels==4]),dim=0),train_index[labels==5]),dim=0),train_index[labels==6]),dim=0),train_index[labels==7]),dim=0),train_index[labels==8]),dim=0),train_index[labels==9]),dim=0),train_index[labels==10]),dim=0),train_index[labels==11]),dim=0),train_index[labels==12]),dim=0)
        negative_index = torch.cat((torch.cat((torch.cat((torch.cat((torch.cat((torch.cat((torch.cat((torch.cat((torch.cat((torch.cat((torch.cat((torch.cat((train_index[labels==13],train_index[labels==14]),dim=0),train_index[labels==15]),dim=0),train_index[labels==16]),dim=0),train_index[labels==17]),dim=0),train_index[labels==18]),dim=0),train_index[labels==19]),dim=0),train_index[labels==20]),dim=0),train_index[labels==21]),dim=0),train_index[labels==22]),dim=0),train_index[labels==23]),dim=0),train_index[labels==24]),dim=0),train_index[labels==25]),dim=0)
    else:
        positive_index = torch.cat((torch.cat((torch.cat((torch.cat((train_index[labels==0],train_index[labels==2]),dim=0),train_index[labels==4]),dim=0),train_index[labels==6]),dim=0),train_index[labels==8]),dim=0)
        negative_index = torch.cat((torch.cat((torch.cat((torch.cat((train_index[labels==1],train_index[labels==3]),dim=0),train_index[labels==5]),dim=0),train_index[labels==7]),dim=0),train_index[labels==9]),dim=0)
    positive_data = data[positive_index,:]
    negative_data = data[negative_index,:]
    np = positive_data.shape[0]
    nn = negative_data.shape[0]
    positive_data = positive_data[torch.randperm(positive_data.shape[0])]
    negative_data = negative_data[torch.randperm(negative_data.shape[0])]
    train_p = int(np*0.8)
    train_n = int(nn*0.8)
    positive_train_data  = positive_data[:train_p,:]
    positive_test_data = positive_data[train_p:,:]
    negative_train_data = negative_data[:train_n,:]
    negative_test_data = negative_data[train_n:,:]
    num_train = train_p +train_n
    num_test = (np+nn)-num_train
    dim = positive_train_data.shape[1]
    return positive_train_data, negative_train_data, positive_test_data, negative_test_data, num_train, num_test,dim

class gen_index_dataset(Dataset):
    def __init__(self, data1, data2, confidence, true_label1, true_label2):
        self.data1 = data1
        self.data2 = data2
        self.confidence = confidence
        self.true_label1 = true_label1
        self.true_label2 = true_label2
        
    def __len__(self):
        return len(self.data1)
        
    def __getitem__(self, index):
        each_data1 = self.data1[index]
        each_data2 = self.data2[index]
        each_confidence = self.confidence[index]
        each_true_label1 = self.true_label1[index]
        each_true_label2 = self.true_label2[index]
        return each_data1, each_data2, each_confidence, each_true_label1, each_true_label2