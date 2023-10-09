import numpy as np
import torch
import torch.nn as nn
from utils.utils_loss import logistic_loss
import torch.nn.functional as F
from utils.utils_algo import accuracy_check

def pretrainLR(model, given_train_loader, test_loader, train_eval_loader, args, loss_fn, device, if_write=False, save_path=""):
    test_acc = accuracy_check(loader=test_loader, model=model, device=device)
    print('#epoch 0', ': test_accuracy', test_acc)
    test_acc_list = []
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)    
    for epoch in range(args.pretrain_ep):
        model.train()
        for (X, y) in given_train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)[:,0]
            pos_index, neg_index = (y == 1), (y == -1)
            total_num = pos_index.sum() + neg_index.sum()
            pos_train_loss, neg_train_loss = 0.0, 0.0
            if pos_index.sum() > 0:
                pos_train_loss = (loss_fn(outputs[pos_index])).sum()
            if neg_index.sum() > 0:
                neg_train_loss = (loss_fn(-outputs[neg_index])).sum()
            train_loss = (pos_train_loss + neg_train_loss) / total_num
            train_loss.backward()
            optimizer.step()
        model.eval()
        train_eval_acc = accuracy_check(loader=train_eval_loader, model=model, device=device)
        test_acc = accuracy_check(loader=test_loader, model=model, device=device)
        print('#epoch', epoch+1, ': train_loss', train_loss.data.item(), ' train_accuracy', train_eval_acc, ' test_accuracy', test_acc)
        if if_write:
            with open(save_path, "a") as f:
                f.writelines("{},{:.6f},{:.6f},{:.6f}\n".format(epoch + 1, train_loss.data.item(), train_eval_acc, test_acc))
        if epoch >= (args.pretrain_ep-10):
            test_acc_list.extend([test_acc])
    return np.mean(test_acc_list), model

def ConfDiffUnbiased(model, given_train_loader, test_loader, args, loss_fn, device, if_write=False, save_path=""):
    test_acc = accuracy_check(loader=test_loader, model=model, device=device)
    print('#epoch 0', ': test_accuracy', test_acc)
    test_acc_list = []
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    prior = args.prior
    
    for epoch in range(args.ep):
        model.train()
        for (X1, X2, conf, y1, y2) in given_train_loader:
            X1, X2, conf = X1.to(device), X2.to(device), conf.to(device)
            optimizer.zero_grad()
            outputs1 = model(X1)[:,0]
            outputs2 = model(X2)[:,0]
            train_loss = (prior - conf) * loss_fn(outputs1) + (1 - prior + conf) * loss_fn(-outputs1) + (prior + conf) * loss_fn(outputs2) + (1 - prior - conf) * loss_fn(-outputs2)
            train_loss = 0.5 * train_loss.mean()
            train_loss.backward()
            optimizer.step()
        model.eval()
        test_acc = accuracy_check(loader=test_loader, model=model, device=device)
        print('#epoch', epoch+1, ': train_loss', train_loss.data.item(), 'test_accuracy', test_acc)
        if if_write:
            with open(save_path, "a") as f:
                f.writelines("{},{:.6f},{:.6f}\n".format(epoch + 1, train_loss.data.item(), test_acc))
        if epoch >= (args.ep-10):
            test_acc_list.extend([test_acc])
    return np.mean(test_acc_list)

def ConfDiffReLU(model, given_train_loader, test_loader, args, loss_fn, device, if_write=False, save_path=""):
    test_acc = accuracy_check(loader=test_loader, model=model, device=device)
    print('#epoch 0', ': test_accuracy', test_acc)
    test_acc_list = []
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    prior = args.prior
    lda = torch.tensor([0.0]).to(device)
    
    for epoch in range(args.ep):
        model.train()
        for (X1, X2, conf, y1, y2) in given_train_loader:
            X1, X2, conf = X1.to(device), X2.to(device), conf.to(device)
            optimizer.zero_grad()
            outputs1 = model(X1)[:,0]
            outputs2 = model(X2)[:,0]
            train_loss1 = torch.max(((prior - conf) * loss_fn(outputs1)).mean(), lda)
            train_loss2 = torch.max(((1 - prior + conf) * loss_fn(-outputs1)).mean(), lda)
            train_loss3 = torch.max(((prior + conf) * loss_fn(outputs2)).mean(), lda)
            train_loss4 = torch.max(((1 - prior - conf) * loss_fn(-outputs2)).mean(), lda)
            train_loss = 0.5 * (train_loss1 + train_loss2 + train_loss3 + train_loss4)
            train_loss.backward()
            optimizer.step()
        model.eval()
        test_acc = accuracy_check(loader=test_loader, model=model, device=device)
        print('#epoch', epoch+1, ': train_loss', train_loss.data.item(), 'test_accuracy', test_acc)
        if if_write:
            with open(save_path, "a") as f:
                f.writelines("{},{:.6f},{:.6f}\n".format(epoch + 1, train_loss.data.item(), test_acc))

        if epoch >= (args.ep-10):
            test_acc_list.extend([test_acc])
    return np.mean(test_acc_list)

def ConfDiffABS(model, given_train_loader, test_loader, args, loss_fn, device, if_write=False, save_path=""):
    test_acc = accuracy_check(loader=test_loader, model=model, device=device)
    print('#epoch 0', ': test_accuracy', test_acc)
    test_acc_list = []
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    prior = args.prior
    lda = torch.tensor([0.0]).to(device)
    
    for epoch in range(args.ep):
        model.train()
        for (X1, X2, conf, y1, y2) in given_train_loader:
            X1, X2, conf = X1.to(device), X2.to(device), conf.to(device)
            optimizer.zero_grad()
            outputs1 = model(X1)[:,0]
            outputs2 = model(X2)[:,0]
            train_loss1 = torch.abs(((prior - conf) * loss_fn(outputs1)).mean())
            train_loss2 = torch.abs(((1 - prior + conf) * loss_fn(-outputs1)).mean())
            train_loss3 = torch.abs(((prior + conf) * loss_fn(outputs2)).mean())
            train_loss4 = torch.abs(((1 - prior - conf) * loss_fn(-outputs2)).mean())
            train_loss = 0.5 * (train_loss1 + train_loss2 + train_loss3 + train_loss4)
            train_loss.backward()
            optimizer.step()
        model.eval()
        test_acc = accuracy_check(loader=test_loader, model=model, device=device)
        print('#epoch', epoch+1, ': train_loss', train_loss.data.item(), 'test_accuracy', test_acc)
        if if_write:
            with open(save_path, "a") as f:
                f.writelines("{},{:.6f},{:.6f}\n".format(epoch + 1, train_loss.data.item(), test_acc))
        if epoch >= (args.ep-10):
            test_acc_list.extend([test_acc])
    return np.mean(test_acc_list)
