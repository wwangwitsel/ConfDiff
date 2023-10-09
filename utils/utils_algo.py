import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils.utils_loss import logistic_loss
from utils.utils_models import linear_model, mlp_model
from cifar_models import resnet
    
def get_model(ds, mo, dim, device):
    if ds == 'cifar10':
        if mo == 'resnet':
            model = resnet(depth=32, num_classes=1).to(device)
    else:
        if mo == 'linear':
            model = linear_model(input_dim=dim, output_dim=1).to(device)
        elif mo == 'mlp':
            model = mlp_model(input_dim=dim, hidden_dim=300, output_dim=1).to(device)
    return model

def accuracy_check(loader, model, device):
    with torch.no_grad():
        total, num_samples = 0, 0
        for images, labels in loader:
            labels, images = labels.to(device), images.to(device)
            outputs = model(images)[:,0]
            predicted = (outputs.data >= 0).float()
            predicted[predicted == 0] = -1.0
            total += (predicted == labels).sum().item()
            num_samples += labels.size(0)
    return total / num_samples

def train_data_confidence_gen(loader, model, device, all_data_confidence):
    model.eval()
    with torch.no_grad():
        start_idx = 0
        for images, labels in loader:
            labels, images = labels.to(device), images.to(device)
            batch_size = images.shape[0]
            outputs = model(images)[:,0]
            confidence = torch.sigmoid(outputs).squeeze()
            all_data_confidence[start_idx:(start_idx+batch_size)] = confidence
            start_idx += batch_size
    return all_data_confidence, start_idx
