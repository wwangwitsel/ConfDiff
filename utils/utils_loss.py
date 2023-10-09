import torch
import torch.nn as nn

def logistic_loss(pred):
     negative_logistic = nn.LogSigmoid()
     logistic = -1. * negative_logistic(pred)
     return logistic
