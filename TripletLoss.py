import torch.nn as nn
import torch.functional as F
import torch
class TripletLoss(nn.Module):

    def __init__(self, margin = 1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
    

    def forward(self, anchor, positive, negative, size_average = True):

        distance_positive = (anchor - positive).pow(2).sum(1) 
        distance_negative = (anchor - negative).pow(2).sum(1)
        losses = nn.ReLU(distance_positive - distance_negative + self.margin,0)
        return losses.mean() if size_average else losses.sum()