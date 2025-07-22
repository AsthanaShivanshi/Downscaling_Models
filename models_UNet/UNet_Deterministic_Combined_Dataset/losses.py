import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class WeightedHuberLoss(nn.Module):
    def __init__(self, weights, delta=0.05):
        super().__init__()
        # Use provided weights directly (assumed normalized)
        weights = np.array(weights, dtype=np.float32)
        weights = weights / weights.sum()  # Just in case
        self.weights = torch.tensor(weights).float()
        self.delta = delta

    def forward(self, input, target):
        weights = self.weights.to(input.device)
        losses = []
        for c in range(input.shape[1]):
            loss = F.huber_loss(input[:, c], target[:, c], delta=self.delta, reduction='mean')
            losses.append(loss * weights[c])
        return torch.stack(losses).sum()

class WeightedMSELoss(nn.Module):
    def __init__(self, weights):
        super().__init__()
        weights = np.array(weights, dtype=np.float32)
        weights = weights / weights.sum()
        self.weights = torch.tensor(weights).float()

    def forward(self, input, target):
        weights = self.weights.to(input.device)
        losses = []
        for c in range(input.shape[1]):
            loss = F.mse_loss(input[:, c], target[:, c], reduction='mean')
            losses.append(loss * weights[c])
        return torch.stack(losses).sum()