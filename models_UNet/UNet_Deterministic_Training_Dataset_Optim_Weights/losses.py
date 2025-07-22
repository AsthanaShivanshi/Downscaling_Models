import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class WeightedHuberLoss(nn.Module):
    def __init__(self, weights, delta=0.05):
        super().__init__()
        # Random initialization between 0.1 and 1.0
        random_weights = np.random.uniform(0.1, 1.0, size=len(weights))
        # Scale by config weights (relative importance)
        scaled_weights = random_weights * np.array(weights, dtype=np.float32)
        # Normalize so sum is 1
        final_weights = scaled_weights / scaled_weights.sum()
        self.weights = torch.tensor(final_weights).float()
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
        random_weights = np.random.uniform(0.1, 1.0, size=len(weights))
        scaled_weights = random_weights * np.array(weights, dtype=np.float32)
        final_weights = scaled_weights / scaled_weights.sum()
        self.weights = torch.tensor(final_weights).float()

    def forward(self, input, target):
        weights = self.weights.to(input.device)
        losses = []
        for c in range(input.shape[1]):
            loss = F.mse_loss(input[:, c], target[:, c], reduction='mean')
            losses.append(loss * weights[c])
        return torch.stack(losses).sum()