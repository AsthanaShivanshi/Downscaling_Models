import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedHuberLoss(nn.Module):
    def __init__(self, weights, delta=0.05):
        super().__init__()
        self.weights = torch.tensor(weights).float()
        assert torch.all(self.weights > 0), "Weights must be positive"
        assert torch.all((self.weights >= 0.1) & (self.weights <= 1.0)), "Weights must be in [0.1, 1.0]"
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
        self.weights = torch.tensor(weights).float()
        assert torch.all(self.weights > 0), "Weights must be positive"
        assert torch.all((self.weights >= 0.1) & (self.weights <= 1.0)), "Weights must be in [0.1, 1.0]"

    def forward(self, input, target):
        weights = self.weights.to(input.device)
        losses = []
        for c in range(input.shape[1]):
            loss = F.mse_loss(input[:, c], target[:, c], reduction='mean')
            losses.append(loss * weights[c])
        return torch.stack(losses).sum()