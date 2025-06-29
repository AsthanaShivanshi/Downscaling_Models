import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomHuberLoss(nn.Module):
    def __init__(self, mean_pr, std_pr, delta=1.0):
        super().__init__()
        self.mean_pr = mean_pr
        self.std_pr = std_pr
        self.delta = delta
        self.huber = nn.HuberLoss(delta=delta, reduction='mean')

    def forward(self, y_pred, y_true):
        # y_true, y_pred: (batch, channels, H, W) or (batch, H, W, channels)
        # Assume channels-last, as in your TF code. If channels-first, adjust indices accordingly.
        # For PyTorch, channels are usually first: (batch, channels, H, W)
        # Let's assume channels-first for standard PyTorch convention.

        # Precipitation channel is channel 5 (index 4 if 0-based)
        pr_true = y_true[:, 4, :, :] * self.std_pr + self.mean_pr
        weights = torch.clamp(pr_true / 6.0, 0.1, 1.0)

        # Huber loss for precipitation channel, weighted
        pr_pred = y_pred[:, 4, :, :]
        pr_target = y_true[:, 4, :, :]
        pr_diff = pr_pred - pr_target
        pr_huber = torch.where(
            torch.abs(pr_diff) < self.delta,
            0.5 * pr_diff ** 2,
            self.delta * (torch.abs(pr_diff) - 0.5 * self.delta)
        )
        pr_loss = torch.mean(weights * pr_huber)

        # Huber loss for other channels (channels 0:5, i.e., 0-4)
        other_pred = y_pred[:, 0:5, :, :]
        other_true = y_true[:, 0:5, :, :]
        other_loss = self.huber(other_pred, other_true)

        return other_loss + pr_loss