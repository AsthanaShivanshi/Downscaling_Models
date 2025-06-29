import torch
import numpy as np
from Network import Generator
from Custom_loss import CustomHuberLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_shape_lr = (5, 26, 60)   # PyTorch: (channels, H, W)
image_shape_hr = (5, 104, 240)
batch_size = 32
epochs = 30

# Load data (assuming npy files are (N, H, W, C), convert to (N, C, H, W))
def load_npy(path):
    arr = np.load(path, mmap_mode='c')
    if arr.ndim == 4:
        arr = np.transpose(arr, (0, 3, 1, 2))
    return arr

mean_pr = np.load('/data/ERA5_mean_train.npy', mmap_mode='c')[:, :, 5]
std_pr = np.load('/data/ERA5_std_train.npy', mmap_mode='c')[:, :, 5]
predictors = load_npy('/data/predictors_train_mean_std_separate.npy')
obs = load_npy('/data/obs_train_mean_std.npy')

# Convert to torch tensors
predictors = torch.tensor(predictors, dtype=torch.float32)
obs = torch.tensor(obs, dtype=torch.float32)

# Dataset and DataLoader
dataset = torch.utils.data.TensorDataset(predictors, obs)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model, loss, optimizer
generator = Generator(in_channels=6, out_channels=6).to(device)
criterion = CustomHuberLoss(mean_pr=torch.tensor(mean_pr), std_pr=torch.tensor(std_pr), delta=1.0)
optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4)

def train():
    generator.train()
    step = 0
    for epoch in range(epochs):
        for batch_gcm, batch_obs in dataloader:
            batch_gcm = batch_gcm.to(device)
            batch_obs = batch_obs.to(device)
            optimizer.zero_grad()
            hr_fake = generator(batch_gcm)
            loss = criterion(hr_fake, batch_obs)
            loss.backward()
            optimizer.step()
            step += 1
            with open('losses.txt', 'a') as f:
                f.write(f'Iteration>{step}, loss={loss.item():.6f}\n')
        # Save model every epoch (or as needed)
        torch.save(generator.state_dict(), f'generator_epoch_{epoch+1:03d}.pth')

if __name__ == "__main__":
    train()