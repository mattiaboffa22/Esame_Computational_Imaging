import glob
import math

from PIL import Image
from matplotlib import pyplot as plt
from pathlib import Path
import torch
from torch import nn
from torch.nn.utils import spectral_norm
from torchvision import transforms
from tqdm.auto import tqdm
from IPPy import utilities
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F


device = utilities.get_device()
book_root = Path('..').resolve()
weights_dir = book_root / 'weights'
weights_dir.mkdir(exist_ok=True)

class MayoDataset(Dataset):
    def __init__(self, data_path, data_shape=64):
        super().__init__()
        self.fname_list = sorted(glob.glob(f'{data_path}/*/*.png'))
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((data_shape, data_shape)),
        ])

    def __len__(self):
        return len(self.fname_list)

    def __getitem__(self, idx):
        x = Image.open(self.fname_list[idx]).convert('L')
        return self.transform(x)


def show_batch(batch, title, ncols=4):
    batch = batch.detach().cpu()
    n = min(len(batch), 8)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    axes = axes.reshape(-1) if hasattr(axes, 'reshape') else [axes]
    for ax, image in zip(axes, batch[:n]):
        ax.imshow(image.squeeze(), cmap='gray')
        ax.axis('off')
    for ax in axes[n:]:
        ax.axis('off')
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

train_dataset = MayoDataset(data_path='./Mayo/train', data_shape=64)
test_dataset = MayoDataset(data_path='./Mayo/test', data_shape=64)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


def norm_layer(channels):
    num_groups = 8 if channels >= 8 else 1
    return nn.GroupNorm(num_groups=num_groups, num_channels=channels)

def show_batch(batch, title, ncols=4):
    batch = batch.detach().cpu()
    n = min(len(batch), 8)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    axes = axes.reshape(-1) if hasattr(axes, 'reshape') else [axes]
    for ax, image in zip(axes, batch[:n]):
        ax.imshow(image.squeeze(), cmap='gray')
        ax.axis('off')
    for ax in axes[n:]:
        ax.axis('off')
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

class GResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.main = nn.Sequential(
            norm_layer(in_ch),
            nn.SiLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            norm_layer(out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        )
        self.skip = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_ch, out_ch, kernel_size=1),
        )

    def forward(self, x):
        return self.main(x) + self.skip(x)


class DResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.main = nn.Sequential(
            spectral_norm(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)),
            nn.AvgPool2d(2),
        )
        self.skip = nn.Sequential(
            nn.AvgPool2d(2),
            spectral_norm(nn.Conv2d(in_ch, out_ch, kernel_size=1)),
        )

    def forward(self, x):
        return self.main(x) + self.skip(x)


class StableGenerator(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.fc = nn.Linear(latent_dim, 512 * 4 * 4)
        self.blocks = nn.Sequential(
            GResidualBlock(512, 256),
            GResidualBlock(256, 128),
            GResidualBlock(128, 64),
            GResidualBlock(64, 32),
        )
        self.to_image = nn.Sequential(
            norm_layer(32),
            nn.SiLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        h = self.fc(z).view(z.shape[0], 512, 4, 4)
        h = self.blocks(h)
        return self.to_image(h)


class StableCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = spectral_norm(nn.Conv2d(1, 32, kernel_size=3, padding=1))
        self.blocks = nn.Sequential(
            DResidualBlock(32, 64),
            DResidualBlock(64, 128),
            DResidualBlock(128, 256),
            DResidualBlock(256, 512),
        )
        self.head = spectral_norm(nn.Linear(512 * 4 * 4, 1))

    def forward(self, x):
        h = self.stem(x)
        h = nn.functional.leaky_relu(h, negative_slope=0.2, inplace=True)
        h = self.blocks(h)
        h = h.flatten(start_dim=1)
        return self.head(h).view(-1)


@torch.no_grad()
def update_ema(ema_model, model, decay=0.999):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.mul_(decay).add_(param, alpha=1.0 - decay)
    for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers()):
        ema_buffer.copy_(buffer)

torch.manual_seed(0)

latent_dim = 128
G = StableGenerator(latent_dim=latent_dim).to(device)
G_ema = StableGenerator(latent_dim=latent_dim).to(device)
G_ema.load_state_dict(G.state_dict())
C = StableCritic().to(device)

opt_G = torch.optim.Adam(G.parameters(), lr=1e-4, betas=(0.0, 0.99))
opt_C = torch.optim.Adam(C.parameters(), lr=2e-4, betas=(0.0, 0.99))
num_epochs = 50
ema_decay = 0.999
r1_weight = 5.0
r1_every = 16
fixed_z = torch.randn(8, latent_dim, device=device)

g_path = weights_dir / 'GAN_G.pth'
g_ema_path = weights_dir / 'GAN_G_EMA.pth'
c_path = weights_dir / 'GAN_C.pth'
g_history, c_history = [], []

for epoch in range(num_epochs):
    G.train()
    C.train()
    g_epoch = 0.0
    c_epoch = 0.0
    progress_bar = tqdm(train_loader, desc=f'GAN epoch {epoch + 1}/{num_epochs}', leave=True)

    for step, x_real in enumerate(progress_bar, start=1):
        x_real = x_real.to(device)
        batch_size = x_real.shape[0]

        z = torch.randn(batch_size, latent_dim, device=device)
        x_fake = G(z)
        c_real = C(x_real)
        c_fake = C(x_fake.detach())
        c_loss = F.relu(1.0 - c_real).mean() + F.relu(1.0 + c_fake).mean()

        if step % r1_every == 0:
            x_real_reg = x_real.detach().requires_grad_(True)
            c_real_reg = C(x_real_reg)
            grad_real = torch.autograd.grad(
                outputs=c_real_reg.sum(),
                inputs=x_real_reg,
                create_graph=True,
            )[0]
            r1_penalty = grad_real.square().reshape(batch_size, -1).sum(dim=1).mean()
            c_loss = c_loss + 0.5 * r1_weight * r1_penalty

        opt_C.zero_grad()
        c_loss.backward()
        opt_C.step()

        z = torch.randn(batch_size, latent_dim, device=device)
        x_fake = G(z)
        g_loss = -C(x_fake).mean()

        opt_G.zero_grad()
        g_loss.backward()
        opt_G.step()
        update_ema(G_ema, G, decay=ema_decay)

        g_epoch += g_loss.item()
        c_epoch += c_loss.item()
        progress_bar.set_postfix(g_loss=f'{g_loss.item():.5f}', c_loss=f'{c_loss.item():.5f}')

    g_history.append(g_epoch / len(train_loader))
    c_history.append(c_epoch / len(train_loader))

torch.save(G.state_dict(), g_path)
torch.save(G_ema.state_dict(), g_ema_path)
torch.save(C.state_dict(), c_path)
print(f'Saved generator weights to: {g_path}')
print(f'Saved EMA generator weights to: {g_ema_path}')
print(f'Saved critic weights to: {c_path}')

reloaded_G = StableGenerator(latent_dim=latent_dim)
reloaded_G.load_state_dict(torch.load(g_ema_path, map_location='cpu', weights_only=True))
reloaded_G = reloaded_G.to(device)
reloaded_G.eval()

with torch.no_grad():
    x_fake = reloaded_G(fixed_z)

show_batch(x_fake, 'Generated Mayo-like slices from the trained GAN (EMA generator)')

plt.figure(figsize=(5, 3))
plt.plot(g_history, label='Generator')
plt.plot(c_history, label='Critic')
plt.title('GAN training losses')
plt.xlabel('Epoch')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
