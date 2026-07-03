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
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from IPPy import operators, utilities
from torch.cuda.amp import autocast, GradScaler
from IPPy.utilities.metrics import PSNR, SSIM

device = utilities.get_device()
print(f'🚨 Device: {device}')
weights_dir = Path('GANSaver').resolve()
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

train_dataset = MayoDataset(data_path='./Mayo/train', data_shape=256)
test_dataset = MayoDataset(data_path='./Mayo/test', data_shape=256)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

def norm_layer(channels):
    num_groups = 8 if channels >= 8 else 1
    return nn.GroupNorm(num_groups=num_groups, num_channels=channels)

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
            GResidualBlock(32, 16),
            GResidualBlock(16, 16),
        )
        self.to_image = nn.Sequential(
            norm_layer(16),
            nn.SiLU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        h = self.fc(z).view(z.shape[0], 512, 4, 4)
        h = self.blocks(h)
        return self.to_image(h)

class StableCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = spectral_norm(nn.Conv2d(1, 16, kernel_size=3, padding=1))
        self.blocks = nn.Sequential(
            DResidualBlock(16, 32),
            DResidualBlock(32, 64),
            DResidualBlock(64, 128),
            DResidualBlock(128, 256),
            DResidualBlock(256, 512),
            DResidualBlock(512, 512),
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

opt_G = torch.optim.Adam(G.parameters(), lr=3e-4, betas=(0.0, 0.99))
opt_C = torch.optim.Adam(C.parameters(), lr=5e-5, betas=(0.0, 0.99))
plotGraph = 2  # Every 5 epochs there is a plot
G_to_C = 1     # For 2 Generator train step there is a critic train step
num_epochs = 50
ema_decay = 0.999
r1_weight = 5
r1_every = 16
fixed_z = torch.randn(8, latent_dim, device=device)

# --- AMP: uno scaler per ciascun optimizer, dato che i due step sono indipendenti ---
scaler_G = GradScaler()
scaler_C = GradScaler()

g_path = weights_dir / 'GAN_G.pth'
g_ema_path = weights_dir / 'GAN_G_EMA.pth'
c_path = weights_dir / 'GAN_C.pth'
g_history, c_history = [], []

#------------------------------------------------------------------------------------------
if(True):
    reloaded_G = StableGenerator(latent_dim=latent_dim)
    reloaded_G.load_state_dict(torch.load(g_ema_path, map_location='cpu', weights_only=True))
    reloaded_G = reloaded_G.to(device)
    reloaded_G.eval()

    def gan_dps_reconstruct(G, y_delta, K, latent_dim, device,
                         sigma_y=0.01, num_steps=100, eta=1e-2,
                         guidance_scale=1.0, lam=1e-3):
        """
        Inversione GAN guidata dai dati (ispirata alla logica DPS).
        Ottimizza lo spazio latente Z di un generatore congelato per corrispondere a y_delta.
        """
        batch_size = y_delta.shape[0]

        # Inizializzazione dello stato latente
        z = torch.randn(batch_size, latent_dim, device=device, requires_grad=True)
        optimizer = torch.optim.Adam([z], lr=eta)

        for step in range(num_steps):
            optimizer.zero_grad()  # Pulizia gradienti all'inizio dello step

            # Generazione dell'immagine e clamping per stabilità coerente con i dati
            x0_hat = G(z).clamp(-1.0, 1.0)

            # Termine di fedeltà ai dati
            data_loss = torch.mean((K(x0_hat) - y_delta) ** 2) / (2 * sigma_y ** 2)

            # Prior sullo spazio latente (incentiva z a rimanere Gaussiano)
            prior_loss = lam * torch.mean(z ** 2)

            # Loss Totale
            loss = guidance_scale * data_loss + prior_loss

            # Stampa di debug formattata meglio (senza sleep bloccanti)
            if step % 10 == 0 or step == num_steps - 1:
                print(f"Step {step:03d}/{num_steps} | Data Loss: {data_loss.item():.4f} | Prior Loss: {prior_loss.item():.4f} | Total Loss: {loss.item():.4f}", end='\r')

            # Backpropagation e aggiornamento
            loss.backward()
            optimizer.step()

        # Output finale congelato
        with torch.no_grad():
            return G(z).clamp(-1.0, 1.0).detach()

    k = operators.DownScaling(img_shape=(256,256), downscale_factor=2)
    noise = 0.005

    sample_image = test_dataset[0].unsqueeze(0).to(device)
    y_clean = k(sample_image)
    y_delta =  y_clean + noise * torch.randn_like(y_clean)

    x_gan_dps = gan_dps_reconstruct(
        reloaded_G, 
        y_delta, 
        K = k,
        latent_dim=128,
        sigma_y=noise, num_steps=1000, eta=1e-2, device=device
    )

    def denorm(x):
        return (x.clamp(-1.0, 1.0) + 1.0) / 2.0


    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # 1. Immagine vera (ground truth), alta risoluzione
    axes[0].imshow(denorm(sample_image).cpu().squeeze(), cmap='gray')
    axes[0].set_title('Ground truth (256×256)')
    axes[0].axis('off')

    # 2. Osservazione corrotta y_delta, bassa risoluzione
    axes[1].imshow(denorm(y_delta).cpu().squeeze(), cmap='gray')
    axes[1].set_title(f'Osservazione $y^\\delta$\n({y_delta.shape[-2]}×{y_delta.shape[-1]}, rumore={noise})')
    axes[1].axis('off')

    # 3. Ricostruzione GAN + DPS
    axes[2].imshow(denorm(x_gan_dps).cpu().squeeze(), cmap='gray')
    axes[2].set_title('Ricostruzione GAN+DPS')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

    psnr = PSNR(x_gan_dps, sample_image)
    ssim = SSIM(x_gan_dps, sample_image)
    print(f"\nPSNR: {psnr:.2f} dB | SSIM: {ssim:.4f}")

    exit()
#------------------------------------------------------------------------------------------

for epoch in range(num_epochs):
    G.train()
    C.train()
    g_epoch = 0.0
    c_epoch = 0.0
    progress_bar = tqdm(train_loader, desc=f'GAN epoch {epoch + 1}/{num_epochs}', leave=True)

    for step, x_real in enumerate(progress_bar, start=1):
        x_real = x_real.to(device, non_blocking=True)
        batch_size = x_real.shape[0]

        # ============== CRITIC STEP ==============
        with autocast():
            z = torch.randn(batch_size, latent_dim, device=device)
            x_fake = G(z)
            c_real = C(x_real)
            c_fake = C(x_fake.detach())
            c_loss = F.relu(1.0 - c_real).mean() + F.relu(1.0 + c_fake).mean()

        # R1 penalty: tenuta fuori da autocast, in fp32 puro.
        # create_graph=True (doppio backward) è numericamente fragile in fp16.
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

        if step % G_to_C == 0:
            opt_C.zero_grad(set_to_none=True)
            scaler_C.scale(c_loss).backward()
            scaler_C.step(opt_C)
            scaler_C.update()

        # ============== GENERATOR STEP ==============
        with autocast():
            z = torch.randn(batch_size, latent_dim, device=device)
            x_fake = G(z)
            g_loss = -C(x_fake).mean()

        opt_G.zero_grad(set_to_none=True)
        scaler_G.scale(g_loss).backward()
        scaler_G.step(opt_G)
        scaler_G.update()

        update_ema(G_ema, G, decay=ema_decay)

        g_epoch += g_loss.item()
        c_epoch += c_loss.item()
        progress_bar.set_postfix(g_loss=f'{g_loss.item():.5f}', c_loss=f'{c_loss.item():.5f}')

    g_history.append(g_epoch / len(train_loader))
    c_history.append(c_epoch / len(train_loader))

    if (epoch + 1) % plotGraph == 0 or epoch == num_epochs - 1:
        plt.figure(figsize=(5, 3))
        plt.plot(g_history, label='Generator')
        plt.plot(c_history, label='Critic')
        plt.title('GAN training losses')
        plt.xlabel('Epoch')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(weights_dir / 'training_losses.png')

torch.save(G.state_dict(), g_path)
torch.save(G_ema.state_dict(), g_ema_path)
torch.save(C.state_dict(), c_path)
print(f'Saved generator weights to: {g_path}')
print(f'Saved EMA generator weights to: {g_ema_path}')
print(f'Saved critic weights to: {c_path}')

