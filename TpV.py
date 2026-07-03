from PIL import Image
import torch
from IPPy import utilities, operators, solvers
from IPPy.utilities import load_image, save_image, normalize
from IPPy.utilities.metrics import PSNR, SSIM
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import glob
import json
import os

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

# Set device
device = utilities.get_device()
print(f"Device used: {device}.")

print(f'🔄️ Loading test dataset...')
batch_size = 32  # Corretto typo
downscale_factor = 4
test_dataset = MayoDataset(data_path='./Mayo/test', data_shape=256)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) # shuffle=False è preferibile per il test

K = operators.DownScaling(
    img_shape=torch.Size([256, 256]),
    downscale_factor=downscale_factor,
    mode='avg',
)

# Set up the solver parameters
lambda_tv = [1e-5, 3e-5, 5e-5]
max_iters = 10
P = [0.1, 0.4]
noise_level = [0.005, 0.01]
solver = solvers.ChambollePockTpVUnconstrained(K)

# Inizializza GlobalResults usando stringhe per i float per perfetta compatibilità JSON
GlobalResults = {
    f"noise_{n}": {f'lambda_{lam:e}': { f'p_{str(p)}': {'psnr': 0.0, 'ssim': 0.0} for p in P} for lam in lambda_tv} for n in noise_level
}

# Disabilita i gradienti per velocizzare l'inferenza e risparmiare memoria GPU
with torch.no_grad():
    for n in noise_level:
        for lam in lambda_tv:
            
            # Accumulatori temporanei per l'intero dataset per questo lambda
            dataset_performances = { f'p_{str(p)}': {'psnr': 0.0, 'ssim': 0.0} for p in P }
            
            # Eseguo la valutazione su un solo betch, il primo del test_loader
            x_true = next(iter(test_loader)).to(device)
            
            # Generazione problema inverso per il batch corrente
            y = K(x_true)
            y_delta = y + n * torch.randn_like(y)
            y_delta = y_delta.to(device)

            lambda_key = f'lambda_{lam:e}'
            noise_key = f'noise_{n}'

            for p in P:
                print(f"\n--- Valutazione Noise Level: {n}, Lambda: {lam:e} e p: {p} ---", end='\r')
                
                p_str = f'p_{str(p)}'
                # Run the solver
                x_sol, info = solver(
                    y_delta,
                    x_true=x_true,
                    starting_point=torch.zeros_like(x_true),
                    lmbda=lam,
                    maxiter=max_iters,
                    p=p,
                    verbose=False, # Impostato a False per evitare spam in console sui batch
                    device=device
                )

                # NOTA: Assumiamo che PSNR e SSIM di IPPy restituiscano la media sul batch
                # Assicurati di usare .item() se restituiscono un tensore PyTorch
                psnr_val = PSNR(x_sol, x_true)
                ssim_val = SSIM(x_sol, x_true)

                dataset_performances[p_str]['psnr'] += psnr_val
                dataset_performances[p_str]['ssim'] += ssim_val
            
            for p in P:
                p_str = f'p_{str(p)}'
                GlobalResults[noise_key][lambda_key][p_str]['psnr'] = dataset_performances[p_str]['psnr']
                GlobalResults[noise_key][lambda_key][p_str]['ssim'] = dataset_performances[p_str]['ssim']
                print(f"p={p} -> Mean PSNR: {GlobalResults[noise_key][lambda_key][p_str]['psnr']:.2f} dB | Mean SSIM: {GlobalResults[noise_key][lambda_key][p_str]['ssim']:.4f}")

print()
print(f'✅ Global results => {json.dumps(GlobalResults, indent=2)}')

os.makedirs('./TpVResult', exist_ok=True)
with open(f'./TpVResult/Results_scale_{downscale_factor}.json', 'w', encoding='utf-8') as f:
    json.dump(GlobalResults, f, indent=2)

# Se desideri salvare un'immagine di esempio, salva l'ultima del ciclo
# save_image(normalize(x_sol[0]), f"./TpVResult/SR_image_TpV_sample.png")