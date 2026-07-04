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

def eval(gt, noise_level, lambda_tv, P, max_iters, solver, K, device, GlobalResults, saveImage=False):
    for n in noise_level:
        for lam in lambda_tv:

            # Accumulatori temporanei per l'intero dataset per questo lambda
            dataset_performances = { f'p_{str(p)}': {'psnr': 0.0, 'ssim': 0.0} for p in P }

            x_true = gt.to(device)

            # Generazione problema inverso per il batch corrente
            y = K(x_true)
            y_delta = y + n * torch.randn_like(y)
            y_delta = y_delta.to(device)

            if saveImage and gt.shape[0] == 1:  # Salva solo se il batch ha una singola immagine
                save_image(normalize(y_delta), f"./TpVResult/Noisy_image_scale_{downscale_factor}_noise_{n}.png")

            lambda_key = f'lambda_{lam:e}'
            noise_key = f'noise_{n}'

            for p in P:

                p_str = f'p_{str(p)}'
                # Run the solver
                with torch.no_grad():  # Disabilita i gradienti per velocizzare l'inferenza e risparmiare memoria GPU
                    x_sol, info = solver(
                        y_delta,
                        x_true=x_true,
                        starting_point=torch.zeros_like(x_true),
                        lmbda=lam,
                        maxiter=max_iters,
                        p=p,
                        verbose=True,
                        device=device
                    )

                if saveImage and gt.shape[0] == 1:  # Salva solo se il batch ha una singola immagine
                    save_image(normalize(x_sol), f"./TpVResult/SR_image_scale_{downscale_factor}_noise_{n}_lambda_{lam:e}_p_{p}.png")

                psnr_val = PSNR(x_sol, x_true)
                ssim_val = SSIM(x_sol, x_true)

                if torch.is_tensor(psnr_val):
                    psnr_val = psnr_val.detach().item()
                if torch.is_tensor(ssim_val):
                    ssim_val = ssim_val.detach().item()

                dataset_performances[p_str]['psnr'] += psnr_val
                dataset_performances[p_str]['ssim'] += ssim_val

            for p in P:
                p_str = f'p_{str(p)}'
                GlobalResults[noise_key][lambda_key][p_str]['psnr'] = dataset_performances[p_str]['psnr']
                GlobalResults[noise_key][lambda_key][p_str]['ssim'] = dataset_performances[p_str]['ssim']
                print(f"p={p}, lambda={lam:e}, n={n} -> Mean PSNR: {GlobalResults[noise_key][lambda_key][p_str]['psnr']:.2f} dB | Mean SSIM: {GlobalResults[noise_key][lambda_key][p_str]['ssim']:.4f}")

# Set device
device = utilities.get_device()
#device = 'cpu'
print(f"Device used: {device}.")

print(f'🔄️ Loading test dataset...')
batch_size = 32  
downscale_factor = 4
test_dataset = MayoDataset(data_path='./Mayo/test', data_shape=256)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) # shuffle=False è preferibile per il test

K = operators.DownScaling(
    img_shape=torch.Size([256, 256]),
    downscale_factor=downscale_factor,
    mode='avg',
)

# Set up the solver parameters
lambda_tv = [2e-3, 9e-2, 6e-2]
max_iters = 150
P = [0.1, 0.4]
noise_level = [0.005, 0.01]
solver = solvers.ChambollePockTpVUnconstrained(K)

GlobalResults = {
    f"noise_{n}": {f'lambda_{lam:e}': { f'p_{str(p)}': {'psnr': 0.0, 'ssim': 0.0} for p in P} for lam in lambda_tv} for n in noise_level
}
GlobalResults['max_iteration'] = max_iters

eval(gt=transforms.Resize((256, 256))(transforms.ToTensor()(Image.open('./0.png').convert('L'))).unsqueeze(0),
     noise_level=noise_level,
     lambda_tv=lambda_tv,
     P=P,
     max_iters=max_iters,
     solver=solver,
     K=K,
     device=device,
     GlobalResults=GlobalResults,
     saveImage=True
)

eval(gt=next(iter(test_loader)),
     noise_level=noise_level,
     lambda_tv=lambda_tv,
     P=P,
     max_iters=max_iters,
     solver=solver,
     K=K,
     device=device,
     GlobalResults=GlobalResults
)

print()
print(f'✅ Global results => {json.dumps(GlobalResults, indent=2)}')

os.makedirs('./TpVResult', exist_ok=True)
with open(f'./TpVResult/Results_scale_{downscale_factor}.json', 'w', encoding='utf-8') as f:
    json.dump(GlobalResults, f, indent=2)

# Se desideri salvare un'immagine di esempio, salva l'ultima del ciclo
# save_image(normalize(x_sol[0]), f"./TpVResult/SR_image_TpV_sample.png")