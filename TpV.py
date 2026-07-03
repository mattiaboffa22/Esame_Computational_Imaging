import torch

from IPPy import utilities, operators, solvers
from IPPy.utilities import load_image, save_image, normalize
from IPPy.utilities.metrics import PSNR, SSIM

# Set device
device = utilities.get_device()
print(f"Device used: {device}.")

# Load GT image
x_true = load_image("267.png")
print(f"Shape of the GT: {list(x_true.shape)}.")

K = operators.DownScaling(
    img_shape=x_true.shape[-2:],
    downscale_factor=2,
    mode='avg',
)

# Build test problem
noise_level = 0.005
y=K(x_true)
y_delta = y + noise_level * torch.randn_like(y)
print(f"Shape of the measurements: {list(y_delta.shape)}.")

save_image(normalize(x_true), "./TpVResult/gt_image.png")
save_image(normalize(y_delta), "./TpVResult/DR_image.png")

# Set up the solver
lambda_tv = 1e-1
max_iters = 200
P = [0.1, 0.4]
solver = solvers.ChambollePockTpVUnconstrained(K)

for p in P:
    # Run the solver
    x_sol, info = solver(
        y_delta,
        x_true=x_true,
        starting_point=torch.zeros_like(x_true),
        lmbda=lambda_tv,
        maxiter=max_iters,
        p=p,
        verbose=True,
    )

    # Compute metrics
    psnr = PSNR(x_sol, x_true)
    ssim = SSIM(x_sol, x_true)
    print(f"PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f}")

    # Save the results
    save_image(normalize(x_sol), f"./TpVResult/SR_image_TpV_{p}.png")