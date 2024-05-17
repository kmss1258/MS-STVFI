from .pytorch_msssim import ssim_matlab
import torch
import math

def ssim_tensor_function(ground_truth, prediction):
    # ground_truth: B, 3, H, W, prediction: B, 3, H, W
    # ground_truth, prediction range: 0 ~ 1

    rounded_prediction = torch.clip(torch.round(prediction * 255), 0, 255) / 255.

    ssim = ssim_matlab(ground_truth, rounded_prediction).detach().cpu().numpy()

    return ssim

def psnr_tensor_function(ground_truth, prediction):
    # ground_truth: B, 3, H, W, prediction: B, 3, H, W
    # ground_truth, prediction range: 0 ~ 1

    quantized_ground_truth = torch.clip(torch.round(ground_truth * 255), 0, 255)
    rounded_prediction = torch.clip(torch.round(prediction * 255), 0, 255)

    mse = torch.mean((quantized_ground_truth - rounded_prediction) ** 2)
    return 20 * torch.log10(255.0 / torch.sqrt(mse)).detach().cpu().numpy()

if __name__ == '__main__':

    import torch.nn as nn
    import torch

    temp_tensor = torch.randn(1, 80, 128, 128)

    model = nn.Sequential(
        nn.ConvTranspose2d(in_channels=80, out_channels=6, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
    )

    out = model(temp_tensor)
    print(out.size())