from math import exp
import torch
import torch.nn.functional as F


def psnr(input, target):
    """Computes peak signal-to-noise ratio."""
    return 10 * torch.log10(1 / F.mse_loss(input, target))

def create_ssim_window(window_size, channel):
    def gaussian(_window_size, sigma):
        gauss = torch.Tensor([exp(-(x - _window_size//2)**2/float(2*sigma**2)) for x in range(_window_size)]) 
        return gauss / gauss.sum()

    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = torch.autograd.Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def ssim(img1, img2, window_size=11, size_average=True, full=False):
    """compute ssim, img1 is input, img2 is target"""
    img1 = img1.unsqueeze(0)
    img2 = img2.unsqueeze(0)
    channel = img1.size()[1]
    window = create_ssim_window(window_size, channel)
    window = window.to(img1.device)
    window = window.type_as(img1)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = v1 / v2  # contrast * sensitivity

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    
    if size_average:
        cs = cs.mean()
        ret = ssim_map.mean()
    else:
        cs = cs
        ret = ssim_map

    if full:
        return ret, cs
    return ret