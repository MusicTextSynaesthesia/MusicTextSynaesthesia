from .CoordinateConfig import CoordinateReconConfig
from Trainer.CoordinateContrastiveTrainer import CoordinateContrastiveTrainer
import math
import torch
import torchaudio


def noise_aug(args, waveform):
    torch.manual_seed(waveform.shape[1])
    noise = (torch.rand_like(waveform) - 0.5) * 2
    aug_wave = waveform + noise * torch.rand(1).item() * 0.4
    return aug_wave


def freq_aug(args, waveform):
    base_freq = args.sample_rate
    target_freq = 1
    while math.gcd(base_freq, target_freq) < 8:         # 公因数越大重采样越快
        target_freq = int(base_freq * (0.95 + 0.1 * torch.rand(1).item()))  # 95%-105%
    aug_wave = torchaudio.transforms.Resample(base_freq, target_freq).cuda()(waveform)
    return aug_wave


def mask_aug(args, waveform):
    waveform = torch.split(waveform, args.sample_rate * 5, dim=1)  # 5s a chunk
    waveform = [w for w in waveform if torch.rand(1).item() < 0.90]  # 10% mask
    waveform = torch.cat(waveform, dim=1)
    return waveform


def mix_aug(args, waveform):
    return noise_aug(args, mask_aug(args, waveform))


class CoordinateAugConfig(CoordinateReconConfig):
    Trainer = CoordinateContrastiveTrainer

    epoch = 5
    model_name = "CoordinateContrastiveModel"
    aug_func = [noise_aug, freq_aug, mask_aug]
    contrastive_weight = 500
