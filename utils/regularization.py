import torch
import torch.nn as nn


class L12Smooth(nn.Module):
    def __init__(self):
        super(L12Smooth, self).__init__()

    def forward(self, input_tensor, a=0.05):
        """input: predictions"""
        return l12_smooth(input_tensor, a)


def l12_smooth(input_tensor, a=0.05):
    """Smoothed L1/2 norm"""
    if type(input_tensor) == list:
        return sum([l12_smooth(tensor) for tensor in input_tensor])

    smooth_abs = torch.where(torch.abs(input_tensor) < a,
                             torch.pow(input_tensor, 4) / (-8 * a ** 3) + torch.square(input_tensor) * 3 / 4 / a + 3 * a / 8,
                             torch.abs(input_tensor))

    return torch.sum(torch.sqrt(smooth_abs))
