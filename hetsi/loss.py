import torch

class CMSE(torch.nn.Module):
    """Mean square error for complex numbers."""
    def __init__(self):
        super(CMSE, self).__init__()
        pass

    def forward(self, a: torch.Tensor, b: torch.Tensor):
        return torch.mean((a - b).abs() ** 2)

class mL1Reg(torch.nn.Module):
    """Mean L1 Regularization of input array.
    scale: float, weighting factor for regularizer. """
    def __init__(self, scale = 1e-1):
        super(mL1Reg, self).__init__()
        self.scale = scale
        pass

    def forward(self, a: torch.Tensor):
        return torch.mean(self.scale * a.abs())