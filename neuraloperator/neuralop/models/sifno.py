import torch
import torch.nn as nn
from neuralop.layers import ScaleInformedSpectralConv

class SIFNO1d(nn.Module):
    def __init__(self, in_channels=1, out_channels=1,
                 width=32, n_layers=4, n_modes=64,
                 emb_channels=16, n_params=1):
        super().__init__()
        self.project_in = nn.Conv1d(in_channels, width, 1)
        self.blocks = nn.ModuleList([
            ScaleInformedSpectralConv(width, width, n_modes,
                                      emb_channels=emb_channels,
                                      n_params=n_params)
            for _ in range(n_layers)
        ])
        self.project_out = nn.Sequential(
            nn.Conv1d(width, width, 1), nn.GELU(),
            nn.Conv1d(width, out_channels, 1)
        )

    def forward(self, x, params):
        x = self.project_in(x)
        for blk in self.blocks:
            x = x + blk(x, params)   # residual
        return self.project_out(x)
    
class WrappedSIFNO(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.sifno = SIFNO1d(**kwargs)

    def forward(self, x_inputs, x_params, y=None, **kwargs):
        return self.sifno(x_inputs, x_params)