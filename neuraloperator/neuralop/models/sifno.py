import torch
import torch.nn as nn
from neuralop.layers import ScaleInformedSpectralConv

def _pointwise_conv_nd(D, in_ch, out_ch):
    """Return 1x..x1 pointwise conv for D=1/2/3; fallback to Linear for D>3."""
    if D == 1:
        return nn.Conv1d(in_ch, out_ch, kernel_size=1)
    elif D == 2:
        return nn.Conv2d(in_ch, out_ch, kernel_size=1)
    elif D == 3:
        return nn.Conv3d(in_ch, out_ch, kernel_size=1)
    else:
        return _PointwiseLinear(in_ch, out_ch)
    
class _PointwiseLinear(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.lin = nn.Linear(in_ch, out_ch)
    def forward(self, x):
        # x: [B,C,*spatial]
        B, C = x.shape[:2]
        S = x.shape[2:]
        y = x.reshape(B, C, -1).transpose(1, 2)
        y = self.lin(y)
        y = y.transpose(1, 2).reshape(B, -1, *S)
        return y
    
class SIFNO(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, width: int = 128, n_layers: int = 6, n_modes: int | list[int] = 64, emb_channels: int = 16, n_params: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.width = width
        self.n_layers = n_layers
        self.n_modes = n_modes
        self.emb_channels = emb_channels
        self.n_params = n_params

        self._built = False

    def _build(self, spatial_dimensions):
        D = len(spatial_dimensions)
        if isinstance(self.n_modes, int):
            n_modes = [self.n_modes] * D
        else:
            n_modes = self.n_modes
            assert len(n_modes) == D, "n_modes must have length equal to spatial dimensions"

        self.project_in = _pointwise_conv_nd(D, self.in_channels, self.width)
        self.project_mid = _pointwise_conv_nd(D, self.width, self.width)
        self.project_out = nn.Sequential(
            _pointwise_conv_nd(D, self.width, self.width),
            nn.GELU(),
            _pointwise_conv_nd(D, self.width, self.out_channels),
        )

        blocks = []
        for _ in range(self.n_layers):
            blocks.append(ScaleInformedSpectralConv(in_channels=self.width, out_channels=self.width, n_modes=n_modes, emb_channels=self.emb_channels, n_params=self.n_params, bias=True, separable=False, implementation="reconstructed", fft_norm="forward"))
        self.blocks = nn.ModuleList(blocks)
        self._built = True

    def forward(self, x_inputs: torch.Tensor, x_params: torch.Tensor):
        if not self._built:
            self._build(x_inputs.shape[2:])
        x = self.project_in(x_inputs)
        for blk in self.blocks:
            x = x + blk(x, x_params)
        x = self.project_mid(x)
        return self.project_out(x)
    
class WrappedSIFNO(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.sifno = SIFNO(**kwargs)

    def forward(self, x_inputs, x_params, y=None, **kwargs):
        return self.sifno(x_inputs, x_params)