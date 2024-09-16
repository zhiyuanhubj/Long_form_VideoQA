import torch
import torch.nn.functional as F
from torch import nn, einsum
from torch.fft import rfft, irfft

from einops import rearrange


def exists(val):
    return val is not None


class DSS(nn.Module):
    def __init__(
        self,
        *,
        dim,
        kernel_N = 512,
        dss_kernel_lambda_imag_exp = True
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

        self.Lambda_real = nn.Parameter(torch.randn(kernel_N))
        self.Lambda_imag = nn.Parameter(torch.randn(kernel_N))


        self.C_real = nn.Parameter(torch.randn(dim, kernel_N))
        self.C_imag = nn.Parameter(torch.randn(dim, kernel_N))
        self.param_D = nn.Parameter(torch.randn(dim))

        self.dss_kernel_lambda_imag_exp = dss_kernel_lambda_imag_exp

    def forward(self, x):
        device, seq_len = x.device, x.shape[1]
        u = self.norm(x)


        residual = u * self.param_D


        Lambda_imag = self.Lambda_imag.exp() if self.dss_kernel_lambda_imag_exp else self.Lambda_imag

        Lambda = -self.Lambda_real.exp() + 1j * Lambda_imag
        C = self.C_real + 1j * self.C_imag

        arange = torch.arange(seq_len, device = device)

        S = (rearrange(Lambda, 'n -> n 1') * rearrange(arange, 'l -> 1 l')).exp()
        C = C * (Lambda.exp() - 1) / Lambda

        K = einsum('h n, n l -> l h', C, S).real

        u_f = rfft(u, n = seq_len * 2, dim = -2)
        K_f = rfft(K, n = seq_len * 2, dim = -2)

        y = irfft(u_f * K_f, seq_len * 2, dim = -2)[..., :seq_len, :]

        return y + residual

class GSS(nn.Module):
    """ Pseudocode 3.2 """

    def __init__(
        self,
        *,
        dim,
        dim_expansion_factor = 4,
        dss_kernel_N = 512,
        dss_kernel_H = 256,
        reverse_seq = False,
        dss_kernel_lambda_imag_exp = True
    ):
        super().__init__()
        self.reverse_seq = reverse_seq
        self.norm = nn.LayerNorm(dim)

        dim_hidden = int(dim_expansion_factor * dim)
        self.to_u = nn.Sequential(nn.Linear(dim, dim_hidden, bias = False), nn.GELU())
        self.to_v = nn.Sequential(nn.Linear(dim, dss_kernel_H, bias = False), nn.GELU())

        self.dss = DSS(dim = dss_kernel_H, kernel_N = dss_kernel_N, dss_kernel_lambda_imag_exp = dss_kernel_lambda_imag_exp)

        self.to_gate = nn.Linear(dss_kernel_H, dim_hidden, bias = False)
        self.to_out = nn.Linear(dim_hidden, dim)

    def forward(self, x):
        if self.reverse_seq:
            x = torch.flip(x, dims = (1,))

        residual, x = x.clone(), self.norm(x)

        u = self.to_u(x)
        v = self.to_v(x)

        v = self.dss(v)

        uc = self.to_gate(v)
        out = self.to_out(uc * u)

        out = out + residual

        if self.reverse_seq:
            out = torch.flip(out, dims = (1,))

        return out