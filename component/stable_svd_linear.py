import torch
import torch.nn as nn


class StableSVDLinear(nn.Module):
    """Low-rank projection on normal channels + dense projection on stripped outlier channels."""

    def __init__(
        self,
        in_features,
        out_features,
        rank,
        normal_indices,
        outlier_indices,
        bias=False,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = int(rank)

        if normal_indices is None:
            normal_indices = torch.arange(self.in_features, dtype=torch.long)
        if outlier_indices is None:
            outlier_indices = torch.empty(0, dtype=torch.long)

        normal_indices = normal_indices.to(torch.long)
        outlier_indices = outlier_indices.to(torch.long)
        self.register_buffer("normal_indices", normal_indices, persistent=True)
        self.register_buffer("outlier_indices", outlier_indices, persistent=True)

        self.has_low_rank = self.rank > 0 and self.normal_indices.numel() > 0
        self.has_outlier = self.outlier_indices.numel() > 0

        if self.has_low_rank:
            self.v_proj = nn.Linear(self.normal_indices.numel(), self.rank, bias=False)
            self.u_proj = nn.Linear(self.rank, self.out_features, bias=bias)
        else:
            self.v_proj = None
            self.u_proj = None
            self.bias = nn.Parameter(torch.zeros(self.out_features)) if bias else None

        if self.has_outlier:
            self.outlier_proj = nn.Linear(self.outlier_indices.numel(), self.out_features, bias=False)
        else:
            self.outlier_proj = None

    def _select_channels(self, x, indices):
        if indices.numel() == 0:
            return None
        if indices.numel() == x.shape[-1]:
            return x
        return x.index_select(-1, indices)

    def forward(self, x):
        out = None

        if self.has_low_rank:
            x_normal = self._select_channels(x, self.normal_indices)
            out = self.u_proj(self.v_proj(x_normal))

        if self.has_outlier:
            x_outlier = self._select_channels(x, self.outlier_indices)
            outlier_out = self.outlier_proj(x_outlier)
            out = outlier_out if out is None else out + outlier_out

        if out is None:
            out_shape = (*x.shape[:-1], self.out_features)
            out = torch.zeros(out_shape, dtype=x.dtype, device=x.device)

        if not self.has_low_rank and getattr(self, "bias", None) is not None:
            out = out + self.bias.to(dtype=out.dtype, device=out.device)

        return out
