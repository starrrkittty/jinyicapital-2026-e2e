"""Independent-stock temporal backbones with explicit structural-padding masks.

These implementations use PyTorch only. A real minute without an observation is
still a token: only ``token_valid=False`` structural padding is removed.
"""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .schema import ContractError


def compact_valid_tokens(tokens: Tensor, valid: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Move a left-padded suffix to the front without removing missing minutes."""
    if tokens.ndim != 3 or min(tokens.shape) == 0 or not tokens.is_floating_point():
        raise ContractError("tokens 必须为非空浮点 [B,T,D] 张量")
    if valid.shape != tokens.shape[:2] or valid.dtype != torch.bool or valid.device != tokens.device:
        raise ContractError("token_valid 必须为同设备 bool [B,T]")
    if not torch.isfinite(tokens).all():
        raise ContractError("tokens 包含非有限数值")
    if (valid[:, :-1] & ~valid[:, 1:]).any():
        raise ContractError("token 结构填充只能位于左侧；缺行情的真实分钟不能删去")
    lengths = valid.sum(dim=1)
    if (lengths == 0).any():
        raise ContractError("时序主干不接受全空 token；请先按模型字段分流样本")
    positions = torch.arange(int(lengths.max()), device=tokens.device).unsqueeze(0)
    prefix_valid = positions < lengths.unsqueeze(1)
    source_positions = (positions + tokens.shape[1] - lengths.unsqueeze(1)).clamp_max(tokens.shape[1] - 1)
    output = tokens.gather(1, source_positions.unsqueeze(-1).expand(-1, -1, tokens.shape[2]))
    return output.masked_fill(~prefix_valid.unsqueeze(-1), 0), prefix_valid, lengths


class TemporalPool(nn.Module):
    """Pool a prefix-valid output using last, mean, or learned attention."""

    def __init__(self, width: int, mode: str):
        super().__init__()
        if mode not in ("last", "mean", "attention"):
            raise ContractError("pooling 必须为 last、mean 或 attention")
        self.mode = mode
        self.attention = nn.Linear(width, 1) if mode == "attention" else None

    def extra_repr(self) -> str:
        return f"mode={self.mode!r}"

    def forward(self, output: Tensor, valid: Tensor, lengths: Tensor) -> Tensor:
        if self.mode == "last":
            return output[torch.arange(len(output), device=output.device), lengths - 1]
        if self.mode == "mean":
            return output.masked_fill(~valid.unsqueeze(-1), 0).sum(dim=1) / lengths.unsqueeze(-1)
        logits = self.attention(output).squeeze(-1).masked_fill(~valid, -torch.inf)
        weights = logits.softmax(dim=1)
        return (output * weights.unsqueeze(-1)).sum(dim=1)


class GRUBackbone(nn.Module):
    def __init__(self, input_dim: int, width: int, layers: int, dropout: float):
        super().__init__()
        self.rnn = nn.GRU(input_dim, width, num_layers=layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.norm = nn.LayerNorm(width)

    def forward(self, tokens: Tensor, valid: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        compact, prefix_valid, lengths = compact_valid_tokens(tokens, valid)
        packed = pack_padded_sequence(compact, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.rnn(packed)  # New zero state for every independent window.
        output, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=compact.shape[1])
        output = self.norm(output).masked_fill(~prefix_valid.unsqueeze(-1), 0)
        return output, prefix_valid, lengths


class TransformerBackbone(nn.Module):
    def __init__(self, input_dim: int, width: int, layers: int, dropout: float,
                 heads: int, ff_multiplier: int, causal: bool):
        super().__init__()
        self.projection = nn.Linear(input_dim, width)
        self.width = width
        self.causal = causal
        # Construct layers individually so they do not share identical cloned initialization.
        self.layers = nn.ModuleList(
            nn.TransformerEncoderLayer(width, heads, dim_feedforward=width * ff_multiplier,
                                       dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
            for _ in range(layers)
        )
        self.norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)
        frequencies = torch.exp(torch.arange(0, width, 2, dtype=torch.float32) * (-math.log(10000.0) / width))
        self.register_buffer("position_frequencies", frequencies)

    def extra_repr(self) -> str:
        return f"width={self.width}, causal={self.causal}"

    def positions(self, length: int, reference: Tensor) -> Tensor:
        """Sinusoidal positions built at runtime, with no fixed learned-table limit."""
        phase = torch.arange(length, device=reference.device, dtype=torch.float32).unsqueeze(1)
        phase = phase * self.position_frequencies.float().unsqueeze(0)
        positions = reference.new_zeros((length, self.width))
        positions[:, 0::2] = phase.sin()
        positions[:, 1::2] = phase.cos()[:, : self.width // 2]
        return positions

    def forward(self, tokens: Tensor, valid: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        compact, prefix_valid, lengths = compact_valid_tokens(tokens, valid)
        output = self.projection(compact)
        output = self.dropout(output + self.positions(output.shape[1], output).unsqueeze(0))
        output = output.masked_fill(~prefix_valid.unsqueeze(-1), 0)
        causal_mask = (torch.ones((output.shape[1], output.shape[1]), dtype=torch.bool, device=output.device).triu(1)
                       if self.causal else None)
        for layer in self.layers:
            output = layer(output, src_mask=causal_mask, src_key_padding_mask=~prefix_valid,
                           is_causal=self.causal)
            output = output.masked_fill(~prefix_valid.unsqueeze(-1), 0)
        return self.norm(output).masked_fill(~prefix_valid.unsqueeze(-1), 0), prefix_valid, lengths


class _CausalConv1d(nn.Conv1d):
    def forward(self, x: Tensor) -> Tensor:
        before = (self.kernel_size[0] - 1) * self.dilation[0]
        return super().forward(F.pad(x, (before, 0)))


class _TemporalConvBlock(nn.Module):
    def __init__(self, width: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.conv1 = _CausalConv1d(width, width, kernel_size, dilation=dilation)
        self.conv2 = _CausalConv1d(width, width, kernel_size, dilation=dilation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, valid: Tensor) -> Tensor:
        hidden = self.norm(x).masked_fill(~valid.unsqueeze(-1), 0).transpose(1, 2)
        hidden = self.dropout(F.gelu(self.conv1(hidden))).masked_fill(~valid.unsqueeze(1), 0)
        hidden = self.dropout(F.gelu(self.conv2(hidden))).transpose(1, 2)
        return (x + hidden).masked_fill(~valid.unsqueeze(-1), 0)


class TCNBackbone(nn.Module):
    def __init__(self, input_dim: int, width: int, layers: int, dropout: float, kernel_size: int):
        super().__init__()
        self.projection = nn.Linear(input_dim, width)
        self.blocks = nn.ModuleList(
            _TemporalConvBlock(width, kernel_size, 2**index, dropout) for index in range(layers)
        )
        self.norm = nn.LayerNorm(width)
        self.receptive_field = 1 + 2 * (kernel_size - 1) * (2**layers - 1)

    def forward(self, tokens: Tensor, valid: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        compact, prefix_valid, lengths = compact_valid_tokens(tokens, valid)
        output = self.projection(compact).masked_fill(~prefix_valid.unsqueeze(-1), 0)
        for block in self.blocks:
            output = block(output, prefix_valid)
        return self.norm(output).masked_fill(~prefix_valid.unsqueeze(-1), 0), prefix_valid, lengths


class _SelectiveStateBlock(nn.Module):
    """A diagonal selective SSM reference, inspired by Mamba's selection mechanism.

    This is an independent adaptation with exact zero-order-hold input integration,
    bounded input/output selectors, and a sequential PyTorch scan. It is not the
    official Mamba architecture or its fused hardware-aware implementation.
    """

    def __init__(self, width: int, state_dim: int, kernel_size: int, dropout: float):
        super().__init__()
        self.width = width
        self.state_dim = state_dim
        self.norm = nn.LayerNorm(width)
        self.in_projection = nn.Linear(width, 2 * width)
        self.convolution = _CausalConv1d(width, width, kernel_size, groups=width)
        self.delta = nn.Linear(width, width)
        self.selectors = nn.Linear(width, 2 * state_dim, bias=False)
        self.log_rate = nn.Parameter(torch.arange(1, state_dim + 1, dtype=torch.float32).log().repeat(width, 1))
        self.skip = nn.Parameter(torch.ones(width))
        self.out_projection = nn.Linear(width, width)
        self.dropout = nn.Dropout(dropout)
        nn.init.constant_(self.delta.bias, math.log(math.expm1(0.1)))

    def forward(self, x: Tensor, valid: Tensor) -> Tensor:
        u, gate = self.in_projection(self.norm(x)).chunk(2, dim=-1)
        u = u.masked_fill(~valid.unsqueeze(-1), 0)
        u = F.silu(self.convolution(u.transpose(1, 2)).transpose(1, 2))
        delta = F.softplus(self.delta(u))
        selectors_b, selectors_c = self.selectors(u).tanh().chunk(2, dim=-1)
        # Positive rates imply strictly stable exp(-delta * rate) transitions.
        rates = F.softplus(self.log_rate) + 1e-4
        state = x.new_zeros((len(x), self.width, self.state_dim))
        outputs = []
        for index in range(x.shape[1]):
            exponent = -delta[:, index, :, None] * rates.unsqueeze(0)
            transition = exponent.exp()
            integral = -torch.expm1(exponent) / rates.unsqueeze(0)
            proposed = (transition * state + integral * selectors_b[:, index, None, :]
                        * u[:, index, :, None])
            state = torch.where(valid[:, index, None, None], proposed, state)
            output = (state * selectors_c[:, index, None, :]).sum(dim=-1) + self.skip * u[:, index]
            outputs.append(output)
        output = torch.stack(outputs, dim=1) * F.silu(gate)
        return (x + self.dropout(self.out_projection(output))).masked_fill(~valid.unsqueeze(-1), 0)


class SelectiveSSMBackbone(nn.Module):
    def __init__(self, input_dim: int, width: int, layers: int, dropout: float,
                 state_dim: int, kernel_size: int):
        super().__init__()
        self.projection = nn.Linear(input_dim, width)
        self.blocks = nn.ModuleList(
            _SelectiveStateBlock(width, state_dim, kernel_size, dropout) for _ in range(layers)
        )
        self.norm = nn.LayerNorm(width)

    def forward(self, tokens: Tensor, valid: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        compact, prefix_valid, lengths = compact_valid_tokens(tokens, valid)
        output = self.projection(compact).masked_fill(~prefix_valid.unsqueeze(-1), 0)
        for block in self.blocks:
            output = block(output, prefix_valid)
        return self.norm(output).masked_fill(~prefix_valid.unsqueeze(-1), 0), prefix_valid, lengths
