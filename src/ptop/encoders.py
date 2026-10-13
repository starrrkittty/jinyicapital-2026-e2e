from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ._vendor.qlib.models import ALSTMModel, Net
from .schema import BOOK12_FIELDS, BOOK_FIELDS, FEATURES, OHLC_FIELDS, TRADE_FIELDS, ContractError


def field_indices(fields: Sequence[str]) -> tuple[int, ...]:
    fields = tuple(fields)
    if not fields or len(set(fields)) != len(fields) or not set(fields).issubset(FEATURES):
        raise ContractError("模型字段必须是固定 schema 的非空、无重复子集")
    return tuple(FEATURES.index(name) for name in fields)


def validate_inputs(x: Tensor, observed: Tensor, step_valid: Tensor) -> None:
    if x.ndim != 3 or x.shape[2] != len(FEATURES) or min(x.shape[:2]) == 0:
        raise ContractError("x 必须为非空 [B,L,25] 张量")
    if x.dtype != torch.float32:
        raise ContractError("x 必须使用 float32")
    if observed.shape != x.shape or observed.dtype != torch.bool:
        raise ContractError("observed 必须为 bool [B,L,25]")
    if step_valid.shape != x.shape[:2] or step_valid.dtype != torch.bool:
        raise ContractError("step_valid 必须为 bool [B,L]")
    if x.device != observed.device or x.device != step_valid.device:
        raise ContractError("输入与 mask 必须位于同一设备")
    if not torch.isfinite(x).all():
        raise ContractError("x 包含非有限数值")
    if (observed & ~step_valid.unsqueeze(-1)).any():
        raise ContractError("observed 不得越过 step_valid")
    if (x[~observed] != 0).any():
        raise ContractError("缺失字段和结构填充的 x 必须为零")
    if (step_valid[:, :-1] & ~step_valid[:, 1:]).any():
        raise ContractError("结构填充只能位于窗口左侧；真实缺失分钟必须保留有效时间槽")


def predictable_mask(observed: Tensor, fields: Sequence[str] = FEATURES) -> Tensor:
    if observed.ndim != 3 or observed.shape[-1] != len(FEATURES) or observed.dtype != torch.bool:
        raise ContractError("observed 必须为 bool [B,L,25]")
    return observed[..., field_indices(fields)].any(dim=(1, 2))


class InputEncoder(nn.Module):
    def __init__(self, fields: Sequence[str], output_dim: int):
        super().__init__()
        self.fields = tuple(fields)
        self.indices = field_indices(self.fields)
        self.output_dim = output_dim

    def prepare(self, x: Tensor, observed: Tensor, step_valid: Tensor) -> Tensor:
        validate_inputs(x, observed, step_valid)
        selected = observed[..., self.indices]
        invalid = ~selected.any(dim=(1, 2))
        if invalid.any():
            rows = invalid.nonzero(as_tuple=True)[0].tolist()
            raise ContractError(f"模型字段子集全空，必须在模型调用前分流样本；batch 行号={rows}")
        return torch.cat((x[..., self.indices], selected.to(x.dtype)), dim=-1)


class RawEncoder(InputEncoder):
    def __init__(self, fields: Sequence[str] = FEATURES):
        super().__init__(fields, 2 * len(fields))

    def forward(self, x: Tensor, observed: Tensor, step_valid: Tensor) -> tuple[Tensor, Tensor]:
        return self.prepare(x, observed, step_valid), step_valid


class ALSTMEncoder(InputEncoder):
    def __init__(self, fields: Sequence[str] = FEATURES, net: nn.Sequential | None = None):
        super().__init__(fields, 64)
        # 从锁定 Qlib 类取得完整投影，保持其层名、激活和初始化。
        self.net = net if net is not None else ALSTMModel(
            d_feat=2 * len(fields), hidden_size=64, num_layers=2, dropout=0.0, rnn_type="GRU"
        ).net

    def forward(self, x: Tensor, observed: Tensor, step_valid: Tensor) -> tuple[Tensor, Tensor]:
        tokens = self.net(self.prepare(x, observed, step_valid))
        return tokens.masked_fill(~step_valid.unsqueeze(-1), 0), step_valid


class UnifiedNetEncoder(InputEncoder):
    def __init__(self, output_dim: int = 64):
        if output_dim not in (64, 192):
            raise ContractError("统一 Net 的配对输出宽度必须为 64 或 192")
        super().__init__(FEATURES, output_dim)
        self.net = Net(input_dim=50, output_dim=output_dim, layers=(256,), act="LeakyReLU")

    def forward(self, x: Tensor, observed: Tensor, step_valid: Tensor) -> tuple[Tensor, Tensor]:
        rows = self.prepare(x, observed, step_valid)[step_valid]
        _check_batchnorm_rows(self.training, len(rows))
        tokens = x.new_zeros((*x.shape[:2], self.output_dim))
        tokens[step_valid] = self.net(rows)
        return tokens, step_valid


def _check_batchnorm_rows(training: bool, count: int) -> None:
    if training and count < 2:
        raise ContractError("训练期 BatchNorm 至少需要两个有效分钟行；请合并合法训练批次")


class GroupedNetEncoder(InputEncoder):
    def __init__(self):
        super().__init__(FEATURES, 192)
        self.groups = (OHLC_FIELDS, TRADE_FIELDS, BOOK_FIELDS)
        self.group_indices = tuple(field_indices(group) for group in self.groups)
        self.nets = nn.ModuleList(
            Net(input_dim=2 * len(group), output_dim=64, layers=(256,), act="LeakyReLU")
            for group in self.groups
        )

    def forward(self, x: Tensor, observed: Tensor, step_valid: Tensor) -> tuple[Tensor, Tensor]:
        self.prepare(x, observed, step_valid)
        _check_batchnorm_rows(self.training, int(step_valid.sum()))
        outputs = []
        # 各分支收集相同有效分钟行，组内全缺失仍保留该时间槽。
        for indices, net in zip(self.group_indices, self.nets, strict=True):
            values = x[..., indices][step_valid]
            flags = observed[..., indices][step_valid].to(x.dtype)
            outputs.append(net(torch.cat((values, flags), dim=-1)))
        tokens = x.new_zeros((*x.shape[:2], self.output_dim))
        tokens[step_valid] = torch.cat(outputs, dim=-1)
        return tokens, step_valid


class PatchEncoder(InputEncoder):
    def __init__(self, patch_size: int = 5):
        if type(patch_size) is not int or patch_size not in (1, 5):
            raise ContractError("配对实验 patch_size 必须为 1 或 5")
        super().__init__(FEATURES, 64)
        self.patch_size = patch_size
        self.projection = nn.Linear(51 * patch_size, self.output_dim)

    def patch_inputs(self, x: Tensor, observed: Tensor, step_valid: Tensor) -> tuple[Tensor, Tensor]:
        values = self.prepare(x, observed, step_valid)
        values = torch.cat((values, step_valid.unsqueeze(-1).to(x.dtype)), dim=-1)
        left_pad = (-x.shape[1]) % self.patch_size
        values = F.pad(values, (0, 0, left_pad, 0))
        valid = F.pad(step_valid, (left_pad, 0))
        # unfold 结果为 [B,N,F,P]，字段内时间由旧到新，端点锚定决策分钟。
        patches = values.unfold(1, self.patch_size, self.patch_size).flatten(2)
        patch_valid = valid.unfold(1, self.patch_size, self.patch_size).any(dim=-1)
        return patches, patch_valid

    def slot_bounds(self, step_valid: Tensor) -> tuple[Tensor, Tensor]:
        """返回每块首末真实槽位的原窗口位置；无真实槽位时均为 -1。"""
        if step_valid.ndim != 2 or step_valid.dtype != torch.bool or step_valid.shape[1] == 0:
            raise ContractError("step_valid 必须为非空 bool [B,L]")
        length = step_valid.shape[1]
        positions = torch.arange(length, device=step_valid.device).expand_as(step_valid)
        positions = positions.masked_fill(~step_valid, -1)
        positions = F.pad(positions, ((-length) % self.patch_size, 0), value=-1)
        blocks = positions.unfold(1, self.patch_size, self.patch_size)
        end = blocks.amax(dim=-1)
        start = blocks.masked_fill(blocks < 0, length).amin(dim=-1)
        return start.masked_fill(end < 0, -1), end

    def forward(self, x: Tensor, observed: Tensor, step_valid: Tensor) -> tuple[Tensor, Tensor]:
        patches, valid = self.patch_inputs(x, observed, step_valid)
        return self.projection(patches).masked_fill(~valid.unsqueeze(-1), 0), valid


def _spatial_layer(in_channels: int, width: int, stride: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, 32, (1, width), stride=(1, stride)), nn.LeakyReLU(0.01)
    )


def _time_layer(in_channels: int, out_channels: int, kernel: int) -> nn.Sequential:
    before = (kernel - 1) // 2
    after = kernel - 1 - before
    return nn.Sequential(
        nn.ZeroPad2d((0, 0, before, after)),
        nn.Conv2d(in_channels, out_channels, (kernel, 1)),
        nn.LeakyReLU(0.01),
    )


class DeepLOBEncoder(InputEncoder):
    def __init__(self):
        super().__init__(BOOK12_FIELDS, 192)
        # 依据论文算法与方案规格独立组织标准算子，三档宽度依次为 12、6、3、1。
        self.blocks = nn.ModuleList(
            nn.Sequential(
                _spatial_layer(channels, width, stride),
                _time_layer(32, 32, 4),
                _time_layer(32, 32, 4),
            )
            for channels, width, stride in ((2, 2, 2), (32, 2, 2), (32, 3, 1))
        )
        self.branches = nn.ModuleList(
            [
                nn.Sequential(_time_layer(32, 64, 1), _time_layer(64, 64, 3)),
                nn.Sequential(_time_layer(32, 64, 1), _time_layer(64, 64, 5)),
                nn.Sequential(
                    nn.MaxPool2d((3, 1), stride=1, padding=(1, 0)), _time_layer(32, 64, 1)
                ),
            ]
        )
        self.dropout = nn.Dropout1d(0.2)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def encode_book(self, book: Tensor) -> Tensor:
        for block in self.blocks:
            book = block(book)
        tokens = torch.cat([branch(book) for branch in self.branches], dim=1).squeeze(-1)
        # Dropout1d 在 [B,C,T] 上沿时间共享每个样本的通道 mask。
        return self.dropout(tokens).transpose(1, 2)

    def forward(self, x: Tensor, observed: Tensor, step_valid: Tensor) -> tuple[Tensor, Tensor]:
        self.prepare(x, observed, step_valid)
        lengths = step_valid.sum(dim=1)
        result = x.new_zeros((*x.shape[:2], self.output_dim))
        # 按长度分组先移除左填充，避免人为前缀经过多层卷积影响有效输出。
        for length in torch.unique(lengths).tolist():
            indices = (lengths == length).nonzero(as_tuple=True)[0]
            values = x[indices, -length:, :][..., self.indices]
            flags = observed[indices, -length:, :][..., self.indices].to(x.dtype)
            tokens = self.encode_book(torch.stack((values, flags), dim=1))
            result = result.index_copy(0, indices, F.pad(tokens, (0, 0, x.shape[1] - length, 0)))
        return result, step_valid
