from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ._vendor.qlib.models import ALSTMModel, GRUModel
from .encoders import ALSTMEncoder, DeepLOBEncoder, InputEncoder, RawEncoder
from .schema import FEATURES, ContractError


def count_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def _compact(tokens: Tensor, valid: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    if valid.shape != tokens.shape[:2] or valid.dtype != torch.bool:
        raise ContractError("token_valid 与 token 的形状不匹配")
    lengths = valid.sum(dim=1)
    if (lengths == 0).any():
        raise ContractError("时序主干不接受全空 token")
    if (valid[:, :-1] & ~valid[:, 1:]).any():
        raise ContractError("token 结构填充必须位于左侧")
    width = int(lengths.max())
    positions = torch.arange(width, device=tokens.device).unsqueeze(0)
    prefix_valid = positions < lengths.unsqueeze(1)
    source_positions = (positions + tokens.shape[1] - lengths.unsqueeze(1)).clamp_max(tokens.shape[1] - 1)
    compact = tokens.gather(1, source_positions.unsqueeze(-1).expand(-1, -1, tokens.shape[2]))
    return compact.masked_fill(~prefix_valid.unsqueeze(-1), 0), prefix_valid, lengths


def _run_rnn(rnn: nn.Module, tokens: Tensor, valid: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    compact, prefix_valid, lengths = _compact(tokens, valid)
    packed = pack_padded_sequence(compact, lengths.cpu(), batch_first=True, enforce_sorted=False)
    output, _ = rnn(packed)
    output, _ = pad_packed_sequence(output, batch_first=True, total_length=compact.shape[1])
    return output, prefix_valid, lengths


class R0Model(nn.Module):
    def __init__(self, fields: Sequence[str] = FEATURES):
        super().__init__()
        self.encoder = RawEncoder(fields)
        source = GRUModel(d_feat=self.encoder.output_dim, hidden_size=64, num_layers=2, dropout=0.0)
        self.rnn = source.rnn
        self.fc_out = source.fc_out

    def forward(self, x: Tensor, observed: Tensor, step_valid: Tensor) -> Tensor:
        tokens, valid = self.encoder(x, observed, step_valid)
        output, _, lengths = _run_rnn(self.rnn, tokens, valid)
        last = output[torch.arange(len(x), device=x.device), lengths - 1]
        return self.fc_out(last).squeeze(-1)


class AttentionModel(nn.Module):
    def __init__(self, encoder: InputEncoder, source: ALSTMModel | None = None):
        super().__init__()
        self.encoder = encoder
        source = source if source is not None else ALSTMModel(
            d_feat=2 * len(encoder.fields), hidden_size=64, num_layers=2, dropout=0.0, rnn_type="GRU"
        )
        self.rnn = source.rnn
        if encoder.output_dim != self.rnn.input_size:
            self.rnn = nn.GRU(encoder.output_dim, 64, num_layers=2, dropout=0.0, batch_first=True)
        self.att_net = source.att_net
        self.fc_out = source.fc_out

    def _score(self, tokens: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
        output, prefix_valid, lengths = _run_rnn(self.rnn, tokens, valid)
        logits = output
        # 保留源 att_net；仅在其唯一 Softmax 前屏蔽结构填充。
        for module in list(self.att_net.children())[:-1]:
            logits = module(logits)
        logits = logits.masked_fill(~prefix_valid.unsqueeze(-1), -torch.inf)
        weights = self.att_net[-1](logits)
        attended = (output * weights).sum(dim=1)
        last = output[torch.arange(len(tokens), device=tokens.device), lengths - 1]
        scores = self.fc_out(torch.cat((last, attended), dim=-1)).squeeze(-1)
        aligned = tokens.new_zeros(valid.shape)
        aligned[valid] = weights.squeeze(-1)[prefix_valid]
        return scores, aligned

    def forward_with_attention(self, x: Tensor, observed: Tensor, step_valid: Tensor) -> tuple[Tensor, Tensor]:
        return self._score(*self.encoder(x, observed, step_valid))

    def forward(self, x: Tensor, observed: Tensor, step_valid: Tensor) -> Tensor:
        return self.forward_with_attention(x, observed, step_valid)[0]


class R1Model(AttentionModel):
    def __init__(self, fields: Sequence[str] = FEATURES):
        source = ALSTMModel(d_feat=2 * len(fields), hidden_size=64, num_layers=2, dropout=0.0, rnn_type="GRU")
        super().__init__(ALSTMEncoder(fields=fields, net=source.net), source=source)


def initialize_keras_lstm(lstm: nn.LSTM) -> None:
    if lstm.num_layers != 1 or lstm.bidirectional or lstm.proj_size or not lstm.bias:
        raise ContractError("Keras 初始化适配限定单层、单向、有 bias 且无投影的 LSTM")
    nn.init.xavier_uniform_(lstm.weight_ih_l0)
    nn.init.orthogonal_(lstm.weight_hh_l0)
    nn.init.zeros_(lstm.bias_ih_l0)
    nn.init.zeros_(lstm.bias_hh_l0)
    # PyTorch 的 i,f,g,o 与 Keras 的 i,f,c,o 对应；两向量合计只给遗忘门加 1。
    with torch.no_grad():
        lstm.bias_ih_l0[lstm.hidden_size : 2 * lstm.hidden_size].fill_(1)


def load_keras_lstm_weights(lstm: nn.LSTM, kernel: Tensor, recurrent_kernel: Tensor, bias: Tensor) -> None:
    """映射标准 Keras 单 bias、i/f/c/o 门权重到本实现的 PyTorch LSTM。"""
    if lstm.num_layers != 1 or lstm.bidirectional or lstm.proj_size or not lstm.bias:
        raise ContractError("Keras 权重映射限定单层、单向、有 bias 且无投影的 LSTM")
    shapes = ((lstm.input_size, 4 * lstm.hidden_size), (lstm.hidden_size, 4 * lstm.hidden_size), (4 * lstm.hidden_size,))
    for name, value, shape in zip(("kernel", "recurrent_kernel", "bias"), (kernel, recurrent_kernel, bias), shapes, strict=True):
        if value.shape != shape or not torch.isfinite(value).all():
            raise ContractError(f"Keras {name} 形状或有限性不满足合同，要求 {shape}")
    with torch.no_grad():
        lstm.weight_ih_l0.copy_(kernel.T)
        lstm.weight_hh_l0.copy_(recurrent_kernel.T)
        lstm.bias_ih_l0.copy_(bias)
        lstm.bias_hh_l0.zero_()


class DeepLOBModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = DeepLOBEncoder()
        self.rnn = nn.LSTM(192, 64, num_layers=1, batch_first=True)
        self.fc_out = nn.Linear(64, 1)
        initialize_keras_lstm(self.rnn)
        nn.init.xavier_uniform_(self.fc_out.weight)
        nn.init.zeros_(self.fc_out.bias)

    def forward(self, x: Tensor, observed: Tensor, step_valid: Tensor) -> Tensor:
        tokens, valid = self.encoder(x, observed, step_valid)
        output, _, lengths = _run_rnn(self.rnn, tokens, valid)
        last = output[torch.arange(len(x), device=x.device), lengths - 1]
        return self.fc_out(last).squeeze(-1)
