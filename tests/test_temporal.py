"""Behavioral checks for the A encoder -> B backbone -> score contract."""

import json
import os
from dataclasses import replace

import pytest
import torch

from ptop.backbones import TemporalPool, _SelectiveStateBlock, compact_valid_tokens
from ptop.bundle import MODEL_NAMES, create_model
from ptop.encoders import predictable_mask
from ptop.schema import BOOK12_FIELDS, FEATURES, ContractError
from ptop.temporal import ENCODER_NAMES, MODERN_MODEL_NAMES, ModelSpec, create_score_model


@pytest.fixture(autouse=True, scope="module")
def few_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def inputs(length=13, lengths=(13, 8, 1, 10), seed=931):
    generator = torch.Generator().manual_seed(seed)
    positions = torch.arange(length).unsqueeze(0)
    valid = positions >= length - torch.tensor(lengths).unsqueeze(1)
    observed = (torch.rand((len(lengths), length, len(FEATURES)), generator=generator) > 0.25) & valid.unsqueeze(-1)
    observed[:, -1, :] = True
    # A true missing minute is not structural padding, even when every field is missing.
    if lengths[0] > 2:
        observed[0, -2, :] = False
    x = torch.randn(observed.shape, generator=generator).masked_fill(~observed, 0)
    return x, observed, valid


def modern_spec(name, **kwargs):
    return ModelSpec(name=name, width=16, head_width=12, layers=2, state_dim=4, lookback=120, **kwargs)


@pytest.mark.parametrize("name", MODERN_MODEL_NAMES)
def test_spec_json_roundtrip_and_factory(name):
    spec = modern_spec(name, encoder="ALSTM", pooling="mean")
    assert ModelSpec.from_dict(json.loads(json.dumps(spec.to_dict()))) == spec
    model = create_score_model(spec.to_dict()).eval()
    scores = model(*inputs())
    assert scores.shape == (4,)
    assert torch.isfinite(scores).all()
    assert model.encoder.fields == FEATURES
    assert model.encoder.output_dim == 64
    assert set(dict(model.named_children())) == {"encoder", "backbone", "pool", "head"}


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_legacy_models_are_exact_reconstructions(name):
    torch.manual_seed(128)
    original = create_model(name).eval()
    torch.manual_seed(128)
    reconstructed = create_score_model(ModelSpec(name=name, lookback=60)).eval()
    assert type(original) is type(reconstructed)
    assert original.state_dict().keys() == reconstructed.state_dict().keys()
    for key, value in original.state_dict().items():
        torch.testing.assert_close(value, reconstructed.state_dict()[key], rtol=0, atol=0)
    torch.testing.assert_close(original(*inputs()), reconstructed(*inputs()), rtol=0, atol=0)
    assert type(create_score_model(name)) is type(original)


@pytest.mark.parametrize("argument", [
    {"name": "B-Mamba"}, {"encoder": "unknown"}, {"width": True}, {"layers": 0}, {"lookback": -1},
    {"heads": 1.5}, {"ff_multiplier": 0}, {"kernel_size": False}, {"state_dim": 0}, {"head_width": None},
    {"dropout": float("nan")}, {"dropout": float("inf")}, {"dropout": 1}, {"dropout": True},
    {"pooling": "median"}, {"causal": "true"}, {"fields": ["close"]}, {"fields": ("close", "close")},
    {"fields": ("not_a_raw_field",)}, {"fields": ("close",), "encoder": "DeepLOB"},
    {"name": "B-Transformer", "width": 13, "heads": 4}, {"name": "R1", "width": 16},
])
def test_spec_rejects_invalid_or_silently_ignored_settings(argument):
    with pytest.raises(ContractError):
        ModelSpec(**({"name": "B-GRU"} | argument))


def test_serialized_spec_rejects_unknown_fields_and_bad_field_types():
    with pytest.raises(ContractError, match="未知字段"):
        ModelSpec.from_dict({"name": "B-GRU", "hidden_size": 16})
    with pytest.raises(ContractError):
        ModelSpec.from_dict({"fields": "close"})
    with pytest.raises(ContractError):
        ModelSpec.from_dict({1: "R1"})
    with pytest.raises(ContractError):
        create_score_model(None)
    spec = ModelSpec.from_dict({"name": "B-GRU", "fields": ["close", "volume"]})
    assert spec.fields == ("close", "volume")


@pytest.mark.parametrize("encoder", ("Raw", "ALSTM"))
def test_custom_fields_preserve_raw_schema_contract_and_ignore_other_fields(encoder):
    model = create_score_model(modern_spec("B-GRU", encoder=encoder, fields=("close", "volume"))).eval()
    batch = inputs()
    changed = batch[0].clone()
    excluded = [index for index, field in enumerate(FEATURES) if field not in model.encoder.fields]
    changed[..., excluded] *= 100
    assert model.encoder.fields == ("close", "volume")
    assert model.encoder.output_dim == (4 if encoder == "Raw" else 64)
    with torch.inference_mode():
        torch.testing.assert_close(model(*batch), model(changed, batch[1], batch[2]), rtol=0, atol=0)


@pytest.mark.parametrize("name", MODERN_MODEL_NAMES)
@pytest.mark.parametrize("pooling", ("last", "mean", "attention"))
def test_encoder_backbone_head_joint_gradients_and_updates(name, pooling):
    torch.manual_seed(711)
    model = create_score_model(modern_spec(name, pooling=pooling))
    groups = {key: [parameter for parameter in getattr(model, key).parameters() if parameter.requires_grad]
              for key in ("encoder", "backbone", "head")}
    before = {key: [parameter.detach().clone() for parameter in values] for key, values in groups.items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
    scores = model(*inputs())
    loss = torch.nn.functional.smooth_l1_loss(scores, torch.tensor([0.02, -0.01, 0.03, -0.02]))
    loss.backward()
    for key, parameters in groups.items():
        assert parameters, key
        assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in parameters), key
        assert any(parameter.grad.abs().sum() > 0 for parameter in parameters), key
    optimizer.step()
    for key, parameters in groups.items():
        assert any(not torch.equal(previous, parameter) for previous, parameter in zip(before[key], parameters)), key
    if pooling == "attention":
        assert model.pool.attention.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("encoder", ENCODER_NAMES)
def test_every_trainable_a_encoder_is_updated_without_precomputed_tokens(encoder):
    torch.manual_seed(332)
    model = create_score_model(modern_spec("B-GRU", encoder=encoder))
    parameters = list(model.encoder.parameters())
    if encoder == "Raw":
        assert not parameters
    else:
        assert parameters
    before = [parameter.detach().clone() for parameter in parameters]
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005)
    model(*inputs()).square().mean().backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in parameters)
    optimizer.step()
    if parameters:
        assert any(not torch.equal(old, parameter) for old, parameter in zip(before, parameters))


@pytest.mark.parametrize("name", MODERN_MODEL_NAMES)
@pytest.mark.parametrize("encoder", ENCODER_NAMES)
def test_all_a_encoders_compose_and_eval_microbatch_matches(name, encoder):
    torch.manual_seed(331)
    model = create_score_model(modern_spec(name, encoder=encoder, pooling="mean"))
    batch = inputs()
    with torch.no_grad():
        model(*batch)  # Fit nontrivial running statistics in A's BatchNorm encoders.
    model.eval()
    with torch.inference_mode():
        whole = model(*batch)
        pieces = torch.cat([model(*(item[index:index+1] for item in batch)) for index in range(4)])
        order = torch.tensor([2, 0, 3, 1])
        shuffled = model(*(item[order] for item in batch))
    torch.testing.assert_close(whole, pieces, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(whole[order], shuffled, rtol=2e-5, atol=2e-6)
    assert torch.isfinite(whole).all()


@pytest.mark.parametrize("name", MODERN_MODEL_NAMES)
@pytest.mark.parametrize("encoder", ("ALSTM", "Patch5", "DeepLOB"))
def test_left_padding_has_no_influence(name, encoder):
    torch.manual_seed(671)
    model = create_score_model(modern_spec(name, encoder=encoder)).eval()
    batch = inputs(length=17, lengths=(11,))
    unpadded = tuple(item[:, -11:] for item in batch)
    padded = (torch.nn.functional.pad(batch[0], (0, 0, 4, 0)),
              torch.nn.functional.pad(batch[1], (0, 0, 4, 0)),
              torch.nn.functional.pad(batch[2], (4, 0)))
    with torch.inference_mode():
        torch.testing.assert_close(model(*padded), model(*unpadded), rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("name", MODERN_MODEL_NAMES)
def test_real_missing_minutes_are_retained(name):
    torch.manual_seed(201)
    model = create_score_model(modern_spec(name)).eval()
    batch = inputs(length=9, lengths=(6, 9))
    assert batch[2][0, -2] and not batch[1][0, -2].any()
    tokens, token_valid = model.encoder(*batch)
    output, valid, lengths = model.backbone(tokens, token_valid)
    assert lengths.tolist() == [6, 9]
    assert valid[0, 4]  # The completely missing minute still occupies its real time slot.
    assert output.shape[:2] == (2, 9)
    compact, compact_mask, compact_lengths = compact_valid_tokens(tokens, token_valid)
    assert compact_lengths.tolist() == [6, 9]
    assert compact_mask[0, 4]
    # Removing that slot must not be the same operation as left-padding removal.
    removed = tuple(torch.cat((item[:, :-2], item[:, -1:]), dim=1) for item in batch)
    with torch.inference_mode():
        assert not torch.allclose(model(*batch), model(*removed), rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("name", MODERN_MODEL_NAMES)
def test_all_empty_and_encoder_field_empty_samples_must_be_diverted(name):
    model = create_score_model(modern_spec(name, encoder="ALSTM-Book")).eval()
    x = torch.zeros((2, 7, len(FEATURES)))
    observed = torch.zeros_like(x, dtype=torch.bool)
    valid = torch.ones((2, 7), dtype=torch.bool)
    observed[0, -1, FEATURES.index("close")] = True
    observed[1, -1, FEATURES.index(BOOK12_FIELDS[0])] = True
    mask = predictable_mask(observed, fields=model.encoder.fields)
    assert mask.tolist() == [False, True]
    with pytest.raises(ContractError, match="必须在模型调用前分流"):
        model(x, observed, valid)
    assert torch.isfinite(model(x[mask], observed[mask], valid[mask])).all()
    with pytest.raises(ContractError):
        model(torch.zeros_like(x), torch.zeros_like(observed), torch.zeros_like(valid))


def test_pool_last_and_mean_use_only_real_slots():
    output = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [1000.0, 1000.0]],
                           [[2.0, 3.0], [6.0, 7.0], [7.0, 8.0]]])
    valid = torch.tensor([[True, True, False], [True, True, True]])
    lengths = valid.sum(dim=1)
    torch.testing.assert_close(TemporalPool(2, "last")(output, valid, lengths), torch.tensor([[3.0, 4.0], [7.0, 8.0]]))
    torch.testing.assert_close(TemporalPool(2, "mean")(output, valid, lengths), torch.tensor([[2.0, 3.0], [5.0, 6.0]]))


def test_compaction_rejects_internal_structural_gap_and_empty_tokens():
    tokens = torch.zeros((2, 3, 5))
    with pytest.raises(ContractError, match="左侧"):
        compact_valid_tokens(tokens, torch.tensor([[True, False, True], [True, True, True]]))
    with pytest.raises(ContractError, match="全空"):
        compact_valid_tokens(tokens, torch.zeros((2, 3), dtype=torch.bool))


@pytest.mark.parametrize("name", MODERN_MODEL_NAMES)
@pytest.mark.parametrize("encoder", ("Raw", "Patch5"))
def test_long_configured_windows_have_no_hidden_position_capacity(name, encoder):
    spec = replace(modern_spec(name, encoder=encoder), lookback=2049, layers=1)
    model = create_score_model(spec).eval()
    batch = inputs(length=2049, lengths=(2049, 1601))
    with torch.inference_mode():
        assert torch.isfinite(model(*batch)).all()
    with pytest.raises(ContractError, match="超过 ModelSpec.lookback"):
        create_score_model(replace(spec, lookback=120))(*batch)


@pytest.mark.parametrize("name", MODERN_MODEL_NAMES)
def test_repeated_inference_does_not_mutate_parameters_buffers_rng_or_state(name):
    model = create_score_model(modern_spec(name, dropout=0.2)).eval()
    saved_state = {key: value.clone() for key, value in model.state_dict().items()}
    rng = torch.random.get_rng_state().clone()
    first_batch = inputs()
    other_batch = inputs(seed=441)
    with torch.inference_mode():
        original = model(*first_batch)
        model(*other_batch)
        after_other_stocks = model(*first_batch)
    torch.testing.assert_close(original, after_other_stocks, rtol=0, atol=0)
    assert torch.equal(rng, torch.random.get_rng_state())
    for key, value in saved_state.items():
        torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)


@pytest.mark.parametrize("name", ("B-TCN", "B-SelectiveSSM", "B-Transformer", "B-GRU"))
def test_backbone_causal_prefix_does_not_see_later_history(name):
    model = create_score_model(modern_spec(name, encoder="Raw"))
    batch = inputs(length=11, lengths=(11,))
    tokens, valid = model.encoder(*batch)
    changed = tokens.clone()
    changed[:, 7:] += 100
    model.eval()
    with torch.inference_mode():
        output, _, _ = model.backbone(tokens, valid)
        modified, _, _ = model.backbone(changed, valid)
    torch.testing.assert_close(output[:, :7], modified[:, :7], rtol=2e-5, atol=2e-6)


def test_transformer_layers_initialize_independently_and_odd_width_is_supported():
    model = create_score_model(ModelSpec(name="B-Transformer", width=15, heads=3))
    assert not torch.equal(model.backbone.layers[0].linear1.weight, model.backbone.layers[1].linear1.weight)
    assert torch.isfinite(model(*inputs())).all()


def test_repr_distinguishes_configuration_without_different_weight_shapes():
    spec = modern_spec("B-Transformer")
    model = create_score_model(spec)
    for changed in (replace(spec, pooling="mean"), replace(spec, causal=False), replace(spec, lookback=60)):
        candidate = create_score_model(changed)
        assert model.state_dict().keys() == candidate.state_dict().keys()
        assert all(value.shape == candidate.state_dict()[key].shape for key, value in model.state_dict().items())
        assert repr(model) != repr(candidate)
    before = repr(model)
    model.pool.mode = "mean"
    assert repr(model) != before  # Also expose behavioral attributes changed after construction.
    before = repr(model)
    model.backbone.causal = False
    assert repr(model) != before


def test_selective_scan_matches_closed_form_stable_constant_input_solution():
    block = _SelectiveStateBlock(width=2, state_dim=2, kernel_size=1, dropout=0.0).eval()
    rate = torch.tensor([[0.8, 1.7], [1.2, 2.3]])
    u = torch.nn.functional.silu(torch.tensor(1.0))
    with torch.no_grad():
        block.in_projection.weight.zero_()
        block.in_projection.bias.fill_(1)
        block.convolution.weight.fill_(1)
        block.convolution.bias.zero_()
        block.delta.weight.zero_()
        block.delta.bias.fill_(torch.log(torch.expm1(torch.tensor(0.1))))
        block.log_rate.copy_(torch.log(torch.expm1(rate - 1e-4)))
        block.selectors.weight.zero_()
        block.selectors.weight[:2, 0] = 0.2 / u
        block.selectors.weight[2:, 0] = 0.3 / u
        block.skip.zero_()
        block.out_projection.weight.copy_(torch.eye(2))
        block.out_projection.bias.zero_()
    valid = torch.tensor([[True] * 5 + [False] * 2])
    with torch.inference_mode():
        actual = block(torch.zeros((1, 7, 2)), valid)
    elapsed = torch.arange(1, 6)[:, None, None] * 0.1
    state = -torch.expm1(-elapsed * rate) / rate * torch.tanh(torch.tensor(0.2)) * u
    expected = state.sum(-1) * torch.tanh(torch.tensor(0.3)) * u
    torch.testing.assert_close(actual[0, :5], expected, rtol=2e-5, atol=2e-6)
    assert torch.count_nonzero(actual[0, 5:]) == 0


@pytest.mark.parametrize("bad_input", ("dtype", "nan", "nonzero_missing", "right_padding"))
def test_modern_models_preserve_strict_a_input_validation(bad_input):
    model = create_score_model(modern_spec("B-GRU"))
    x, observed, valid = inputs()
    if bad_input == "dtype":
        x = x.double()
    elif bad_input == "nan":
        x[0, -1, 0] = float("nan")
    elif bad_input == "nonzero_missing":
        x[0, -2, 0] = 1
    else:
        valid[0, -2] = False
    with pytest.raises(ContractError):
        model(x, observed, valid)


@pytest.mark.parametrize("name", MODERN_MODEL_NAMES)
def test_cpu_bfloat16_autocast_preserves_joint_training(name):
    torch.manual_seed(415)
    model = create_score_model(modern_spec(name))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        scores = model(*inputs())
        loss = scores.float().square().mean()
    loss.backward()
    assert torch.isfinite(scores).all()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())


@pytest.mark.parametrize("name", MODERN_MODEL_NAMES)
def test_cpu_bfloat16_validation_under_inference_mode_is_finite(name):
    model = create_score_model(modern_spec(name)).eval()
    batch = inputs()
    with torch.inference_mode():
        reference = model(*batch)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual = model(*batch)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.float(), reference, rtol=0.05, atol=0.01)


@pytest.mark.skipif(os.environ.get("PTOP_TEST_CUDA") != "1", reason="CUDA requires explicit isolated-device opt-in")
@pytest.mark.parametrize("name", MODERN_MODEL_NAMES)
def test_isolated_cuda_inference_matches_cpu_and_joint_gradients_are_finite(name):
    assert torch.cuda.is_available(), "PTOP_TEST_CUDA=1 requires an explicitly assigned CUDA_VISIBLE_DEVICES"
    torch.manual_seed(782)
    cpu = create_score_model(modern_spec(name)).eval()
    cuda = create_score_model(modern_spec(name)).cuda().eval()
    cuda.load_state_dict(cpu.state_dict(), strict=True)
    cpu_inputs = inputs()
    cuda_inputs = tuple(item.cuda() for item in cpu_inputs)
    with torch.inference_mode():
        expected = cpu(*cpu_inputs)
        actual = cuda(*cuda_inputs)
        pieces = torch.cat([cuda(*(item[index:index+1] for item in cuda_inputs)) for index in range(4)])
    torch.testing.assert_close(actual.cpu(), expected, rtol=2e-4, atol=2e-5)
    torch.testing.assert_close(actual, pieces, rtol=2e-4, atol=2e-5)
    cuda.train()
    before = {key: parameter.detach().clone() for key, parameter in cuda.named_parameters()}
    optimizer = torch.optim.AdamW(cuda.parameters(), lr=0.002)
    cuda(*cuda_inputs).square().mean().backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in cuda.parameters())
    optimizer.step()
    for prefix in ("encoder.", "backbone.", "head."):
        assert any(not torch.equal(before[key], parameter) for key, parameter in cuda.named_parameters()
                   if key.startswith(prefix))
