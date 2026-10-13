import importlib.resources
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from ptop._vendor.qlib import COMPONENTS_VERSION
from ptop.bundle import MODEL_NAMES, REQUIRED_CONTRACTS, create_model, load_bundle, save_bundle
from ptop.preprocessing import FrozenPreprocessor
from ptop.schema import FEATURES, QLIB_COMMIT, ContractError


def test_all_runtime_modules_import_without_external_qlib():
    # 模拟缺少外部 Qlib，验证每个运行模块的真实导入；实际无 Qlib 环境另行运行全套测试。
    code = """
import importlib.abc
import importlib
import pkgutil
import sys
class RejectExternalQlib(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'qlib' or fullname.startswith('qlib.'):
            raise ModuleNotFoundError('external Qlib is unavailable')
sys.meta_path.insert(0, RejectExternalQlib())
import ptop
for item in pkgutil.walk_packages(ptop.__path__, ptop.__name__ + '.'):
    importlib.import_module(item.name)
assert not any(name == 'qlib' or name.startswith('qlib.') for name in sys.modules)
"""
    root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "src")
    result = subprocess.run([sys.executable, "-c", code], env=environment, cwd=root,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_source_and_license_are_available_in_package():
    package = importlib.resources.files("ptop._vendor.qlib")
    assert "Copyright (c) Microsoft Corporation." in package.joinpath("LICENSE").read_text(encoding="utf-8")
    assert QLIB_COMMIT in package.joinpath("SOURCE.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_embedded_components_train_save_reload_and_infer(tmp_path, name):
    torch.set_num_threads(1)
    torch.manual_seed(17)
    index = pd.MultiIndex.from_product([pd.date_range("2024-01-02 09:31", periods=8, freq="min"), [1]],
                                      names=["datetime", "instrument"])
    values = pd.DataFrame(np.arange(200, dtype=np.float64).reshape(8, 25), index=index, columns=FEATURES)
    values.iloc[3, 0] = np.nan
    processor = FrozenPreprocessor("N1").fit(values, values.notna(), referenced_rows=index,
                                              fit_start="2024-01-02", fit_end="2024-01-02").freeze(
                                                  scale_review="synthetic embedded component test")
    normalized = processor.transform(values, values.notna())
    x = torch.as_tensor(np.stack([normalized.to_numpy()] * 2))
    observed = torch.as_tensor(np.stack([values.notna().to_numpy()] * 2))
    valid = torch.ones(2, 8, dtype=torch.bool)
    model = create_model(name).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    snapshot = {key: value.detach().clone() for key, value in model.named_parameters()}
    optimizer.zero_grad()
    loss = model(x, observed, valid).square().mean()
    assert torch.isfinite(loss)
    loss.backward()
    assert all(value.grad is not None and torch.isfinite(value.grad).all() for value in model.parameters())
    optimizer.step()
    assert any(not torch.equal(value, snapshot[key]) for key, value in model.named_parameters())
    model.eval()
    expected = model(x, observed, valid)
    path = tmp_path / name
    save_bundle(path, model, processor, model_name=name, window_length=8,
                contracts={key: "synthetic" for key in REQUIRED_CONTRACTS}, seed=17,
                training_record={"scope": "synthetic CPU verification"}, instrument_map={1: "SYNTHETIC"})
    restored, recovered, manifest = load_bundle(path)
    assert manifest["bundle_version"] == 2
    assert manifest["qlib_components_version"] == COMPONENTS_VERSION
    assert manifest["qlib_commit"] == QLIB_COMMIT
    assert "pyqlib" not in manifest["dependencies"]
    assert processor.fingerprint == recovered.fingerprint
    torch.testing.assert_close(expected, restored(x, observed, valid), atol=0, rtol=0)
    manifest["qlib_components_version"] += 1
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ContractError, match="组件版本"):
        load_bundle(path)
