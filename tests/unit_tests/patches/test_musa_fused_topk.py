# Copyright (c) 2026 BAAI. All rights reserved.

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_fl.dispatch.backends.vendor.musa.impl.topk_sampling import (
    patch_fused_topk_sampling,
)


def tensor_metadata(shape, dtype, device, contiguous=True):
    return SimpleNamespace(
        shape=shape,
        ndim=len(shape),
        dtype=dtype,
        device=device,
        is_contiguous=lambda: contiguous,
    )


@pytest.fixture
def sampler_case(monkeypatch):
    native = Mock(return_value=("native", None))
    fused = Mock(return_value="fused")
    noise = Mock()
    noise.exponential_.return_value = noise
    monkeypatch.setattr(torch, "empty_like", Mock(return_value=noise))
    monkeypatch.setitem(
        sys.modules, "flaggems_vllm", SimpleNamespace(topk_sample_from_logits=fused)
    )
    sampler = type(
        "Sampler",
        (),
        {
            "forward_native": lambda self, *args: native(*args),
            "use_fp64_gumbel": False,
            "logprobs_mode": "raw_logprobs",
        },
    )
    patch_fused_topk_sampling(SimpleNamespace(TopKTopPSampler=sampler))
    device = SimpleNamespace(type="musa")
    logits = tensor_metadata((1, 6761), torch.float32, device)
    k = tensor_metadata((1,), torch.int32, device)
    return sampler(), logits, k, native, fused, noise


def test_musa_fused_topk(sampler_case):
    sampler, logits, k, native, fused, noise = sampler_case
    assert sampler.forward_native(logits, {}, k, None) == ("fused", None)
    native.assert_not_called()
    noise.exponential_.assert_called_once_with()
    fused.assert_called_once_with(logits, k, noise)


@pytest.mark.parametrize(
    "unsupported",
    [
        "top_p",
        "request_seed",
        "fp64",
        "processed_logits",
        "processed_logprobs",
        "wide_vocab",
        "non_contiguous",
        "cpu",
        "invalid_k",
        "no_top_k",
    ],
)
def test_musa_fused_topk_retains_native_path(sampler_case, unsupported):
    sampler, logits, k, native, fused, noise = sampler_case
    generators = {}
    p = None
    if unsupported == "top_p":
        p = object()
    elif unsupported == "request_seed":
        generators = {0: object()}
    elif unsupported == "fp64":
        sampler.use_fp64_gumbel = True
    elif unsupported in ("processed_logits", "processed_logprobs"):
        sampler.logprobs_mode = unsupported
    elif unsupported == "wide_vocab":
        logits.shape = (1, 8193)
    elif unsupported == "non_contiguous":
        logits.is_contiguous = lambda: False
    elif unsupported == "cpu":
        logits.device = torch.device("cpu")
    elif unsupported == "invalid_k":
        k.shape = (2,)
    elif unsupported == "no_top_k":
        k = None

    assert sampler.forward_native(logits, generators, k, p) == ("native", None)
    native.assert_called_once_with(logits, generators, k, p)
    fused.assert_not_called()
    noise.exponential_.assert_not_called()


def test_musa_fused_topk_reports_missing_kernel(monkeypatch):
    monkeypatch.setitem(sys.modules, "flaggems_vllm", SimpleNamespace())
    with pytest.raises(RuntimeError, match="requires FlagGems-vllm"):
        patch_fused_topk_sampling(SimpleNamespace())
