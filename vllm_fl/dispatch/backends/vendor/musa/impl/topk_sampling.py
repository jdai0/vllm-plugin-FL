# Copyright (c) 2026 BAAI. All rights reserved.

from functools import wraps

import torch


def patch_fused_topk_sampling(sampler_module):
    """Route eligible vLLM native sampler calls to FlagGems-vllm."""
    try:
        from flaggems_vllm import topk_sample_from_logits
    except ImportError as exc:
        raise RuntimeError(
            "VLLM_FL_MUSA_FUSED_TOPK=1 requires FlagGems-vllm with "
            "topk_sample_from_logits"
        ) from exc

    sampler = sampler_module.TopKTopPSampler
    original = sampler.forward_native
    processed_modes = getattr(
        sampler_module,
        "PROCESSED_LOGPROBS_MODES",
        ("processed_logits", "processed_logprobs"),
    )

    @wraps(original)
    def forward_native(self, logits, generators, k, p):
        if (
            p is None
            and k is not None
            and not generators
            and not getattr(self, "use_fp64_gumbel", True)
            and self.logprobs_mode not in processed_modes
            and logits.ndim == 2
            and logits.dtype == torch.float32
            and logits.device.type == "musa"
            and logits.is_contiguous()
            and 1 <= logits.shape[1] <= 8192
            and k.shape == (logits.shape[0],)
            and k.dtype in (torch.int32, torch.int64)
            and k.device == logits.device
            and k.is_contiguous()
        ):
            # Draw from the same Torch RNG as vLLM's FP32 random_sample.
            noise = torch.empty_like(logits).exponential_()
            return topk_sample_from_logits(logits, k, noise), None
        return original(self, logits, generators, k, p)

    sampler.forward_native = forward_native
