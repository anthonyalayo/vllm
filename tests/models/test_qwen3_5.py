# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib

import pytest
import torch
from transformers import Qwen3_5TextConfig

from vllm.config import (
    DeviceConfig,
    ModelConfig,
    ParallelConfig,
    SpeculativeConfig,
    VllmConfig,
    set_current_vllm_config,
)


@pytest.mark.cpu_test
@pytest.mark.usefixtures("dist_init")
@pytest.mark.parametrize(
    ("dedicated", "tied", "pp_size", "draft_identity"),
    [
        (False, False, 1, "same"),
        (False, True, 1, "same"),
        (False, False, 2, "same"),
        (False, True, 2, "same"),
        (True, False, 1, "same"),
        (True, True, 1, "same"),
        (False, False, 1, "other_revision"),
        (False, False, 1, "other_checkpoint"),
    ],
)
def test_nvidia_mtp_vocab_is_attached_before_use(
    tmp_path, dedicated, tied, pp_size, draft_identity
):
    """PP1 shares target vocabulary without allocating or loading draft copies."""
    from vllm.distributed.parallel_state import in_single_pp_stage, single_pp_stage
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        ParallelLMHead,
        VocabParallelEmbedding,
    )
    from vllm.models.qwen3_5.nvidia.mtp import Qwen3_5MTP
    from vllm.v1.worker.gpu.spec_decode.eagle.utils import (
        maybe_share_target_embed,
        share_target_lm_head,
    )

    config = Qwen3_5TextConfig(
        architectures=["Qwen3_5ForCausalLM"],
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=128,
        mtp_num_hidden_layers=1,
        mtp_use_dedicated_embeddings=dedicated,
        tie_word_embeddings=tied,
    )
    config.save_pretrained(tmp_path)
    draft_path = tmp_path / "draft"
    if draft_identity == "other_checkpoint":
        config.save_pretrained(draft_path)
    target_config = ModelConfig(
        model=str(tmp_path),
        tokenizer_mode="skip",
        max_model_len=128,
        revision="target-revision",
    )
    parallel_config = ParallelConfig(pipeline_parallel_size=pp_size)
    speculative_config = SpeculativeConfig(
        method="mtp",
        model=str(draft_path) if draft_identity == "other_checkpoint" else None,
        num_speculative_tokens=3,
        revision=(
            "custom-draft-revision"
            if draft_identity == "other_revision"
            else "target-revision"
        ),
        target_model_config=target_config,
        target_parallel_config=parallel_config,
    )
    vllm_config = VllmConfig(
        model_config=target_config,
        device_config=DeviceConfig(device="cpu"),
        parallel_config=parallel_config,
        speculative_config=speculative_config,
    )
    with set_current_vllm_config(vllm_config), torch.device("cpu"):
        draft = Qwen3_5MTP(vllm_config=vllm_config)
        target = torch.nn.Module()
        target.embed_tokens = VocabParallelEmbedding(128, 64)
        target_head = ParallelLMHead(128, 64)
        if tied:
            target_head = target_head.tie_weights(target.embed_tokens)

    # Ordinary PP1 does not activate the special multi-rank PP context.
    assert not in_single_pp_stage()
    if dedicated or pp_size > 1 or draft_identity != "same":
        assert draft.model.embed_tokens is not None
        assert draft.lm_head is not None
        assert not draft.is_unused_checkpoint_weight("model.embed_tokens.weight")
        if tied:
            assert draft.lm_head.weight is draft.model.embed_tokens.weight
        return

    assert draft.model.embed_tokens is None
    assert draft.lm_head is None
    assert not hasattr(draft, "has_own_embed_tokens")
    assert not any(
        "embed_tokens" in name or "lm_head" in name
        for name, _ in draft.named_parameters()
    )
    assert draft.is_unused_checkpoint_weight("model.language_model.embed_tokens.weight")
    assert draft.is_unused_checkpoint_weight("lm_head.weight")
    assert draft.is_unused_checkpoint_weight("model.layers.0.mlp.down_proj.weight")
    assert not draft.is_unused_checkpoint_weight("mtp.fc.weight")
    from safetensors.torch import save_file

    from vllm.config import LoadConfig
    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

    weights = {
        name.replace("model.", "mtp.", 1): torch.full_like(param, 0.25)
        for name, param in draft.named_parameters()
    }
    # Invalid shapes prove the target vocabulary is filtered before loading.
    weights["model.language_model.embed_tokens.weight"] = torch.empty(0)
    weights["lm_head.weight"] = torch.empty(0)
    save_file(weights, tmp_path / "model.safetensors")
    DefaultModelLoader(LoadConfig(load_format="safetensors")).load_weights(
        draft, speculative_config.draft_model_config
    )
    assert torch.all(draft.model.fc.weight == 0.25)
    with single_pp_stage():
        assert not in_single_pp_stage()
        maybe_share_target_embed(draft, draft.model, target)
        share_target_lm_head(draft, target_head, None)
    assert draft.model.embed_tokens is target.embed_tokens
    assert draft.lm_head is target_head


def test_qwen3_5_models_use_platform_specific_implementation():
    from vllm.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5MTP
    from vllm.platforms import current_platform

    # The NVIDIA and ROCm variants drop torch.compile; other platforms
    # keep the previous torch.compile implementation in
    # vllm/model_executor/models/qwen3_5{,_mtp}.py.
    if current_platform.is_cuda():
        model_module = "vllm.models.qwen3_5.nvidia.model"
        mtp_module = "vllm.models.qwen3_5.nvidia.mtp"
    elif current_platform.is_rocm():
        model_module = "vllm.models.qwen3_5.amd.model"
        mtp_module = "vllm.models.qwen3_5.amd.mtp"
    else:
        model_module = "vllm.model_executor.models.qwen3_5"
        mtp_module = "vllm.model_executor.models.qwen3_5_mtp"
    assert Qwen3_5ForCausalLM.__module__ == model_module
    assert Qwen3_5MTP.__module__ == mtp_module


@pytest.mark.parametrize("platform", ["nvidia", "amd"])
def test_qwen3_5_platform_models_do_not_use_torch_compile_wrapper(platform: str):
    # Import the platform modules directly so the assertion is independent of
    # the platform this test runs on.
    from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper

    model_module = importlib.import_module(f"vllm.models.qwen3_5.{platform}.model")
    mtp_module = importlib.import_module(f"vllm.models.qwen3_5.{platform}.mtp")

    assert not issubclass(model_module.Qwen3_5Model, TorchCompileWithNoGuardsWrapper)
    assert not issubclass(
        mtp_module.Qwen3_5MultiTokenPredictor, TorchCompileWithNoGuardsWrapper
    )
    assert not issubclass(mtp_module.Qwen3_5MTP, TorchCompileWithNoGuardsWrapper)


def test_qwen3_5_legacy_models_keep_torch_compile_wrapper():
    # The legacy implementation in vllm/model_executor/models/ serves the
    # other platforms; it must keep the torch.compile wrapper.
    from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper
    from vllm.model_executor.models import qwen3_5 as model_module
    from vllm.model_executor.models import qwen3_5_mtp as mtp_module

    assert issubclass(model_module.Qwen3_5Model, TorchCompileWithNoGuardsWrapper)
    assert issubclass(
        mtp_module.Qwen3_5MultiTokenPredictor, TorchCompileWithNoGuardsWrapper
    )
    assert issubclass(mtp_module.Qwen3_5MTP, TorchCompileWithNoGuardsWrapper)
