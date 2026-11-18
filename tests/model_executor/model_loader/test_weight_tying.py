# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_tying import maybe_retie_word_embeddings

VOCAB_SIZE = 16
HIDDEN_SIZE = 4


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "selector", ["embed_tokens", "language_model.model.embed_tokens"]
)
def test_embedding_offload_selector_excludes_lm_head(selector):
    """Embedding selectors must not select the separately projected LM head."""
    from vllm.model_executor.offloader import UVAOffloader

    offloader = UVAOffloader(3 * 1024**3, {selector})
    assert offloader.targets("language_model.model.embed_tokens.weight")
    assert not offloader.targets("language_model.lm_head.weight")
    assert not offloader.targets("language_model.model.layers.0.mlp.weight")
    assert not offloader.targets("language_model.model.embed_tokens_extra.weight")


@pytest.mark.cpu_test
@pytest.mark.usefixtures("dist_init")
def test_embedding_offload_does_not_use_functional_call(monkeypatch):
    """Tied vocabularies remain untouched when zero-copy UVA is disabled."""
    from vllm.model_executor.model_loader.utils import maybe_offload_embeddings
    from vllm.model_executor.offloader import (
        UVAOffloader,
        get_offloader,
        set_offloader,
    )

    monkeypatch.setenv("VLLM_WEIGHT_OFFLOADING_DISABLE_UVA", "1")
    offloader = UVAOffloader(3 * 1024**3, {"embed_tokens"})
    model = UntiedModel()
    model.lm_head.tie_weights(model.embed_tokens)
    original_weight = model.embed_tokens.weight
    previous = get_offloader()
    try:
        set_offloader(offloader)
        maybe_offload_embeddings(model)
    finally:
        set_offloader(previous)
    assert not offloader.supports_embedding_offload
    assert offloader.cpu_offload_bytes == 0
    assert model.embed_tokens.weight is original_weight
    assert model.lm_head.weight is original_weight


class UntiedModel(nn.Module):
    """Nests the head like a multimodal model, which is the harder case."""

    def __init__(self):
        super().__init__()
        self.language_model = nn.Module()
        self.language_model.model = nn.Module()
        self.language_model.model.embed_tokens = VocabParallelEmbedding(
            VOCAB_SIZE, HIDDEN_SIZE
        )
        self.language_model.lm_head = ParallelLMHead(VOCAB_SIZE, HIDDEN_SIZE)
        self.embed_tokens.weight.data.fill_(1.0)
        self.lm_head.weight.data.fill_(1.0)

    @property
    def embed_tokens(self) -> VocabParallelEmbedding:
        return self.language_model.model.embed_tokens

    @property
    def lm_head(self) -> ParallelLMHead:
        return self.language_model.lm_head


def make_model_config(untied_by_checkpoint=False):
    return SimpleNamespace(
        model="dummy-model",
        word_embeddings_untied_by_checkpoint=untied_by_checkpoint,
    )


@pytest.mark.cpu_test
@pytest.mark.usefixtures("dist_init")
@pytest.mark.parametrize("identical", [True, False])
def test_retie_only_when_identical(identical: bool):
    """A redundant copy of a tied lm_head is shared again to reclaim memory."""
    model = UntiedModel()
    if not identical:
        model.lm_head.weight.data.fill_(2.0)

    maybe_retie_word_embeddings(model, make_model_config(untied_by_checkpoint=True))

    assert (model.lm_head.weight is model.embed_tokens.weight) is identical
    if not identical:
        assert torch.all(model.lm_head.weight == 2.0)


@pytest.mark.cpu_test
@pytest.mark.usefixtures("dist_init")
def test_quantized_lm_head_is_left_alone():
    """A quantized head may store its weights packed under another name."""
    model = UntiedModel()
    model.lm_head.quant_method = SimpleNamespace()

    maybe_retie_word_embeddings(model, make_model_config(untied_by_checkpoint=True))

    assert model.lm_head.weight is not model.embed_tokens.weight


@pytest.mark.cpu_test
@pytest.mark.usefixtures("dist_init")
def test_no_retie_without_checkpoint_override():
    """Word embeddings the config genuinely unties are left alone."""
    model = UntiedModel()

    maybe_retie_word_embeddings(model, make_model_config())

    assert model.lm_head.weight is not model.embed_tokens.weight
