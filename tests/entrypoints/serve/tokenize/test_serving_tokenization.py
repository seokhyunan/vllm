# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from vllm.config.multimodal import MultiModalConfig
from vllm.entrypoints.openai.models.protocol import BaseModelPath
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.openai.parser.harmony_utils import get_encoding
from vllm.entrypoints.serve.render.serving import OpenAIServingRender
from vllm.entrypoints.serve.tokenize.protocol import (
    TokenizeChatRequest,
    TokenizeCompletionRequest,
)
from vllm.entrypoints.serve.tokenize.serving import OpenAIServingTokenization
from vllm.v1.engine.async_llm import AsyncLLM

MODEL_NAME = "openai-community/gpt2"
BASE_MODEL_PATHS = [
    BaseModelPath(name=MODEL_NAME, model_path=MODEL_NAME),
]


@dataclass
class MockHFConfig:
    model_type: str = "any"


@dataclass
class MockModelConfig:
    task = "generate"
    runner_type = "generate"
    model = MODEL_NAME
    tokenizer = MODEL_NAME
    trust_remote_code = False
    tokenizer_mode = "auto"
    max_model_len = 100
    tokenizer_revision = None
    multimodal_config = MultiModalConfig()
    hf_config = MockHFConfig()
    hf_text_config = MockHFConfig()
    logits_processors: list[str] | None = None
    diff_sampling_param: dict | None = None
    allowed_local_media_path: str = ""
    allowed_media_domains: list[str] | None = None
    encoder_config = None
    generation_config: str = "auto"
    media_io_kwargs: dict[str, dict[str, Any]] = field(default_factory=dict)
    skip_tokenizer_init = False
    is_encoder_decoder: bool = False
    is_multimodal_model: bool = False
    renderer_num_workers: int = 1

    def get_diff_sampling_param(self):
        return self.diff_sampling_param or {}


def _build_serving_tokenization(engine: AsyncLLM) -> OpenAIServingTokenization:
    models = OpenAIServingModels(
        engine_client=engine,
        base_model_paths=BASE_MODEL_PATHS,
    )
    serving_render = OpenAIServingRender(
        model_config=engine.model_config,
        renderer=engine.renderer,
        model_registry=models.registry,
        request_logger=None,
        chat_template=None,
        chat_template_content_format="auto",
    )
    return OpenAIServingTokenization(
        engine,
        models,
        openai_serving_render=serving_render,
        request_logger=None,
        chat_template=None,
        chat_template_content_format="auto",
    )


def _build_mock_engine(*, model_type: str = "any") -> AsyncLLM:
    mock_engine = MagicMock(spec=AsyncLLM)
    mock_engine.errored = False
    mock_engine.model_config = MockModelConfig()
    mock_engine.model_config.hf_config = MockHFConfig(model_type=model_type)
    mock_engine.model_config.hf_text_config = MockHFConfig(model_type=model_type)
    mock_engine.input_processor = MagicMock()
    mock_engine.renderer = MagicMock()
    return mock_engine


@pytest.mark.asyncio
async def test_tokenize_chat_skips_mm_cache_for_renderer_only_path():
    mock_engine = _build_mock_engine()

    serving = _build_serving_tokenization(mock_engine)
    serving.openai_serving_render.preprocess_chat = AsyncMock(
        return_value=(
            [{"role": "user", "content": "Test"}],
            [{"prompt_token_ids": [1, 2, 3]}],
        )
    )

    request = TokenizeChatRequest(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "Test prompt"}],
    )

    response = await serving.create_tokenize(request, MagicMock(headers={}))

    assert response.tokens == [1, 2, 3]
    assert (
        serving.openai_serving_render.preprocess_chat.call_args.kwargs["skip_mm_cache"]
        is True
    )


@pytest.mark.asyncio
async def test_tokenize_completion_skips_mm_cache_for_renderer_only_path():
    mock_engine = _build_mock_engine()

    serving = _build_serving_tokenization(mock_engine)
    serving.openai_serving_render.preprocess_completion = AsyncMock(
        return_value=[{"prompt_token_ids": [1, 2, 3]}]
    )

    request = TokenizeCompletionRequest(
        model=MODEL_NAME,
        prompt="Test prompt",
    )

    response = await serving.create_tokenize(request, MagicMock(headers={}))

    assert response.tokens == [1, 2, 3]
    assert (
        serving.openai_serving_render.preprocess_completion.call_args.kwargs[
            "skip_mm_cache"
        ]
        is True
    )


def _decode_harmony_tokenize_response(response) -> str:
    return get_encoding().decode(response.tokens)


@pytest.mark.asyncio
async def test_harmony_tokenize_default_thinking_prompt():
    mock_engine = _build_mock_engine(model_type="gpt_oss")
    serving = _build_serving_tokenization(mock_engine)
    serving.openai_serving_render.preprocess_chat = AsyncMock(
        side_effect=AssertionError("Harmony tokenization must use Harmony rendering")
    )

    request = TokenizeChatRequest(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "What is 1 + 1?"}],
        add_generation_prompt=True,
        chat_template_kwargs={"enable_thinking": True},
    )

    response = await serving.create_tokenize(request, MagicMock(headers={}))
    prompt = _decode_harmony_tokenize_response(response)

    assert prompt.endswith("<|start|>assistant<|channel|>analysis<|message|>")
    serving.openai_serving_render.preprocess_chat.assert_not_called()


@pytest.mark.asyncio
async def test_harmony_tokenize_reasoning_prefill_continuation():
    mock_engine = _build_mock_engine(model_type="gpt_oss")
    serving = _build_serving_tokenization(mock_engine)
    serving.openai_serving_render.preprocess_chat = AsyncMock(
        side_effect=AssertionError("Harmony tokenization must use Harmony rendering")
    )

    request = TokenizeChatRequest(
        model=MODEL_NAME,
        messages=[
            {"role": "user", "content": "What is 1 + 1?"},
            {
                "role": "assistant",
                "reasoning": "We know 1 + 1 is",
                "content": "",
            },
        ],
        add_generation_prompt=False,
        continue_final_message=True,
        chat_template_kwargs={
            "continuation_mode": "from_reasoning",
            "enable_thinking": True,
        },
    )

    response = await serving.create_tokenize(request, MagicMock(headers={}))
    prompt = _decode_harmony_tokenize_response(response)

    assert prompt.endswith("We know 1 + 1 is")
    assert "<|channel|>analysis<|message|>We know 1 + 1 is" in prompt
    assert "<|channel|>final<|message|>" not in prompt
    serving.openai_serving_render.preprocess_chat.assert_not_called()


@pytest.mark.asyncio
async def test_harmony_tokenize_answer_prefill_continuation():
    mock_engine = _build_mock_engine(model_type="gpt_oss")
    serving = _build_serving_tokenization(mock_engine)
    serving.openai_serving_render.preprocess_chat = AsyncMock(
        side_effect=AssertionError("Harmony tokenization must use Harmony rendering")
    )

    request = TokenizeChatRequest(
        model=MODEL_NAME,
        messages=[
            {"role": "user", "content": "What is 1 + 1?"},
            {
                "role": "assistant",
                "reasoning": "We already computed it.",
                "content": "The answer is",
            },
        ],
        add_generation_prompt=False,
        continue_final_message=True,
        chat_template_kwargs={
            "continuation_mode": "from_answer",
            "enable_thinking": True,
        },
    )

    response = await serving.create_tokenize(request, MagicMock(headers={}))
    prompt = _decode_harmony_tokenize_response(response)

    assert "<|channel|>analysis<|message|>We already computed it.<|end|>" in prompt
    assert "<|channel|>final<|message|>The answer is" in prompt
    assert prompt.endswith("The answer is")
    serving.openai_serving_render.preprocess_chat.assert_not_called()
