# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import datetime
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal

from openai.types.responses.tool import Tool
from openai_harmony import (
    Author,
    Conversation,
    DeveloperContent,
    HarmonyEncodingName,
    Message,
    ReasoningEffort,
    RenderConversationConfig,
    Role,
    StreamableParser,
    SystemContent,
    TextContent,
    ToolDescription,
    load_harmony_encoding,
)

from vllm import envs
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionToolsParam
from vllm.logger import init_logger

logger = init_logger(__name__)

REASONING_EFFORT = {
    "high": ReasoningEffort.HIGH,
    "medium": ReasoningEffort.MEDIUM,
    "low": ReasoningEffort.LOW,
}

_harmony_encoding = None

# Builtin tools that should be included in the system message when
# they are available and requested by the user.
# Tool args are provided by MCP tool descriptions. Output
# of the tools are stringified.
BUILTIN_TOOL_TO_MCP_SERVER_LABEL: dict[str, str] = {
    "python": "code_interpreter",
    "browser": "web_search_preview",
    "container": "container",
}

# Derive MCP_BUILTIN_TOOLS from the canonical mapping
MCP_BUILTIN_TOOLS: set[str] = set(BUILTIN_TOOL_TO_MCP_SERVER_LABEL.values())
HARMONY_CHANNELS = ("analysis", "final", "commentary")
HARMONY_END = "<|end|>"


def has_custom_tools(tool_types: set[str]) -> bool:
    """
    Checks if the given tool types are custom tools
    (i.e. any tool other than MCP builtin tools)
    """
    return not tool_types.issubset(MCP_BUILTIN_TOOLS)


def get_encoding():
    global _harmony_encoding
    if _harmony_encoding is None:
        _harmony_encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    return _harmony_encoding


def get_system_message(
    model_identity: str | None = None,
    reasoning_effort: Literal["high", "medium", "low"] | None = None,
    start_date: str | None = None,
    browser_description: str | None = None,
    python_description: str | None = None,
    container_description: str | None = None,
    instructions: str | None = None,
    with_custom_tools: bool = False,
) -> Message:
    sys_msg_content = SystemContent.new()
    if model_identity is not None:
        sys_msg_content = sys_msg_content.with_model_identity(model_identity)
    if instructions is not None and envs.VLLM_GPT_OSS_HARMONY_SYSTEM_INSTRUCTIONS:
        current_identity = sys_msg_content.model_identity
        new_identity = (
            f"{current_identity}\n{instructions}" if current_identity else instructions
        )
        sys_msg_content = sys_msg_content.with_model_identity(new_identity)
    if reasoning_effort is not None:
        sys_msg_content = sys_msg_content.with_reasoning_effort(
            REASONING_EFFORT[reasoning_effort]
        )
    if start_date is None:
        # NOTE(woosuk): This brings non-determinism in vLLM.
        # Set VLLM_SYSTEM_START_DATE to pin it.
        start_date = envs.VLLM_SYSTEM_START_DATE or datetime.datetime.now().strftime(
            "%Y-%m-%d"
        )
    sys_msg_content = sys_msg_content.with_conversation_start_date(start_date)
    if browser_description is not None:
        sys_msg_content = sys_msg_content.with_tools(browser_description)
    if python_description is not None:
        sys_msg_content = sys_msg_content.with_tools(python_description)
    if container_description is not None:
        sys_msg_content = sys_msg_content.with_tools(container_description)
    sys_msg = Message.from_role_and_content(Role.SYSTEM, sys_msg_content)
    return sys_msg


def create_tool_definition(tool: ChatCompletionToolsParam | Tool):
    if isinstance(tool, ChatCompletionToolsParam):
        return ToolDescription.new(
            name=tool.function.name,
            description=tool.function.description,
            parameters=tool.function.parameters,
        )
    return ToolDescription.new(
        name=tool.name,
        description=tool.description,
        parameters=tool.parameters,
    )


def get_developer_message(
    instructions: str | None = None,
    tools: list[Tool | ChatCompletionToolsParam] | None = None,
) -> Message:
    dev_msg_content = DeveloperContent.new()
    if instructions is not None and not envs.VLLM_GPT_OSS_HARMONY_SYSTEM_INSTRUCTIONS:
        dev_msg_content = dev_msg_content.with_instructions(instructions)
    if tools is not None:
        function_tools: list[Tool | ChatCompletionToolsParam] = []
        for tool in tools:
            if tool.type in (
                "web_search_preview",
                "code_interpreter",
                "container",
            ):
                pass

            elif tool.type == "function":
                function_tools.append(tool)
            else:
                raise ValueError(f"tool type {tool.type} not supported")
        if function_tools:
            function_tool_descriptions = [
                create_tool_definition(tool) for tool in function_tools
            ]
            dev_msg_content = dev_msg_content.with_function_tools(
                function_tool_descriptions
            )
    dev_msg = Message.from_role_and_content(Role.DEVELOPER, dev_msg_content)
    return dev_msg


def get_user_message(content: str) -> Message:
    return Message.from_role_and_content(Role.USER, content)


def parse_chat_inputs_to_harmony_messages(
    chat_msgs: list,
    *,
    preserve_last_assistant_analysis: bool = False,
) -> list[Message]:
    """
    Parse a list of messages from request.messages in the Chat Completion API to
    Harmony messages.
    """
    msgs: list[Message] = []
    tool_id_names: dict[str, str] = {}

    # Collect tool id to name mappings for tool response recipient values
    for chat_msg in chat_msgs:
        for tool_call in chat_msg.get("tool_calls", []):
            tool_id_names[tool_call.get("id")] = tool_call.get("function", {}).get(
                "name"
            )

    for chat_msg in chat_msgs:
        msgs.extend(parse_chat_input_to_harmony_message(chat_msg, tool_id_names))

    msgs = auto_drop_analysis_messages(
        msgs,
        preserve_last_assistant_analysis=preserve_last_assistant_analysis,
    )
    return msgs


def auto_drop_analysis_messages(
    msgs: list[Message],
    *,
    preserve_last_assistant_analysis: bool = False,
) -> list[Message]:
    """
    Harmony models expect the analysis messages (representing raw chain of thought) to
    be dropped after an assistant message to the final channel is produced from the
    reasoning of those messages.

    The openai-harmony library does this if the very last assistant message is to the
    final channel, but it does not handle the case where we're in longer multi-turn
    conversations and the client gave us reasoning content from previous turns of
    the conversation with multiple assistant messages to the final channel in the
    conversation.

    So, we find the index of the last assistant message to the final channel and drop
    all analysis messages that precede it, leaving only the analysis messages that
    are relevant to the current part of the conversation.
    """
    last_assistant_final_index = -1
    for i in range(len(msgs) - 1, -1, -1):
        msg = msgs[i]
        if msg.author.role == "assistant" and msg.channel == "final":
            last_assistant_final_index = i
            break

    first_preserved_index = last_assistant_final_index
    if preserve_last_assistant_analysis and last_assistant_final_index == len(msgs) - 1:
        prev_idx = last_assistant_final_index - 1
        if (
            prev_idx >= 0
            and msgs[prev_idx].author.role == "assistant"
            and msgs[prev_idx].channel == "analysis"
        ):
            first_preserved_index = prev_idx

    cleaned_msgs: list[Message] = []
    for i, msg in enumerate(msgs):
        if i < first_preserved_index and msg.channel == "analysis":
            continue
        cleaned_msgs.append(msg)

    return cleaned_msgs


def flatten_chat_text_content(content: str | list | None) -> str | None:
    """
    Extract the text parts from a chat message content field and flatten them
    into a single string.
    """
    if isinstance(content, list):
        return "".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )
    return content


def parse_chat_input_to_harmony_message(
    chat_msg, tool_id_names: dict[str, str] | None = None
) -> list[Message]:
    """
    Parse a message from request.messages in the Chat Completion API to
    Harmony messages.
    """
    tool_id_names = tool_id_names or {}

    if not isinstance(chat_msg, dict):
        # Handle Pydantic models
        chat_msg = chat_msg.model_dump(exclude_none=True)

    role = chat_msg.get("role")
    msgs: list[Message] = []

    # Assistant message with tool calls
    tool_calls = chat_msg.get("tool_calls", [])

    if role == "assistant" and tool_calls:
        content = flatten_chat_text_content(chat_msg.get("content"))
        if content:
            commentary_msg = Message.from_role_and_content(Role.ASSISTANT, content)
            commentary_msg = commentary_msg.with_channel("commentary")
            msgs.append(commentary_msg)

        reasoning = chat_msg.get("reasoning")
        if reasoning:
            analysis_msg = Message.from_role_and_content(Role.ASSISTANT, reasoning)
            analysis_msg = analysis_msg.with_channel("analysis")
            msgs.append(analysis_msg)

        for call in tool_calls:
            func = call.get("function", {})
            name = func.get("name", "")
            arguments = func.get("arguments", "") or ""
            msg = Message.from_role_and_content(Role.ASSISTANT, arguments)
            msg = msg.with_channel("commentary")
            msg = msg.with_recipient(f"functions.{name}")
            # Officially, this should be `<|constrain|>json` but there is not clear
            # evidence that improves accuracy over `json` and some anecdotes to the
            # contrary. Further testing of the different content_types is needed.
            msg = msg.with_content_type("json")
            msgs.append(msg)
        return msgs

    # Tool role message (tool output)
    if role == "tool":
        tool_call_id = chat_msg.get("tool_call_id", "")
        name = tool_id_names.get(tool_call_id, "")
        content = chat_msg.get("content", "") or ""
        content = flatten_chat_text_content(content)

        msg = (
            Message.from_author_and_content(
                Author.new(Role.TOOL, f"functions.{name}"), content
            )
            .with_channel("commentary")
            .with_recipient("assistant")
        )
        return [msg]

    # Non-tool reasoning content
    reasoning = chat_msg.get("reasoning")
    if role == "assistant" and reasoning:
        analysis_msg = Message.from_role_and_content(Role.ASSISTANT, reasoning)
        analysis_msg = analysis_msg.with_channel("analysis")
        msgs.append(analysis_msg)

    # Default: user/assistant/system messages with content
    content = chat_msg.get("content") or ""
    if content is None:
        content = ""
    if isinstance(content, str):
        contents = [TextContent(text=content)]
    else:
        # TODO: Support refusal.
        contents = [TextContent(text=c.get("text", "")) for c in content]

    # Only add assistant messages if they have content, as reasoning or tool calling
    # assistant messages were already added above.
    if role == "assistant" and contents and contents[0].text:
        msg = Message.from_role_and_contents(role, contents)
        # Send non-tool assistant messages to the final channel
        msg = msg.with_channel("final")
        msgs.append(msg)
    # For user/system/developer messages, add them directly even if no content.
    elif role != "assistant":
        msg = Message.from_role_and_contents(role, contents)
        msgs.append(msg)

    return msgs


def render_for_completion(messages: list[Message]) -> list[int]:
    conversation = Conversation.from_messages(messages)
    token_ids = get_encoding().render_conversation_for_completion(
        conversation, Role.ASSISTANT
    )
    return token_ids


def _encode_harmony(text: str) -> list[int]:
    return get_encoding().encode(text, allowed_special="all")


def _harmony_channel_prefix(channel: str) -> str:
    return f"<|channel|>{channel}<|message|>"


def _harmony_assistant_channel_prefix(channel: str) -> str:
    return f"<|start|>assistant{_harmony_channel_prefix(channel)}"


def _strip_trailing_end_token(token_ids: list[int]) -> list[int]:
    end_token_ids = _encode_harmony(HARMONY_END)
    if (
        len(token_ids) >= len(end_token_ids)
        and token_ids[-len(end_token_ids) :] == end_token_ids
    ):
        return token_ids[: -len(end_token_ids)]
    return token_ids


def _render_conversation_without_auto_drop(messages: list[Message]) -> list[int]:
    return get_encoding().render_conversation(
        Conversation.from_messages(messages),
        RenderConversationConfig(auto_drop_analysis=False),
    )


def _last_assistant_channel(messages: Sequence[Message]) -> str | None:
    for msg in reversed(messages):
        if msg.author.role == "assistant":
            return msg.channel
    return None


def _get_harmony_continuation_mode(
    chat_template_kwargs: Mapping[str, Any] | None,
) -> Literal["from_reasoning", "from_answer"] | None:
    if not chat_template_kwargs:
        return None
    continuation_mode = chat_template_kwargs.get("continuation_mode")
    if continuation_mode in ("from_reasoning", "from_answer"):
        return continuation_mode
    return None


def _is_harmony_thinking_enabled(
    chat_template_kwargs: Mapping[str, Any] | None,
) -> bool:
    if not chat_template_kwargs:
        return True
    return chat_template_kwargs.get("enable_thinking") is not False


def get_harmony_completion_channel(
    chat_msgs: list | None = None,
    chat_template_kwargs: Mapping[str, Any] | None = None,
    *,
    continue_final_message: bool = False,
) -> Literal["analysis", "final"]:
    continuation_mode = _get_harmony_continuation_mode(chat_template_kwargs)
    if continue_final_message:
        if continuation_mode == "from_reasoning":
            return "analysis"
        if continuation_mode == "from_answer":
            return "final"
        if chat_msgs:
            last_msg = chat_msgs[-1]
            if not isinstance(last_msg, dict):
                last_msg = last_msg.model_dump(exclude_none=True)
            if last_msg.get("role") == "assistant":
                if flatten_chat_text_content(last_msg.get("content")):
                    return "final"
                if last_msg.get("reasoning"):
                    return "analysis"
    return "analysis" if _is_harmony_thinking_enabled(chat_template_kwargs) else "final"


def _render_open_harmony_channel(
    messages: list[Message],
    channel: Literal["analysis", "final"],
) -> list[int]:
    if _last_assistant_channel(messages) == channel:
        token_ids = _render_conversation_without_auto_drop(messages)
        return _strip_trailing_end_token(token_ids)

    token_ids = render_for_completion(messages)
    token_ids.extend(_encode_harmony(_harmony_channel_prefix(channel)))
    return token_ids


def render_for_chat_completion(
    messages: list[Message],
    *,
    chat_template_kwargs: Mapping[str, Any] | None = None,
    add_generation_prompt: bool = True,
    continue_final_message: bool = False,
) -> list[int]:
    if continue_final_message:
        channel = get_harmony_completion_channel(
            chat_template_kwargs=chat_template_kwargs,
            continue_final_message=True,
        )
        if _get_harmony_continuation_mode(chat_template_kwargs) is None:
            last_channel = _last_assistant_channel(messages)
            if last_channel in ("analysis", "final"):
                channel = last_channel

        if channel == "analysis":
            return _render_open_harmony_channel(messages, "analysis")

        if _last_assistant_channel(messages) == "final":
            token_ids = _render_conversation_without_auto_drop(messages)
            return _strip_trailing_end_token(token_ids)

        token_ids = _render_conversation_without_auto_drop(messages)
        token_ids.extend(_encode_harmony(_harmony_assistant_channel_prefix("final")))
        return token_ids

    if not add_generation_prompt:
        return _render_conversation_without_auto_drop(messages)

    channel = get_harmony_completion_channel(chat_template_kwargs=chat_template_kwargs)
    return _render_open_harmony_channel(messages, channel)


def should_preserve_last_assistant_analysis(
    chat_msgs: list,
    chat_template_kwargs: Mapping[str, Any] | None = None,
    *,
    continue_final_message: bool = False,
) -> bool:
    if not continue_final_message:
        return False
    if _get_harmony_continuation_mode(chat_template_kwargs) != "from_answer":
        return False
    if not chat_msgs:
        return False
    last_msg = chat_msgs[-1]
    if not isinstance(last_msg, dict):
        last_msg = last_msg.model_dump(exclude_none=True)
    return last_msg.get("role") == "assistant" and bool(last_msg.get("reasoning"))


def _starts_with_harmony_message_header(token_ids: Sequence[int]) -> bool:
    for prefix in (
        "<|start|>assistant",
        *[f"<|channel|>{channel}" for channel in HARMONY_CHANNELS],
    ):
        prefix_token_ids = _encode_harmony(prefix)
        if (
            len(token_ids) >= len(prefix_token_ids)
            and list(token_ids[: len(prefix_token_ids)]) == prefix_token_ids
        ):
            return True
    return False


def normalize_chat_output_with_prefill(
    token_ids: Sequence[int],
    *,
    chat_msgs: list | None = None,
    chat_template_kwargs: Mapping[str, Any] | None = None,
    continue_final_message: bool = False,
) -> list[int]:
    normalized_token_ids = list(token_ids)
    if _starts_with_harmony_message_header(normalized_token_ids):
        return normalized_token_ids

    channel = get_harmony_completion_channel(
        chat_msgs,
        chat_template_kwargs,
        continue_final_message=continue_final_message,
    )
    prefixed_token_ids = _encode_harmony(_harmony_channel_prefix(channel))
    prefixed_token_ids.extend(normalized_token_ids)
    return prefixed_token_ids


def parse_chat_output_with_prefill(
    token_ids: Sequence[int],
    *,
    chat_msgs: list | None = None,
    chat_template_kwargs: Mapping[str, Any] | None = None,
    continue_final_message: bool = False,
) -> tuple[str | None, str | None, bool]:
    return parse_chat_output(
        normalize_chat_output_with_prefill(
            token_ids,
            chat_msgs=chat_msgs,
            chat_template_kwargs=chat_template_kwargs,
            continue_final_message=continue_final_message,
        )
    )


def get_final_prefill_content(chat_msgs: list) -> str | None:
    if not chat_msgs:
        return None
    last_msg = chat_msgs[-1]
    if not isinstance(last_msg, dict):
        last_msg = last_msg.model_dump(exclude_none=True)
    if last_msg.get("role") != "assistant":
        return None
    return flatten_chat_text_content(last_msg.get("content")) or None


def get_stop_tokens_for_assistant_actions() -> list[int]:
    return get_encoding().stop_tokens_for_assistant_actions()


def get_streamable_parser_for_assistant() -> StreamableParser:
    return StreamableParser(get_encoding(), role=Role.ASSISTANT)


def parse_output_into_messages(token_ids: Iterable[int]) -> StreamableParser:
    parser = get_streamable_parser_for_assistant()
    for token_id in token_ids:
        parser.process(token_id)
    return parser


def parse_chat_output(
    token_ids: Sequence[int],
) -> tuple[str | None, str | None, bool]:
    """
    Parse the output of a Harmony chat completion into reasoning and final content.
    Note that when the `openai` tool parser is used, serving_chat only uses this
    for the reasoning content and gets the final content from the tool call parser.

    When the `openai` tool parser is not enabled, or when `GptOssReasoningParser` is
    in use,this needs to return the final content without any tool calls parsed.

    Empty reasoning or final content is returned as None instead of an empty string.
    """
    parser = parse_output_into_messages(token_ids)
    output_msgs = parser.messages
    is_tool_call = False  # TODO: update this when tool call is supported

    # Get completed messages from the parser
    # - analysis channel: hidden reasoning
    # - commentary channel without recipient (preambles): visible to user
    # - final channel: visible to user
    # - commentary with recipient (tool calls): handled separately by tool parser
    reasoning_texts = [
        msg.content[0].text for msg in output_msgs if msg.channel == "analysis"
    ]
    final_texts = [
        msg.content[0].text
        for msg in output_msgs
        if msg.channel == "final" or (msg.channel == "commentary" and not msg.recipient)
    ]

    # Extract partial messages from the parser
    if parser.current_channel == "analysis" and parser.current_content:
        reasoning_texts.append(parser.current_content)
    elif parser.current_channel == "final" and parser.current_content:
        final_texts.append(parser.current_content)
    elif (
        parser.current_channel == "commentary"
        and not parser.current_recipient
        and parser.current_content
    ):
        # Preambles (commentary without recipient) are visible to user
        final_texts.append(parser.current_content)

    # Flatten multiple messages into a single string
    reasoning: str | None = "\n".join(reasoning_texts)
    final_content: str | None = "\n".join(final_texts)

    # Return None instead of empty string since existing callers check for None
    reasoning = reasoning or None
    final_content = final_content or None

    return reasoning, final_content, is_tool_call
