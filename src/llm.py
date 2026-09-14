"""LLM call abstraction. Config-driven via JSONL files.

Supports three API types:
  - openai-chat       → OpenAI SDK, chat.completions.create()
  - openai-responses  → OpenAI SDK, responses.create()
  - gigachat          → GigaChat Python SDK (Sber GigaChat)

Tool-calling support:
  - call_llm_with_tools() runs a multi-turn tool loop.
  - Per-API extraction of tool calls from responses.
  - Per-API continuation with tool results.
  - Trajectory tracking on LLMResponse.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path


# Config fields that are NOT passed through as API kwargs
_KNOWN_FIELDS = frozenset({
    "name", "api", "model", "base_url", "api_key", "reasoning",
    "price_1m_input", "price_1m_output", "currency", "provider_website",
    "system_prompt_override", "temperature", "max_output_tokens",
    "scope", "project",
})


@dataclass
class LLMResponse:
    content: str
    elapsed_s: float
    usage: dict
    raw: object
    # Tool-calling trajectory: list of {role, content, tool_calls?, tool_call_id?}
    trajectory: list[dict] = field(default_factory=list)
    # Tool metrics: {calls, errors, latency_s, documents_touched}
    tool_metrics: dict = field(default_factory=dict)


def load_config(path: str | Path) -> list[dict]:
    """Read a JSONL config file, return list of config dicts."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def get_model_config(name: str, configs: list[dict]) -> dict:
    """Look up a model/judge by name in a list of config dicts."""
    for c in configs:
        if c["name"] == name:
            return c
    raise KeyError(f"Model '{name}' not found in config")


def resolve_env(value: str) -> str:
    """If value is an env var name, resolve it; otherwise return as-is."""
    if value in os.environ:
        return os.environ[value]
    return value


def _extra_kwargs(config: dict) -> dict:
    """Extract pass-through kwargs from config (fields not in _KNOWN_FIELDS)."""
    return {k: v for k, v in config.items() if k not in _KNOWN_FIELDS}


def call_llm(config: dict, messages: list[dict], **kwargs) -> LLMResponse:
    """Call an LLM based on config dict. Routes by config['api']."""
    api = config["api"]
    if api == "openai-chat":
        return _call_openai_chat(config, messages, **kwargs)
    if api == "openai-responses":
        return _call_openai_responses(config, messages, **kwargs)
    if api == "gigachat":
        return _call_gigachat(config, messages, **kwargs)
    raise ValueError(f"Unknown api type: {api}")


def _call_openai_chat(
    config: dict,
    messages: list[dict],
    **kwargs,
) -> LLMResponse:
    """OpenAI-compatible chat completions (NeuralDeep, Yandex chat, etc)."""
    from openai import OpenAI

    client_kwargs = {
        "base_url": resolve_env(config["base_url"]),
        "api_key": resolve_env(config["api_key"]),
        "timeout": 120,
    }
    if "project" in config:
        client_kwargs["project"] = resolve_env(config["project"])
    client = OpenAI(**client_kwargs)

    model = config.get("model", config["name"])
    call_kwargs = dict(messages=messages, model=model)
    call_kwargs["temperature"] = config.get("temperature", 0.0)
    if "max_output_tokens" in config:
        call_kwargs["max_tokens"] = config["max_output_tokens"]
    if config.get("reasoning"):
        call_kwargs["reasoning_effort"] = config["reasoning"]
    call_kwargs.update(_extra_kwargs(config))
    call_kwargs.update(kwargs)

    t0 = time.time()
    response = client.chat.completions.create(**call_kwargs)
    elapsed = round(time.time() - t0, 2)
    content = response.choices[0].message.content
    usage = response.usage.model_dump() if response.usage else {}
    return LLMResponse(content=content, elapsed_s=elapsed, usage=usage, raw=response)


def _call_openai_responses(
    config: dict,
    messages: list[dict],
    **kwargs,
) -> LLMResponse:
    """OpenAI Responses API (YandexGPT 5.1).

    Maps chat messages → Responses API format:
      system message → instructions
      user/assistant messages → input
    """
    from openai import OpenAI

    client_kwargs = {
        "base_url": resolve_env(config["base_url"]),
        "api_key": resolve_env(config["api_key"]),
        "timeout": 120,
    }
    if "project" in config:
        client_kwargs["project"] = resolve_env(config["project"])
    client = OpenAI(**client_kwargs)

    # Yandex needs gpt://<folder>/<model> URI
    model = config.get("model", config["name"])
    if "project" in config:
        folder = resolve_env(config["project"])
        model = f"gpt://{folder}/{model}"

    instructions = ""
    input_msgs = []
    for msg in messages:
        if msg["role"] == "system":
            instructions = msg["content"]
        else:
            input_msgs.append({"role": msg["role"], "content": msg["content"]})

    call_kwargs = dict(model=model, instructions=instructions, input=input_msgs)
    call_kwargs["temperature"] = config.get("temperature", 0.0)
    if "max_output_tokens" in config:
        call_kwargs["max_output_tokens"] = config["max_output_tokens"]
    if config.get("reasoning"):
        call_kwargs["reasoning"] = {"effort": config["reasoning"]}
    call_kwargs.update(_extra_kwargs(config))
    call_kwargs.update(kwargs)

    t0 = time.time()
    response = client.responses.create(**call_kwargs)
    elapsed = round(time.time() - t0, 2)
    content = response.output_text
    usage = {
        "prompt_tokens": getattr(response.usage, "input_tokens", 0) if response.usage else 0,
        "completion_tokens": getattr(response.usage, "output_tokens", 0) if response.usage else 0,
        "total_tokens": (getattr(response.usage, "input_tokens", 0) + getattr(response.usage, "output_tokens", 0)) if response.usage else 0,
    }
    return LLMResponse(content=content, elapsed_s=elapsed, usage=usage, raw=response)


_gigachat_clients: dict[str, object] = {}


def _get_gigachat_client(config: dict):
    """Lazy GigaChat client per base_url+credentials."""
    from gigachat import GigaChat

    key = (config["base_url"], config["api_key"])
    if key not in _gigachat_clients:
        _gigachat_clients[key] = GigaChat(
            base_url=resolve_env(config["base_url"]),
            credentials=resolve_env(config["api_key"]),
            scope=resolve_env(config.get("scope", "GIGACHAT_API_PERS")),
            verify_ssl_certs=False,
            timeout=120,
        )
    return _gigachat_clients[key]


def _call_gigachat(
    config: dict,
    messages: list[dict],
    **kwargs,
) -> LLMResponse:
    """GigaChat via official Python SDK. Retries on 429 with backoff."""
    from gigachat.models import Chat, Messages, MessagesRole

    client = _get_gigachat_client(config)
    role_map = {
        "system": MessagesRole.SYSTEM,
        "user": MessagesRole.USER,
        "assistant": MessagesRole.ASSISTANT,
    }
    gc_messages = [
        Messages(role=role_map.get(m["role"], MessagesRole.USER), content=m["content"])
        for m in messages
    ]
    model = config.get("model", config["name"])
    reasoning = config.get("reasoning")
    chat_kwargs = dict(
        model=model,
        messages=gc_messages,
        temperature=config.get("temperature", 0.0),
        max_tokens=config.get("max_output_tokens", 8192 if not reasoning else 16384),
    )
    if reasoning:
        chat_kwargs["reasoning_effort"] = reasoning
    chat_kwargs.update(_extra_kwargs(config))
    chat_kwargs.update(kwargs)
    chat = Chat(**chat_kwargs)

    t0 = time.time()
    for attempt in range(5):
        try:
            response = client.chat(chat)
            elapsed = round(time.time() - t0, 2)
            content = response.choices[0].message.content
            # Retry on empty content (reasoning models sometimes return None)
            if not content and attempt < 4:
                time.sleep(2 ** (attempt + 1))
                continue
            usage = response.usage.model_dump() if response.usage else {}
            return LLMResponse(content=content, elapsed_s=elapsed, usage=usage, raw=response)
        except Exception as e:
            if "429" in str(e) and attempt < 4:
                wait = 2 ** (attempt + 1)  # 2, 4, 8, 16 seconds
                time.sleep(wait)
                continue
            raise


# ---------------------------------------------------------------------------
# Tool-calling support
# ---------------------------------------------------------------------------

def call_llm_with_tools(
    config: dict,
    messages: list[dict],
    tools: list[dict],
    dispatch_fn,
    ctx: dict,
    max_tool_calls: int = 8,
) -> LLMResponse:
    """Run a multi-turn tool loop. Routes by config['api'].

    Args:
        config: Model config dict.
        messages: Initial messages (system + user).
        tools: Tool schemas (format depends on API type).
        dispatch_fn: Function(name, args, ctx) -> str result.
        ctx: Context dict passed to dispatch_fn.
        max_tool_calls: Hard limit on tool calls before forcing a final answer.

    Returns:
        LLMResponse with content, trajectory, and tool_metrics.
    """
    api = config["api"]
    if api == "openai-chat":
        return _tool_loop_openai_chat(config, messages, tools, dispatch_fn, ctx, max_tool_calls)
    if api == "openai-responses":
        return _tool_loop_openai_responses(config, messages, tools, dispatch_fn, ctx, max_tool_calls)
    if api == "gigachat":
        return _tool_loop_gigachat(config, messages, tools, dispatch_fn, ctx, max_tool_calls)
    raise ValueError(f"Unknown api type: {api}")


def _run_tool_calls(
    tool_calls: list[dict],
    dispatch_fn,
    ctx: dict,
    metrics: dict,
) -> list[dict]:
    """Execute tool calls and return result messages. Updates metrics in-place."""
    results = []
    for tc in tool_calls:
        name = tc["name"]
        args = tc["args"]
        tc_id = tc["id"]

        t0 = time.time()
        result = dispatch_fn(name, args, ctx)
        latency = round(time.time() - t0, 3)

        metrics["calls"] += 1
        metrics["latency_s"] += latency
        if result.startswith("ERROR"):
            metrics["errors"] += 1
        else:
            # Track documents touched
            if name in ("read_document", "list_sections", "search_by_article"):
                doc_id = args.get("document_id") or args.get("code", "")
                if doc_id:
                    metrics["documents_touched"].add(doc_id)

        results.append({
            "role": "tool",
            "tool_call_id": tc_id,
            "name": name,
            "content": result,
        })
    return results


def _tool_loop_openai_chat(
    config: dict,
    messages: list[dict],
    tools: list[dict],
    dispatch_fn,
    ctx: dict,
    max_tool_calls: int,
) -> LLMResponse:
    """Tool loop for OpenAI-compatible chat completions API."""
    from openai import OpenAI

    client_kwargs = {
        "base_url": resolve_env(config["base_url"]),
        "api_key": resolve_env(config["api_key"]),
        "timeout": 120,
    }
    if "project" in config:
        client_kwargs["project"] = resolve_env(config["project"])
    client = OpenAI(**client_kwargs)

    model = config.get("model", config["name"])
    call_kwargs = dict(model=model, temperature=config.get("temperature", 0.0))
    if "max_output_tokens" in config:
        call_kwargs["max_tokens"] = config["max_output_tokens"]
    if config.get("reasoning"):
        call_kwargs["reasoning_effort"] = config["reasoning"]
    call_kwargs.update(_extra_kwargs(config))

    trajectory = list(messages)
    metrics = {"calls": 0, "errors": 0, "latency_s": 0.0, "documents_touched": set()}
    total_usage = {}
    t0 = time.time()

    for round_num in range(max_tool_calls + 1):
        is_final = round_num >= max_tool_calls
        api_kwargs = dict(call_kwargs)
        api_kwargs["messages"] = trajectory
        if not is_final:
            api_kwargs["tools"] = tools

        response = client.chat.completions.create(**api_kwargs)
        if response.usage:
            u = response.usage.model_dump()
            for k, v in u.items():
                if isinstance(v, (int, float)) and v is not None:
                    total_usage[k] = total_usage.get(k, 0) + v

        msg = response.choices[0].message
        content = msg.content or ""

        # Extract tool calls
        tool_calls = []
        if hasattr(msg, "tool_calls") and msg.tool_calls and not is_final:
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                except (json.JSONDecodeError, TypeError):
                    args = {}
                tool_calls.append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "args": args,
                })

        # Add assistant message to trajectory
        assistant_msg = {"role": "assistant", "content": content}
        if tool_calls:
            assistant_msg["tool_calls"] = [
                {"id": tc["id"], "type": "function",
                 "function": {"name": tc["name"], "arguments": json.dumps(tc["args"], ensure_ascii=False)}}
                for tc in tool_calls
            ]
        trajectory.append(assistant_msg)

        if not tool_calls or is_final:
            # No more tool calls — we have the final answer
            elapsed = round(time.time() - t0, 2)
            return LLMResponse(
                content=content,
                elapsed_s=elapsed,
                usage=total_usage,
                raw=response,
                trajectory=trajectory,
                tool_metrics={
                    "calls": metrics["calls"],
                    "errors": metrics["errors"],
                    "latency_s": round(metrics["latency_s"], 3),
                    "documents_touched": sorted(metrics["documents_touched"]),
                },
            )

        # Execute tool calls
        tool_results = _run_tool_calls(tool_calls, dispatch_fn, ctx, metrics)
        trajectory.extend(tool_results)

    # Shouldn't reach here, but just in case
    elapsed = round(time.time() - t0, 2)
    return LLMResponse(
        content="",
        elapsed_s=elapsed,
        usage=total_usage,
        raw=None,
        trajectory=trajectory,
        tool_metrics={
            "calls": metrics["calls"],
            "errors": metrics["errors"],
            "latency_s": round(metrics["latency_s"], 3),
            "documents_touched": sorted(metrics["documents_touched"]),
        },
    )


def _tool_loop_openai_responses(
    config: dict,
    messages: list[dict],
    tools: list[dict],
    dispatch_fn,
    ctx: dict,
    max_tool_calls: int,
) -> LLMResponse:
    """Tool loop for OpenAI Responses API (YandexGPT 5.1).

    The Responses API uses a different format:
    - system → instructions
    - user/assistant → input
    - tool calls → function_call items in output
    - tool results → function_call_output items in input
    """
    from openai import OpenAI

    client_kwargs = {
        "base_url": resolve_env(config["base_url"]),
        "api_key": resolve_env(config["api_key"]),
        "timeout": 120,
    }
    if "project" in config:
        client_kwargs["project"] = resolve_env(config["project"])
    client = OpenAI(**client_kwargs)

    model = config.get("model", config["name"])
    if "project" in config:
        folder = resolve_env(config["project"])
        model = f"gpt://{folder}/{model}"

    # Split messages into instructions + input
    instructions = ""
    input_items = []
    for msg in messages:
        if msg["role"] == "system":
            instructions = msg["content"]
        else:
            input_items.append({"role": msg["role"], "content": msg["content"]})

    call_kwargs = dict(model=model, temperature=config.get("temperature", 0.0))
    if "max_output_tokens" in config:
        call_kwargs["max_output_tokens"] = config["max_output_tokens"]
    if config.get("reasoning"):
        call_kwargs["reasoning"] = {"effort": config["reasoning"]}
    call_kwargs.update(_extra_kwargs(config))

    # Convert tool schemas to Responses API format (flat, not nested under "function")
    responses_tools = []
    for t in tools:
        if "function" in t:
            f = t["function"]
            responses_tools.append({
                "type": "function",
                "name": f["name"],
                "description": f.get("description", ""),
                "parameters": f.get("parameters", {}),
            })
        else:
            responses_tools.append(t)

    trajectory = list(messages)
    metrics = {"calls": 0, "errors": 0, "latency_s": 0.0, "documents_touched": set()}
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    t0 = time.time()

    for round_num in range(max_tool_calls + 1):
        is_final = round_num >= max_tool_calls
        api_kwargs = dict(call_kwargs)
        api_kwargs["instructions"] = instructions
        api_kwargs["input"] = input_items
        if not is_final:
            api_kwargs["tools"] = responses_tools

        response = client.responses.create(**api_kwargs)
        if response.usage:
            total_usage["prompt_tokens"] += getattr(response.usage, "input_tokens", 0) or 0
            total_usage["completion_tokens"] += getattr(response.usage, "output_tokens", 0) or 0
            total_usage["total_tokens"] = total_usage["prompt_tokens"] + total_usage["completion_tokens"]

        content = response.output_text or ""

        # Extract tool calls from response output
        tool_calls = []
        if not is_final and hasattr(response, "output"):
            for item in response.output:
                if getattr(item, "type", None) == "function_call":
                    try:
                        args = json.loads(item.arguments) if item.arguments else {}
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                    tool_calls.append({
                        "id": item.call_id,
                        "name": item.name,
                        "args": args,
                    })

        # Add assistant content to trajectory
        if content:
            trajectory.append({"role": "assistant", "content": content})

        if not tool_calls or is_final:
            elapsed = round(time.time() - t0, 2)
            return LLMResponse(
                content=content,
                elapsed_s=elapsed,
                usage=total_usage,
                raw=response,
                trajectory=trajectory,
                tool_metrics={
                    "calls": metrics["calls"],
                    "errors": metrics["errors"],
                    "latency_s": round(metrics["latency_s"], 3),
                    "documents_touched": sorted(metrics["documents_touched"]),
                },
            )

        # Add function calls to input_items for the Responses API
        for tc in tool_calls:
            input_items.append({
                "type": "function_call",
                "call_id": tc["id"],
                "name": tc["name"],
                "arguments": json.dumps(tc["args"], ensure_ascii=False),
            })

        # Execute tool calls
        tool_results = _run_tool_calls(tool_calls, dispatch_fn, ctx, metrics)

        # Add tool results to input_items
        for i, tc in enumerate(tool_calls):
            input_items.append({
                "type": "function_call_output",
                "call_id": tc["id"],
                "output": tool_results[i]["content"],
            })

    elapsed = round(time.time() - t0, 2)
    return LLMResponse(
        content="",
        elapsed_s=elapsed,
        usage=total_usage,
        raw=None,
        trajectory=trajectory,
        tool_metrics={
            "calls": metrics["calls"],
            "errors": metrics["errors"],
            "latency_s": round(metrics["latency_s"], 3),
            "documents_touched": sorted(metrics["documents_touched"]),
        },
    )


def _tool_loop_gigachat(
    config: dict,
    messages: list[dict],
    tools: list[dict],
    dispatch_fn,
    ctx: dict,
    max_tool_calls: int,
) -> LLMResponse:
    """Tool loop for GigaChat via official Python SDK.

    GigaChat may not support parallel tool calls — execute one at a time.
    If GigaChat doesn't return structured tool calls, treat output as final answer.
    """
    from gigachat.models import Chat, Messages, MessagesRole

    client = _get_gigachat_client(config)
    role_map = {
        "system": MessagesRole.SYSTEM,
        "user": MessagesRole.USER,
        "assistant": MessagesRole.ASSISTANT,
    }

    model = config.get("model", config["name"])
    reasoning = config.get("reasoning")

    # Convert tool schemas to GigaChat format
    gc_tools = []
    for t in tools:
        if "function" in t:
            f = t["function"]
            gc_tools.append({
                "name": f["name"],
                "description": f.get("description", ""),
                "parameters": f.get("parameters", {}),
            })
        else:
            gc_tools.append(t)

    trajectory = list(messages)
    metrics = {"calls": 0, "errors": 0, "latency_s": 0.0, "documents_touched": set()}
    total_usage = {}
    t0 = time.time()

    for round_num in range(max_tool_calls + 1):
        is_final = round_num >= max_tool_calls

        # Convert trajectory to GigaChat messages
        gc_messages = []
        for msg in trajectory:
            role = role_map.get(msg["role"], MessagesRole.USER)
            gc_messages.append(Messages(role=role, content=msg.get("content", "")))

        chat_kwargs = dict(
            model=model,
            messages=gc_messages,
            temperature=config.get("temperature", 0.0),
            max_tokens=config.get("max_output_tokens", 8192 if not reasoning else 16384),
        )
        if reasoning:
            chat_kwargs["reasoning_effort"] = reasoning
        if not is_final and gc_tools:
            chat_kwargs["tools"] = gc_tools
        chat_kwargs.update(_extra_kwargs(config))
        chat = Chat(**chat_kwargs)

        # Retry on 429
        response = None
        for attempt in range(5):
            try:
                response = client.chat(chat)
                break
            except Exception as e:
                if "429" in str(e) and attempt < 4:
                    time.sleep(2 ** (attempt + 1))
                    continue
                raise

        if response is None:
            raise RuntimeError("GigaChat failed after retries")

        if response.usage:
            u = response.usage.model_dump()
            for k, v in u.items():
                if isinstance(v, (int, float)) and v is not None:
                    total_usage[k] = total_usage.get(k, 0) + v

        msg = response.choices[0].message
        content = msg.content or ""

        # Extract tool calls — GigaChat uses functions list
        tool_calls = []
        if hasattr(msg, "functions") and msg.functions and not is_final:
            for fc in msg.functions:
                # GigaChat may return args as dict or string
                args = fc.arguments if isinstance(fc.arguments, dict) else {}
                if isinstance(fc.arguments, str):
                    try:
                        args = json.loads(fc.arguments)
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                tool_calls.append({
                    "id": getattr(fc, "id", f"call_{round_num}"),
                    "name": fc.name,
                    "args": args,
                })

        # Add assistant message to trajectory
        trajectory.append({"role": "assistant", "content": content})

        if not tool_calls or is_final:
            # No tool calls or hit limit — return final answer
            elapsed = round(time.time() - t0, 2)
            return LLMResponse(
                content=content,
                elapsed_s=elapsed,
                usage=total_usage,
                raw=response,
                trajectory=trajectory,
                tool_metrics={
                    "calls": metrics["calls"],
                    "errors": metrics["errors"],
                    "latency_s": round(metrics["latency_s"], 3),
                    "documents_touched": sorted(metrics["documents_touched"]),
                },
            )

        # Execute tool calls one at a time (GigaChat may not support parallel)
        for tc in tool_calls:
            tool_results = _run_tool_calls([tc], dispatch_fn, ctx, metrics)
            # Add to trajectory
            trajectory.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "name": tc["name"],
                "content": tool_results[0]["content"],
            })

    elapsed = round(time.time() - t0, 2)
    return LLMResponse(
        content="",
        elapsed_s=elapsed,
        usage=total_usage,
        raw=None,
        trajectory=trajectory,
        tool_metrics={
            "calls": metrics["calls"],
            "errors": metrics["errors"],
            "latency_s": round(metrics["latency_s"], 3),
            "documents_touched": sorted(metrics["documents_touched"]),
        },
    )
