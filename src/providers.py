from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import dataclass


@dataclass
class LLMResponse:
    content: str
    elapsed_s: float
    usage: dict
    raw: object


class OpenAIProvider:
    def __init__(self, model: str, base_url: str, api_key: str, project: str | None = None):
        from openai import OpenAI

        self.model = model
        client_kwargs = {
            "base_url": base_url,
            "api_key": api_key,
            "timeout": 120,
        }
        if project:
            # Yandex needs gpt://<folder>/<model> URI
            self.model = f"gpt://{project}/{model}"
            client_kwargs["project"] = project
        self.client = OpenAI(**client_kwargs)

    def call_llm(self, messages: list[dict], **kwargs) -> LLMResponse:
        t0 = time.time()
        response = self.client.chat.completions.create(model=self.model, messages=messages, **kwargs)
        elapsed = round(time.time() - t0, 2)
        content = response.choices[0].message.content
        usage = response.usage.model_dump() if response.usage else {}
        return LLMResponse(content=content, elapsed_s=elapsed, usage=usage, raw=response)


class GigaChatProvider:
    """GigaChat via official Python SDK. Retries on 429 with backoff."""

    def __init__(self, model: str, base_url: str, api_key: str, scope: str = "GIGACHAT_API_PERS"):
        from gigachat import GigaChat

        self.model = model
        self.client = GigaChat(
            base_url=base_url,
            credentials=api_key,
            scope=scope,
            verify_ssl_certs=False,
            timeout=120,
        )

    def call_llm(self, messages: list[dict], **kwargs) -> LLMResponse:
        from gigachat.models import Chat, Messages, MessagesRole

        role_map = {
            "system": MessagesRole.SYSTEM,
            "user": MessagesRole.USER,
            "assistant": MessagesRole.ASSISTANT,
        }
        gc_messages = [
            Messages(role=role_map.get(m["role"], MessagesRole.USER), content=m["content"])
            for m in messages
        ]
        chat = Chat(model=self.model, messages=gc_messages, **kwargs)

        t0 = time.time()
        for attempt in range(5):
            try:
                response = self.client.chat(chat)
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


class LayaProvider:
    def __init__(self, model: str, subfolder: str | None = None):
        import laya
        import torch

        device = "mps" if torch.backends.mps.is_available() else None
        self.agent = laya.load(model, subfolder=subfolder, device=device)
        # States run ~1.1-1.7k tokens; checkpoint default is 1024, advertised up to 8k.
        self.agent.cfg["max_len"] = 2048

    def predict(self, state: dict, questions: dict) -> tuple[dict, float]:
        t0 = time.time()
        res = self.agent.predict(state, questions)
        return res["answers"], round(time.time() - t0, 2)


class JevProvider:
    """TypeSafe Jev: hosted typed-decision API, same schema as Laya."""

    def __init__(self, model: str, base_url: str, api_key: str):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key

    def predict(self, state: dict, questions: dict) -> tuple[dict, float]:
        body = json.dumps({
            "model": self.model,
            "state": state,
            "questions": questions,
        }, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            self.base_url.rstrip("/") + "/v1/systemone",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=120) as r:
            res = json.loads(r.read())
        return res["answers"], round(time.time() - t0, 2)


_PROVIDERS = {
    "openai-chat": OpenAIProvider,
    "gigachat": GigaChatProvider,
    "laya": LayaProvider,
    "jev": JevProvider,
}


def get_llm_provider(api: str, **kwargs):
    if api not in _PROVIDERS:
        raise ValueError(f"Unknown api type: {api}")
    return _PROVIDERS[api](**kwargs)
