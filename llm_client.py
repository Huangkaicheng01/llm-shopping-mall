import json
import os

from openai import OpenAI


def _llm_resolve() -> tuple[str, str, str]:
    """
    返回 (api_key, base_url, model)。
    配置了 LLM_API_BASE 时：密钥优先 LLM_API_KEY，模型优先 LLM_MODEL；未设模型且 host 含 dashscope 时默认 qwen-turbo。
    未配置 LLM_API_BASE 时：沿用 OPENAI_* 与 OPENAI_API_BASE；未传 base_url 时由 HTTP 客户端默认连 OpenAI 官服。
    """
    llm_base = (os.environ.get("LLM_API_BASE") or "").strip().rstrip("/")
    openai_base = (os.environ.get("OPENAI_API_BASE") or "").strip().rstrip("/")

    if llm_base:
        api_key = (os.environ.get("LLM_API_KEY") or "").strip()
        if not api_key:
            api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
        model = (os.environ.get("LLM_MODEL") or os.environ.get("OPENAI_MODEL") or "").strip()
        if not model:
            model = "qwen-turbo" if "dashscope" in llm_base.lower() else "gpt-4o-mini"
        return api_key, llm_base, model

    oa_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    lm_key = (os.environ.get("LLM_API_KEY") or "").strip()
    api_key = oa_key or lm_key
    model = (os.environ.get("OPENAI_MODEL") or os.environ.get("LLM_MODEL") or "gpt-4o-mini").strip()
    return api_key, openai_base, model


def llm_api_key() -> str:
    return _llm_resolve()[0]


def llm_api_base() -> str:
    return _llm_resolve()[1]


def llm_model() -> str:
    return _llm_resolve()[2]


def llm_call_json(messages: list[dict], *, temperature: float = 0.15, max_tokens: int = 800) -> dict:
    api_key, base, model = _llm_resolve()
    if not api_key:
        raise RuntimeError("未配置 API 密钥：请在 .env 中设置 OPENAI_API_KEY 或 LLM_API_KEY")

    lm_key = (os.environ.get("LLM_API_KEY") or "").strip()
    oa_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    oa_base = (os.environ.get("OPENAI_API_BASE") or "").strip()
    if not base and lm_key and not oa_key and not oa_base:
        raise RuntimeError(
            "已配置 LLM_API_KEY，但未配置 LLM_API_BASE（也未配置 OPENAI_API_BASE）。"
            "未指定网关时请求会发往 OpenAI 官方。使用通义请在 .env 增加：\n"
            "LLM_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

    kwargs = {"api_key": api_key}
    if base:
        kwargs["base_url"] = base
    client = OpenAI(**kwargs)
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
        temperature=temperature,
        max_tokens=max_tokens,
    )
    raw = (resp.choices[0].message.content or "").strip() or "{}"
    return json.loads(raw)


def llm_call_text(messages: list[dict], *, temperature: float = 0.2, max_tokens: int = 300) -> str:
    api_key, base, model = _llm_resolve()
    if not api_key:
        raise RuntimeError("未配置 API 密钥：请在 .env 中设置 OPENAI_API_KEY 或 LLM_API_KEY")
    kwargs = {"api_key": api_key}
    if base:
        kwargs["base_url"] = base
    client = OpenAI(**kwargs)
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()
