"""
Prompt 模板加载与 messages 组装。

- intent_extract：自然语言找货 → 结构化筛选条件
- profile_preference_extract：论文「偏好抽取」场景 → UserProfile schema_version 1.0 JSON
"""
import json
from pathlib import Path
from string import Template


_PROMPT_DIR = Path(__file__).resolve().parent / "prompts"


def _load_prompt_template(name: str) -> str:
    p = _PROMPT_DIR / f"{name}.txt"
    if not p.is_file():
        raise FileNotFoundError(f"缺少 Prompt 模板文件：{p}")
    return p.read_text(encoding="utf-8")


def build_intent_extract_messages(*, user_text: str, allowed_categories: list[str]) -> list[dict[str, str]]:
    tpl = _load_prompt_template("intent_extract")
    system_text = tpl.format(allowed_categories_json=json.dumps(allowed_categories, ensure_ascii=False))
    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]


def build_profile_preference_extract_messages(
    *,
    allowed_categories: list[str],
    selected_interest_tags: str = "",
    interaction_history: str = "",
    product_information: str = "",
    reviews_text: str = "",
) -> list[dict[str, str]]:
    """
    偏好抽取：system 为论文用英文指令 + 与仓库 `schemas/user_profile.schema.json` 对齐的输出约定；
    user 为三段证据文本（可为空，空则占位说明）。
    """
    tpl = Template(_load_prompt_template("profile_preference_extract"))
    system_text = tpl.substitute(
        allowed_categories_json=json.dumps(allowed_categories, ensure_ascii=False),
    )

    def _section(title: str, body: str) -> str:
        b = (body or "").strip()
        if not b:
            b = "(No data provided.)"
        return f"### {title}\n{b}"

    user_body = "\n\n".join(
        [
            "Extract user preferences from the following evidence:",
            _section("Selected interest tags (explicit preferences)", selected_interest_tags),
            _section("Interaction history", interaction_history),
            _section("Product information (catalog snippets)", product_information),
            _section("User reviews", reviews_text),
        ]
    )
    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_body},
    ]


def build_profile_summary_messages(*, history_profiles_text: str) -> list[dict[str, str]]:
    system_text = (
        "你是电商用户画像分析助手。请基于用户历史多版画像，"
        "输出一句中文总结（不超过60字），概括该用户的长期偏好、短期意图和价格敏感度。"
        "只输出这句话，不要加编号、引号或其它说明。"
    )
    user_text = (
        "以下是该用户最近多版画像（JSON/摘要），请融合后给出一句话画像描述：\n\n"
        + (history_profiles_text.strip() or "(No profile history provided.)")
    )
    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]
