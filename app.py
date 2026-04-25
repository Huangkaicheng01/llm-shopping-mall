"""
购物商城演示应用：Flask 商城；登录用户「为您推荐」按 SQLite 兴趣标签（商品类目）权重选品，
LLM 找货与语义兜底等仍使用 BERT 向量与余弦相似度。
"""
import json
import os
import sqlite3
from collections import defaultdict
from functools import wraps
from pathlib import Path

from dotenv import load_dotenv
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    Response,
    abort,
    jsonify,
    session,
    flash,
)
import pandas as pd
from werkzeug.security import check_password_hash, generate_password_hash

from auth_db import (
    bump_user_tag_for_category,
    create_user,
    get_latest_user_profile,
    get_password_hash_for_login,
    get_user_by_email,
    get_user_by_id,
    init_auth_db,
    list_user_tags,
    list_recent_user_profiles,
    replace_user_interest_tags,
    save_user_profile,
    set_interest_onboarding_done,
    update_user_profile_summary,
)
from catalog_db import (
    catalog_db_path,
    init_catalog,
    insert_product_review,
    list_product_reviews,
    load_products_dataframe,
    seed_demo_product_reviews_if_empty,
    should_reseed_from_env,
)
from llm_client import llm_api_base, llm_api_key, llm_call_json, llm_call_text
from prompt_service import (
    build_intent_extract_messages,
    build_profile_preference_extract_messages,
    build_profile_summary_messages,
)
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import numpy as np  # 向量运算、堆叠与 reshape

app = Flask(__name__)

# 项目根目录，用于开发模式下扫描模板与静态资源修改时间
_ROOT = Path(__file__).resolve().parent

# 从项目根目录的 .env 加载环境变量（不覆盖已在系统中设置的同名变量）
load_dotenv(_ROOT / '.env')

app.secret_key = (os.environ.get('SECRET_KEY') or '').strip() or 'dev-only-set-SECRET_KEY-in-env-for-production'

_CATALOG_DB = catalog_db_path(_ROOT)
init_catalog(_ROOT, reseed=should_reseed_from_env())
init_auth_db(_CATALOG_DB)
seed_demo_product_reviews_if_empty(_CATALOG_DB)


def get_current_user():
    """当前登录用户 dict（含 user_id, email, name），未登录返回 None。"""
    uid = session.get('user_id')
    if not uid:
        return None
    u = get_user_by_id(_CATALOG_DB, int(uid))
    if not u:
        session.pop('user_id', None)
    return u


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get('user_id'):
            if request.path.startswith('/api/'):
                return jsonify(ok=False, login_required=True), 401
            flash('请先登录后再继续。', 'info')
            return redirect(url_for('login', next=request.path))
        return view(*args, **kwargs)

    return wrapped


@app.context_processor
def inject_current_user():
    return {'current_user': get_current_user()}


def _dev_newest_asset_mtime():
    """开发用：取 templates/ 与 static/ 下所有文件的最新修改时间，用于前端热刷新判断。"""
    newest = 0.0
    for sub in ('templates', 'static'):
        d = _ROOT / sub
        if not d.is_dir():
            continue
        for f in d.rglob('*'):
            if f.is_file():
                try:
                    newest = max(newest, f.stat().st_mtime)
                except OSError:
                    pass
    return newest


def load_products():
    """
    从 SQLite 商品库加载（instance/catalog.db）。
    首次空库或 CATALOG_RESEED=1 时执行 data/catalog_seed.sql；日常在库内改数据后重启应用。
    列：product_id, title, category, description, price, image（image 为相对 static/ 的路径）。
    """
    return load_products_dataframe(_CATALOG_DB)


# 商品数据：SQLite，见 catalog_db.py、instance/catalog.db、data/catalog_seed.sql
products = load_products()

# ---------------------------------------------------------------------------
# 用户行为日志：点击(click) / 购买(purchase)；内存中追加，重启后清空
# ---------------------------------------------------------------------------
user_activity = pd.DataFrame({
    'user_id': [],
    'product_id': [],
    'action': []
})

# 「我不喜欢」屏蔽的类目（按用户）；收藏的商品 ID（演示用内存数据，重启清空）
disliked_categories = defaultdict(set)
user_favorites = defaultdict(set)


def reset_in_memory_user_state(user_id: int) -> None:
    """登录或注册成功后清空该用户在内存中的浏览、屏蔽类目与收藏，避免沿用本会话中的旧演示数据。"""
    global user_activity
    uid = int(user_id)
    user_activity = user_activity[user_activity['user_id'] != uid].copy()
    disliked_categories.pop(uid, None)
    user_favorites.pop(uid, None)


# 中文商品描述请用中文 BERT；bert-base-uncased 面向英文，向量不可靠，易出现「笔记本→牛奶」等跨类误推
# 若运行环境缺少 PyTorch，则回退到轻量级哈希向量，保证应用可启动（语义质量会下降）。
try:
    model = pipeline('feature-extraction', model='bert-base-chinese')
except Exception:
    model = None


def get_embeddings(text):
    """对单条文本做特征提取，返回该句对应的一维向量（取 [CLS] 或首 token 表示，与 pipeline 输出一致）。"""
    if model is not None:
        return model(text)[0][0]
    v = np.zeros(256, dtype=np.float64)
    for tok in str(text or '').lower().split():
        v[hash(tok) % v.shape[0]] += 1.0
    n = np.linalg.norm(v)
    if n > 0:
        v = v / n
    return v.tolist()


# 为每个商品预计算描述向量，供推荐与相似度计算复用
products['embeddings'] = products['description'].apply(get_embeddings)


def _llm_api_key() -> str:
    return llm_api_key()


def _llm_api_base() -> str:
    return llm_api_base()


def _normalize_llm_categories(raw_list, allowed):
    allowed_set = set(allowed)
    out = []
    for c in raw_list or []:
        if not isinstance(c, str):
            continue
        c = c.strip()
        if not c:
            continue
        if c in allowed_set:
            out.append(c)
            continue
        for a in allowed:
            if c in a or a in c:
                out.append(a)
                break
    return list(dict.fromkeys(out))


def _call_llm_for_filters(user_text: str) -> dict:
    allowed = sorted(products['category'].unique().tolist())
    messages = build_intent_extract_messages(
        user_text=user_text[:4000],
        allowed_categories=allowed,
    )
    data = llm_call_json(
        messages=messages,
        temperature=0.15,
        max_tokens=800,
    )
    cats = _normalize_llm_categories(data.get('categories'), allowed)
    kws = []
    for k in data.get('keywords') or []:
        if isinstance(k, str) and k.strip():
            kws.append(k.strip())
    def _num(v):
        if v is None or v == '':
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    return {
        'categories': cats,
        'keywords': kws,
        'price_min': _num(data.get('price_min')),
        'price_max': _num(data.get('price_max')),
        'explain': (data.get('explain') or '') if isinstance(data.get('explain'), str) else '',
    }


def _apply_llm_filters_to_products(filters: dict) -> pd.DataFrame:
    df = products
    cats = filters.get('categories') or []
    if cats:
        df = df[df['category'].isin(cats)]
    for kw in filters.get('keywords') or []:
        if not kw:
            continue
        mask = (
            df['title'].str.contains(kw, case=False, na=False, regex=False)
            | df['description'].str.contains(kw, case=False, na=False, regex=False)
        )
        df = df[mask]
    lo, hi = filters.get('price_min'), filters.get('price_max')
    if lo is not None:
        df = df[df['price'] >= lo]
    if hi is not None:
        df = df[df['price'] <= hi]
    return df


def _semantic_rank_by_query(query_text: str, top_n: int = 10) -> pd.DataFrame:
    """用户整句编码后与商品描述向量比相似度，作兜底排序。"""
    q = (query_text or '').strip()[:512]
    if not q:
        return products.iloc[0:0].copy()
    qv = np.asarray(get_embeddings(q), dtype=np.float64).reshape(1, -1)
    scored = []
    for _, row in products.iterrows():
        emb = np.asarray(row['embeddings'], dtype=np.float64).reshape(1, -1)
        s = float(cosine_similarity(qv, emb)[0][0])
        scored.append((int(row['product_id']), s))
    scored.sort(key=lambda x: x[1], reverse=True)
    ids = [x[0] for x in scored[:top_n]]
    return products[products['product_id'].isin(ids)]


def _products_df_to_json_list(df: pd.DataFrame, limit: int = 10):
    out = []
    for _, r in df.head(limit).iterrows():
        out.append({
            'product_id': int(r['product_id']),
            'title': str(r['title']),
            'category': str(r['category']),
            'description': str(r['description']),
            'price': float(r['price']),
            'image': str(r['image']),
        })
    return out


def _build_interaction_history_text(user_id: int, limit: int = 50) -> str:
    rows = user_activity[user_activity['user_id'] == int(user_id)].tail(limit)
    if rows.empty:
        return ''
    lines = []
    for _, r in rows.iterrows():
        pid = int(r['product_id'])
        action = str(r['action'])
        title_row = products.loc[products['product_id'] == pid, 'title']
        title = str(title_row.iloc[0]) if not title_row.empty else '未知商品'
        lines.append(f'{action}: product_id={pid}, title={title}')
    return '\n'.join(lines)


def _build_product_information_text(user_id: int, limit: int = 30) -> str:
    rows = user_activity[user_activity['user_id'] == int(user_id)]
    if rows.empty:
        return ''
    ids = [int(x) for x in rows['product_id'].tolist()]
    # 按最近交互顺序去重，截断以控制 token。
    dedup_ids = list(dict.fromkeys(reversed(ids)))
    selected = dedup_ids[:limit]
    lines = []
    for pid in selected:
        p = products.loc[products['product_id'] == pid]
        if p.empty:
            continue
        row = p.iloc[0]
        lines.append(
            f"product_id={int(row['product_id'])} | title={row['title']} | category={row['category']} | "
            f"price={float(row['price']):.2f} | description={row['description']}"
        )
    return '\n'.join(lines)


def _build_reviews_text(user_id: int, limit: int = 30) -> str:
    with sqlite3.connect(_CATALOG_DB) as conn:
        rows = conn.execute(
            """
            SELECT r.review_id, r.product_id, r.body, r.created_at, p.title
            FROM product_reviews r
            JOIN products p ON p.product_id = r.product_id
            WHERE r.user_id = ?
            ORDER BY r.review_id DESC
            LIMIT ?
            """,
            (int(user_id), int(limit)),
        ).fetchall()
    if not rows:
        return ''
    lines = []
    for rid, pid, body, created_at, title in rows:
        lines.append(f'review_id={rid}, product_id={pid}, title={title}, at={created_at}, body={body}')
    return '\n'.join(lines)


def _build_selected_interest_tags_text(user_id: int) -> str:
    """显式兴趣标签证据：来自 user_tags（含权重与更新时间）。"""
    rows = list_user_tags(_CATALOG_DB, int(user_id))
    if not rows:
        return ''
    lines = []
    for r in rows:
        tag = str(r.get('tag') or '').strip()
        weight = int(r.get('weight') or 0)
        updated = str(r.get('updated_at') or '')
        lines.append(f'tag={tag}, weight={weight}, updated_at={updated}')
    return '\n'.join(lines)


def _clip_preview(text: str, max_chars: int = 1200) -> str:
    s = (text or '').strip()
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + '\n...(truncated)'


def _build_profile_history_text(user_id: int, limit: int = 8) -> str:
    rows = list_recent_user_profiles(_CATALOG_DB, int(user_id), limit=limit)
    if not rows:
        return ''
    parts = []
    for r in rows:
        summary = (r.get('summary_text') or '').strip()
        body = (r.get('profile_json') or '').strip()
        parts.append(
            f"## profile_id={r.get('profile_id')} created_at={r.get('created_at')} source={r.get('source')}\n"
            + (f"summary={summary}\n" if summary else "")
            + body
        )
    return '\n\n'.join(parts)


def recommend_products(user_id, top_n=5):
    """
    已登录用户：根据 SQLite 表 user_tags（标签名与商品 category 一致）的权重排序，
    在对应类目下选取商品；「我不喜欢」屏蔽的类目不参与。
    """
    if user_id is None:
        return products.iloc[0:0].copy()
    uid = int(user_id)
    tags = list_user_tags(_CATALOG_DB, uid)
    if not tags:
        return products.iloc[0:0].copy()
    tag_weights = {str(t['tag']): int(t['weight']) for t in tags}
    blocked = disliked_categories.get(uid, set())

    scored = []
    for _, product in products.iterrows():
        cat = str(product['category'])
        if cat not in tag_weights:
            continue
        if cat in blocked:
            continue
        pid = int(product['product_id'])
        w = tag_weights[cat]
        scored.append((pid, w, pid))

    scored.sort(key=lambda x: (-x[1], x[2]))
    recommended_ids = [x[0] for x in scored[:top_n]]
    if not recommended_ids:
        return products.iloc[0:0].copy()
    out = products[products['product_id'].isin(recommended_ids)].copy()
    order_map = {pid: i for i, pid in enumerate(recommended_ids)}
    out['_sort'] = out['product_id'].map(order_map)
    out = out.sort_values('_sort').drop(columns=['_sort'])
    return out


def filter_catalog(df, query, category):
    """首页商品列表：先按类目精确筛选，再按关键词在标题/描述/类目中子串搜索。"""
    out = df
    cat = (category or '').strip()
    if cat:
        out = out[out['category'] == cat]
    q = (query or '').strip()
    if not q:
        return out
    mask = (
        out['title'].str.contains(q, case=False, na=False)
        | out['description'].str.contains(q, case=False, na=False)
        | out['category'].str.contains(q, case=False, na=False)
    )
    return out[mask]


def filter_products_by_query(df, query):
    """兼容旧代码：仅按关键词筛选，不按类目。新逻辑请使用 filter_catalog。"""
    return filter_catalog(df, query, '')


def build_user_activity_display(user_id=None):
    """合并行为与商品标题；若传入 user_id 则只显示该用户的行为。"""
    display = user_activity.merge(
        products[['product_id', 'title']],
        on='product_id',
        how='left',
    )
    display['title'] = display['title'].fillna('（未知商品）')
    if user_id is not None:
        display = display[display['user_id'] == int(user_id)]
    return display


@app.route('/api/llm-recommend', methods=['POST'])
def api_llm_recommend():
    """
    接收用户自然语言，调用云端 LLM 解析为类目/关键词/价格区间，再过滤商品；
    解析失败或结果为空时用本地 BERT 与用户整句的语义相似度兜底。
    """
    if not request.is_json:
        return jsonify(ok=False, error='请使用 Content-Type: application/json'), 400
    data = request.get_json(silent=True) or {}
    query = (data.get('query') or '').strip()
    if not query:
        return jsonify(ok=False, error='请输入一段描述'), 400
    if len(query) > 4000:
        return jsonify(ok=False, error='文本过长（最多 4000 字）'), 400

    if not _llm_api_key():
        return jsonify(
            ok=False,
            error='未配置 API 密钥。请在 .env 填写 OPENAI_API_KEY 或 LLM_API_KEY；使用通义时还需 LLM_API_BASE（见 .env.example）',
        ), 503

    filters_used: dict = {}
    llm_error = None
    try:
        filters_used = _call_llm_for_filters(query)
    except Exception as exc:
        llm_error = str(exc)
        filters_used = {'categories': [], 'keywords': [], 'price_min': None, 'price_max': None, 'explain': ''}

    used_semantic = False
    if llm_error:
        base = _semantic_rank_by_query(query, top_n=10)
        used_semantic = True
    else:
        base = _apply_llm_filters_to_products(filters_used)
        if base.empty:
            base = _semantic_rank_by_query(query, top_n=10)
            used_semantic = True

    items = _products_df_to_json_list(base, limit=10)
    return jsonify(
        ok=True,
        products=items,
        filters=filters_used,
        used_semantic_fallback=used_semantic,
        llm_error=llm_error,
    )


@app.route('/api/profile-extract', methods=['POST'])
@login_required
def api_profile_extract():
    """根据当前用户的行为、相关商品与评论，调用 LLM 生成结构化用户画像。"""
    if not _llm_api_key():
        return jsonify(
            ok=False,
            error='未配置 API 密钥。请在 .env 填写 OPENAI_API_KEY 或 LLM_API_KEY；使用通义时还需 LLM_API_BASE（见 .env.example）',
        ), 503

    uid = int(session['user_id'])
    allowed = _product_categories_sorted()
    selected_interest_tags = _build_selected_interest_tags_text(uid)
    interaction_history = _build_interaction_history_text(uid)
    product_information = _build_product_information_text(uid)
    reviews_text = _build_reviews_text(uid)
    try:
        messages = build_profile_preference_extract_messages(
            allowed_categories=allowed,
            selected_interest_tags=selected_interest_tags,
            interaction_history=interaction_history,
            product_information=product_information,
            reviews_text=reviews_text,
        )
        profile = llm_call_json(messages=messages, temperature=0.2, max_tokens=2200)
    except Exception as exc:
        return jsonify(ok=False, error=f'画像生成失败：{exc}'), 502

    summary_text = ''
    saved_profile_id = None
    try:
        profile_text = json.dumps(profile, ensure_ascii=False)
        saved_profile_id = save_user_profile(
            _CATALOG_DB,
            uid,
            profile_text,
            source='llm_profile_extract_auto',
            summary_text='',
        )
        history_text = _build_profile_history_text(uid, limit=8)
        summary_messages = build_profile_summary_messages(history_profiles_text=history_text)
        summary_text = llm_call_text(summary_messages, temperature=0.2, max_tokens=120)[:200]
        if summary_text and saved_profile_id is not None:
            update_user_profile_summary(_CATALOG_DB, saved_profile_id, summary_text)
    except Exception:
        # 画像主体已生成，不因自动保存或摘要失败中断主流程。
        pass

    return jsonify(
        ok=True,
        profile=profile,
        profile_id=saved_profile_id,
        profile_summary=summary_text,
        inputs={
            'selected_interest_tag_lines': 0 if not selected_interest_tags else len(selected_interest_tags.splitlines()),
            'interaction_lines': 0 if not interaction_history else len(interaction_history.splitlines()),
            'product_lines': 0 if not product_information else len(product_information.splitlines()),
            'review_lines': 0 if not reviews_text else len(reviews_text.splitlines()),
        },
        evidence_preview={
            'selected_interest_tags': _clip_preview(selected_interest_tags),
            'interaction_history': _clip_preview(interaction_history),
            'product_information': _clip_preview(product_information),
            'reviews_text': _clip_preview(reviews_text),
        },
    )


@app.route('/api/profile-save', methods=['POST'])
@login_required
def api_profile_save():
    """保存前端确认后的画像 JSON 到 user_profiles。"""
    if not request.is_json:
        return jsonify(ok=False, error='请使用 Content-Type: application/json'), 400
    data = request.get_json(silent=True) or {}
    profile = data.get('profile')
    if not isinstance(profile, dict):
        return jsonify(ok=False, error='profile 须为 JSON 对象'), 400
    uid = int(session['user_id'])
    try:
        profile_text = json.dumps(profile, ensure_ascii=False)
    except (TypeError, ValueError):
        return jsonify(ok=False, error='profile 不是可序列化的 JSON 对象'), 400
    pid = save_user_profile(_CATALOG_DB, uid, profile_text, source='llm_profile_extract')
    return jsonify(ok=True, profile_id=pid)


def _product_exists(product_id: int) -> bool:
    return bool((products['product_id'] == int(product_id)).any())


def _product_categories_sorted() -> list[str]:
    return sorted(products['category'].astype(str).str.strip().unique().tolist())


@app.route('/api/interest-tags/status', endpoint='api_interest_tags_status')
@login_required
def api_interest_tags_status():
    uid = int(session['user_id'])
    user = get_user_by_id(_CATALOG_DB, uid)
    if not user:
        session.pop('user_id', None)
        return jsonify(ok=False, login_required=True), 401
    done = bool(user.get('interest_onboarding_done'))
    tag_rows = list_user_tags(_CATALOG_DB, uid)
    selected = [str(t['tag']) for t in tag_rows]
    return jsonify(
        ok=True,
        required=not done,
        categories=_product_categories_sorted(),
        selected=selected,
    )


@app.route('/api/interest-tags', methods=['POST'], endpoint='api_interest_tags_save')
@login_required
def api_interest_tags_save():
    uid = int(session['user_id'])
    user = get_user_by_id(_CATALOG_DB, uid)
    if not user:
        session.pop('user_id', None)
        return jsonify(ok=False, login_required=True), 401
    if not request.is_json:
        return jsonify(ok=False, error='请使用 Content-Type: application/json'), 400
    data = request.get_json(silent=True) or {}
    raw = data.get('tags')
    if not isinstance(raw, list):
        return jsonify(ok=False, error='tags 须为字符串数组'), 400
    allowed = set(_product_categories_sorted())
    picked: list[str] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, str):
            continue
        t = item.strip()
        if not t or t not in allowed or t in seen:
            continue
        seen.add(t)
        picked.append(t)
    if not picked:
        return jsonify(ok=False, error='请至少选择一个商品类目'), 400
    replace_user_interest_tags(_CATALOG_DB, uid, picked)
    set_interest_onboarding_done(_CATALOG_DB, uid, True)
    return jsonify(ok=True, tags=picked)


@app.route('/api/products/<int:product_id>/reviews', methods=['GET', 'POST'])
def api_product_reviews(product_id: int):
    if not _product_exists(product_id):
        return jsonify(ok=False, error='商品不存在'), 404
    if request.method == 'GET':
        uid = session.get('user_id')
        uid = int(uid) if uid is not None else None
        rows = list_product_reviews(_CATALOG_DB, product_id)
        for r in rows:
            r['is_mine'] = bool(uid is not None and r.get('user_id') == uid)
        return jsonify(ok=True, reviews=rows)

    if not session.get('user_id'):
        return jsonify(ok=False, login_required=True), 401
    if not request.is_json:
        return jsonify(ok=False, error='请使用 Content-Type: application/json'), 400
    data = request.get_json(silent=True) or {}
    body = (data.get('body') or '').strip()
    if not body:
        return jsonify(ok=False, error='评论内容不能为空'), 400
    if len(body) > 2000:
        return jsonify(ok=False, error='评论过长（最多 2000 字）'), 400
    uid = int(session['user_id'])
    user = get_user_by_id(_CATALOG_DB, uid)
    if not user:
        session.pop('user_id', None)
        return jsonify(ok=False, login_required=True), 401
    name = (user.get('name') or user.get('email') or '用户').strip()
    try:
        rid = insert_product_review(_CATALOG_DB, product_id, uid, name, body)
    except ValueError as exc:
        return jsonify(ok=False, error=str(exc)), 400
    return jsonify(ok=True, review_id=rid)


@app.route('/api/home-widgets')
def api_home_widgets():
    """首页无刷新更新：返回推荐区与行为列表 HTML 片段（JSON）。"""
    user_id = session.get('user_id')
    user_id = int(user_id) if user_id is not None else None
    search_query = request.args.get('q', '').strip()
    active_category = request.args.get('category', '').strip()
    recommended = recommend_products(user_id=user_id)
    user_activity_display = build_user_activity_display(user_id)
    recommended_html = render_template(
        '_fragment_recommend_grid.html',
        recommended=recommended,
        rec_user_id=user_id,
        rec_back='index',
        search_query=search_query,
        active_category=active_category,
    )
    activity_html = render_template(
        '_fragment_activity_list.html',
        user_activity=user_activity_display,
    )
    return jsonify(recommended_html=recommended_html, activity_html=activity_html)


@app.route('/__dev/live-reload')
def dev_live_reload():
    """
    仅 DEBUG 下可用：返回进程号与静态资源最新 mtime。
    前端轮询用于在保存代码或模板后自动刷新浏览器。
    """
    if not app.debug:
        abort(404)
    payload = {'pid': os.getpid(), 'assets': _dev_newest_asset_mtime()}
    return Response(json.dumps(payload), mimetype='application/json')


@app.route('/')
def index():
    """首页：商品列表（?category= 类目、?q= 搜索）、推荐与行为展示。"""
    search_query = request.args.get('q', '').strip()
    active_category = request.args.get('category', '').strip()
    catalog = filter_catalog(products, search_query, active_category)
    product_categories = sorted(products['category'].unique().tolist())
    uid = session.get('user_id')
    uid = int(uid) if uid is not None else None
    recommended = recommend_products(user_id=uid)
    user_activity_display = build_user_activity_display(uid)
    return render_template(
        'index.html',
        products=catalog,
        user_activity=user_activity_display,
        recommended=recommended,
        search_query=search_query,
        active_category=active_category,
        product_categories=product_categories,
        rec_user_id=uid,
        rec_back='index',
        llm_enabled=bool(_llm_api_key()),
    )


@app.route('/recommendations')
@login_required
def recommendations():
    """登录后按当前会话用户展示个性化推荐列表（独立页面）。"""
    uid = int(session['user_id'])
    recommended = recommend_products(user_id=uid)
    return render_template(
        'recommendations.html',
        recommended=recommended,
        rec_user_id=uid,
        rec_back='recommendations',
        search_query='',
        active_category='',
        rec_sidebar_ajax=False,
    )


@app.route('/recommendations/<int:_legacy_user_id>')
def recommendations_legacy(_legacy_user_id):
    """旧链接兼容：统一跳转到基于会话的推荐页。"""
    return redirect(url_for('recommendations'), code=302)


@app.route('/interact', methods=['POST'])
def interact():
    """
    记录一次浏览(click)或购买(purchase)，写回全局 user_activity 后回到首页（保留类目与搜索）。
    若用户曾对该商品所属类目点过「我不喜欢」，本次交互会解除该类目的屏蔽，推荐列表可再次纳入该品类。
    """
    sid = session.get('user_id')
    if not sid:
        if request.form.get('ajax') == '1':
            return jsonify(ok=False, login_required=True), 401
        flash('请先登录后再记录浏览或购买。', 'info')
        return redirect(url_for('login', next=url_for('index')))

    user_id = int(request.form['user_id'])
    if user_id != int(sid):
        abort(403)
    product_id = int(request.form['product_id'])
    action = request.form.get('action') or 'click'

    global user_activity
    user_activity = pd.concat(
        [user_activity, pd.DataFrame([{'user_id': user_id, 'product_id': product_id, 'action': action}])],
        ignore_index=True,
    )

    p_cat = products.loc[products['product_id'] == product_id, 'category']
    if not p_cat.empty:
        cat = p_cat.iloc[0]
        disliked_categories[user_id].discard(cat)
        bump_user_tag_for_category(_CATALOG_DB, user_id, str(cat))

    if request.form.get('ajax') == '1':
        return jsonify(ok=True)

    idx_kwargs = {}
    q = request.form.get('next_q', '').strip()
    if q:
        idx_kwargs['q'] = q
    cat = request.form.get('next_category', '').strip()
    if cat:
        idx_kwargs['category'] = cat
    return redirect(url_for('index', **idx_kwargs))


def _redirect_after_rec_sidebar_action():
    """推荐卡片上操作后的跳转：回到首页并保留搜索与类目筛选。"""
    idx_kwargs = {}
    q = request.form.get('next_q', '').strip()
    if q:
        idx_kwargs['q'] = q
    cat = request.form.get('next_category', '').strip()
    if cat:
        idx_kwargs['category'] = cat
    return redirect(url_for('index', **idx_kwargs))


@app.route('/rec/favorite', methods=['POST'])
def rec_favorite():
    """收藏推荐商品（写入内存，可用于扩展展示）。"""
    sid = session.get('user_id')
    if not sid:
        if request.form.get('ajax') == '1':
            return jsonify(ok=False, login_required=True), 401
        flash('请先登录。', 'info')
        return redirect(url_for('login', next=request.referrer or url_for('index')))
    user_id = int(request.form['user_id'])
    if user_id != int(sid):
        abort(403)
    product_id = int(request.form['product_id'])
    user_favorites[user_id].add(product_id)
    if request.form.get('ajax') == '1':
        return jsonify(ok=True)
    back = request.form.get('back', 'index')
    if back == 'recommendations':
        return redirect(url_for('recommendations'))
    return _redirect_after_rec_sidebar_action()


@app.route('/rec/dislike-category', methods=['POST'])
def rec_dislike_category():
    """不喜欢：将该商品所属类目加入黑名单，后续推荐不再出现该类目下任何商品。"""
    sid = session.get('user_id')
    if not sid:
        if request.form.get('ajax') == '1':
            return jsonify(ok=False, login_required=True), 401
        flash('请先登录。', 'info')
        return redirect(url_for('login', next=request.referrer or url_for('index')))
    user_id = int(request.form['user_id'])
    if user_id != int(sid):
        abort(403)
    category = (request.form.get('category') or '').strip()
    if category:
        disliked_categories[user_id].add(category)
    if request.form.get('ajax') == '1':
        return jsonify(ok=True)
    back = request.form.get('back', 'index')
    if back == 'recommendations':
        return redirect(url_for('recommendations'))
    return _redirect_after_rec_sidebar_action()


@app.route('/register', methods=['GET', 'POST'])
def register():
    if session.get('user_id'):
        return redirect(url_for('index'))
    if request.method == 'POST':
        email = (request.form.get('email') or '').strip()
        password = request.form.get('password') or ''
        name = (request.form.get('name') or '').strip()
        if not email or not password or not name:
            flash('请填写邮箱、密码与姓名。', 'error')
            return render_template('register.html')
        if get_user_by_email(_CATALOG_DB, email):
            flash('该邮箱已注册，请直接登录。', 'error')
            return render_template('register.html')
        pw_hash = generate_password_hash(password)
        new_id = create_user(_CATALOG_DB, email, pw_hash, name)
        session['user_id'] = new_id
        reset_in_memory_user_state(new_id)
        flash('注册成功，欢迎！', 'success')
        return redirect(url_for('index'))
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if session.get('user_id'):
        return redirect(url_for('index'))
    if request.method == 'POST':
        email = (request.form.get('email') or '').strip()
        password = request.form.get('password') or ''
        row = get_password_hash_for_login(_CATALOG_DB, email)
        if not row or not check_password_hash(row[1], password):
            flash('邮箱或密码不正确。', 'error')
            return render_template('login.html')
        session['user_id'] = row[0]
        reset_in_memory_user_state(row[0])
        nxt = (request.args.get('next') or request.form.get('next') or '').strip()
        if nxt.startswith('/') and not nxt.startswith('//'):
            return redirect(nxt)
        return redirect(url_for('index'))
    return render_template('login.html')


@app.route('/logout', methods=['POST'])
def logout():
    session.pop('user_id', None)
    flash('您已退出登录。', 'info')
    return redirect(url_for('index'))


@app.route('/account')
@login_required
def account():
    uid = int(session['user_id'])
    user = get_user_by_id(_CATALOG_DB, uid)
    if not user:
        session.pop('user_id', None)
        flash('会话已失效，请重新登录。', 'info')
        return redirect(url_for('login'))
    tags = list_user_tags(_CATALOG_DB, uid)
    latest_profile = get_latest_user_profile(_CATALOG_DB, uid)
    product_categories = _product_categories_sorted()
    selected_interest_tags = {str(t['tag']) for t in tags}
    latest_profile_dict = None
    latest_profile_pretty = ''
    latest_profile_summary = ''
    if latest_profile:
        try:
            latest_profile_dict = json.loads(latest_profile.get('profile_json') or '{}')
            latest_profile_pretty = json.dumps(latest_profile_dict, ensure_ascii=False, indent=2)
        except (TypeError, ValueError):
            latest_profile_dict = None
            latest_profile_pretty = latest_profile.get('profile_json') or ''
        latest_profile_summary = str(latest_profile.get('summary_text') or '').strip()
    return render_template(
        'account.html',
        user=user,
        tags=tags,
        product_categories=product_categories,
        selected_interest_tags=selected_interest_tags,
        llm_enabled=bool(_llm_api_key()),
        latest_profile=latest_profile,
        latest_profile_dict=latest_profile_dict,
        latest_profile_pretty=latest_profile_pretty,
        latest_profile_summary=latest_profile_summary,
    )


if __name__ == "__main__":
    app.run(debug=True)
