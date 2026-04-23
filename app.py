"""
购物商城演示应用：Flask 商城；登录用户「为您推荐」按 SQLite 兴趣标签（商品类目）权重选品，
LLM 找货与语义兜底等仍使用 BERT 向量与余弦相似度。
"""
import json
import os
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
from openai import OpenAI
import pandas as pd
from werkzeug.security import check_password_hash, generate_password_hash

from auth_db import (
    bump_user_tag_for_category,
    create_user,
    get_password_hash_for_login,
    get_user_by_email,
    get_user_by_id,
    init_auth_db,
    list_user_tags,
)
from catalog_db import catalog_db_path, init_catalog, load_products_dataframe, should_reseed_from_env
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
model = pipeline('feature-extraction', model='bert-base-chinese')


def get_embeddings(text):
    """对单条文本做特征提取，返回该句对应的一维向量（取 [CLS] 或首 token 表示，与 pipeline 输出一致）。"""
    return model(text)[0][0]


# 为每个商品预计算描述向量，供推荐与相似度计算复用
products['embeddings'] = products['description'].apply(get_embeddings)


# ---------------------------------------------------------------------------
# 云端 LLM：一句话描述 → 结构化筛选 + 商品列表
# 大模型 HTTP 调用：用官方 openai 包里的 OpenAI 客户端发请求（与「是否调用 OpenAI 公司」无关）。
# 未传 base_url 时该库默认连 api.openai.com；传了 LLM_API_BASE 则连通义等任意兼容网关。
# ---------------------------------------------------------------------------
def _llm_resolve():
    """
    返回 (api_key, base_url, model)。
    配置了 LLM_API_BASE 时：密钥优先 LLM_API_KEY，模型优先 LLM_MODEL；未设模型且 host 含 dashscope 时默认 qwen-turbo。
    未配置 LLM_API_BASE 时：沿用 OPENAI_* 与 OPENAI_API_BASE；未传 base_url 时由 HTTP 客户端默认连 OpenAI 官服。
    """
    llm_base = (os.environ.get('LLM_API_BASE') or '').strip().rstrip('/')
    openai_base = (os.environ.get('OPENAI_API_BASE') or '').strip().rstrip('/')

    if llm_base:
        api_key = (os.environ.get('LLM_API_KEY') or '').strip()
        if not api_key:
            api_key = (os.environ.get('OPENAI_API_KEY') or '').strip()
        model = (os.environ.get('LLM_MODEL') or os.environ.get('OPENAI_MODEL') or '').strip()
        if not model:
            model = 'qwen-turbo' if 'dashscope' in llm_base.lower() else 'gpt-4o-mini'
        return api_key, llm_base, model

    oa_key = (os.environ.get('OPENAI_API_KEY') or '').strip()
    lm_key = (os.environ.get('LLM_API_KEY') or '').strip()
    api_key = oa_key or lm_key
    model = (os.environ.get('OPENAI_MODEL') or os.environ.get('LLM_MODEL') or 'gpt-4o-mini').strip()
    return api_key, openai_base, model


def _llm_api_key() -> str:
    return _llm_resolve()[0]


def _llm_api_base() -> str:
    return _llm_resolve()[1]


def _llm_model() -> str:
    return _llm_resolve()[2]


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
    system = (
        '你是购物网站的检索助手。用户用自然语言描述想买的商品。'
        '请提取检索条件，只输出一个 JSON 对象，不要 markdown 代码块，不要其它文字。\n'
        '字段说明：\n'
        '- "categories": 字符串数组，每项必须严格属于下列「允许类目」之一；不确定则 []。\n'
        '  允许类目：' + json.dumps(allowed, ensure_ascii=False) + '\n'
        '- "keywords": 字符串数组，用于在商品标题、描述中做子串匹配，可多个，可 []\n'
        '- "price_min": 数字或 null，人民币最低价（含）\n'
        '- "price_max": 数字或 null，人民币最高价（含）\n'
        '- "explain": 一句中文，简述你如何理解用户需求\n'
    )
    api_key, base, model = _llm_resolve()
    if not api_key:
        raise RuntimeError('未配置 API 密钥：请在 .env 中设置 OPENAI_API_KEY 或 LLM_API_KEY')
    lm_key = (os.environ.get('LLM_API_KEY') or '').strip()
    oa_key = (os.environ.get('OPENAI_API_KEY') or '').strip()
    oa_base = (os.environ.get('OPENAI_API_BASE') or '').strip()
    if not base and lm_key and not oa_key and not oa_base:
        raise RuntimeError(
            '已配置 LLM_API_KEY，但未配置 LLM_API_BASE（也未配置 OPENAI_API_BASE）。'
            '未指定网关时请求会发往 OpenAI 官方。使用通义请在 .env 增加：\n'
            'LLM_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1'
        )
    kwargs = {'api_key': api_key}
    if base:
        kwargs['base_url'] = base
    client = OpenAI(**kwargs)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': user_text[:4000]},
        ],
        response_format={'type': 'json_object'},
        temperature=0.15,
        max_tokens=800,
    )
    raw = (resp.choices[0].message.content or '').strip() or '{}'
    data = json.loads(raw)
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
    return render_template('account.html', user=user, tags=tags)


if __name__ == "__main__":
    app.run(debug=True)
