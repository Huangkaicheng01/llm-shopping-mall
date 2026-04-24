"""
商品目录：SQLite 持久化（instance/catalog.db）。

- 首次启动（库为空）或设置 CATALOG_RESEED=1 重启时：执行 data/catalog_seed.sql 写入商品
- 日常增删改：用 DB 可视化工具或 sqlite3 命令行直接改库；改完重启 Flask 即可（或后续可加管理接口）
- 若需从 Excel/CSV 批量生成种子文件：运行 python tools/gen_catalog_seed.py

查看 / 管理数据库（任选其一）：
  · DB Browser for SQLite：https://sqlitebrowser.org/  → 打开 instance/catalog.db
  · VS Code 扩展：SQLite、SQLite Viewer 等 → 打开同一文件
  · 命令行：sqlite3 instance/catalog.db  然后 .tables  /  SELECT * FROM products;
"""
from __future__ import annotations

import os
import random
import sqlite3
from pathlib import Path

import pandas as pd


def catalog_db_path(root: Path) -> Path:
    return root / 'instance' / 'catalog.db'


def catalog_seed_sql_path(root: Path) -> Path:
    return root / 'data' / 'catalog_seed.sql'


def _configure_connection(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute('PRAGMA journal_mode=WAL')
    cur.execute('PRAGMA synchronous=NORMAL')
    cur.execute('PRAGMA temp_store=MEMORY')
    cur.execute('PRAGMA foreign_keys=ON')
    cur.close()


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS products (
            product_id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            category TEXT NOT NULL,
            description TEXT NOT NULL,
            price REAL NOT NULL,
            image TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_products_category ON products(category);
        CREATE INDEX IF NOT EXISTS idx_products_price ON products(price);
        CREATE INDEX IF NOT EXISTS idx_products_category_price ON products(category, price);
        CREATE TABLE IF NOT EXISTS product_reviews (
            review_id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id INTEGER NOT NULL,
            user_id INTEGER,
            reviewer_name TEXT NOT NULL,
            body TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (product_id) REFERENCES products(product_id) ON DELETE CASCADE,
            FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE SET NULL
        );
        CREATE INDEX IF NOT EXISTS idx_product_reviews_product ON product_reviews(product_id);
        """
    )
    conn.commit()


def seed_from_sql_file(conn: sqlite3.Connection, sql_path: Path) -> None:
    if not sql_path.is_file():
        raise FileNotFoundError(
            f'缺少商品种子 SQL：{sql_path}\n'
            f'若仓库中无此文件，可在项目根执行：python tools/gen_catalog_seed.py（会读取 data/products.csv 生成一次种子）'
        )
    script = sql_path.read_text(encoding='utf-8')
    conn.executescript(script)
    conn.commit()


def init_catalog(root: Path, *, reseed: bool = False) -> None:
    """创建 instance 目录与表；空库或 reseed 时执行 catalog_seed.sql。"""
    db_path = catalog_db_path(root)
    seed_path = catalog_seed_sql_path(root)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        _configure_connection(conn)
        ensure_schema(conn)
        cur = conn.execute('SELECT COUNT(*) FROM products')
        n = int(cur.fetchone()[0])
        if reseed:
            conn.execute('DELETE FROM product_reviews')
            conn.execute('DELETE FROM products')
            conn.commit()
            seed_from_sql_file(conn, seed_path)
        elif n == 0:
            seed_from_sql_file(conn, seed_path)


def load_products_dataframe(db_path: Path) -> pd.DataFrame:
    """读入商品表为 DataFrame，供 BERT 向量与推荐逻辑使用。"""
    with sqlite3.connect(db_path) as conn:
        _configure_connection(conn)
        df = pd.read_sql_query(
            'SELECT product_id, title, category, description, price, image FROM products ORDER BY product_id',
            conn,
        )
    df['product_id'] = df['product_id'].astype(int)
    df['price'] = df['price'].astype(float)
    for col in ('title', 'category', 'description', 'image'):
        df[col] = df[col].astype(str).str.strip()
    return df


def should_reseed_from_env() -> bool:
    v = (os.environ.get('CATALOG_RESEED') or '').strip().lower()
    return v in ('1', 'true', 'yes', 'on')


# ---------------------------------------------------------------------------
# 商品评论：演示种子（按类目随机文案）与已登录用户发表
# ---------------------------------------------------------------------------

_DEMO_REVIEWER_NAMES = (
    '淘***88',
    'J***d',
    '会**员',
    '匿**名用户',
    '买***家',
    '好***评君',
    '老***顾客',
    '用**户9527',
    '东***哥',
    '小***红',
)

_CATEGORY_REVIEW_BODIES: dict[str, tuple[str, ...]] = {
    '电子产品': (
        '物流很快，包装完好，开机验机无亮点，办公够用。',
        '和描述一致，接口齐全，发热控制还可以。',
        '性价比不错，售后客服回复也及时。',
        '用了一周，续航符合预期，推荐入手。',
        '做工扎实，就是说明书字有点小。',
    ),
    '运动户外': (
        '质量对得起价格，每天锻炼都在用。',
        '防滑效果不错，出汗也不打滑。',
        '尺码按说明选的，合身，满意。',
        '户外用了一次，耐磨性还可以。',
        '入门够用，进阶可能会想升级款。',
    ),
    '食品饮料': (
        '日期新鲜，口感和超市买的一样。',
        '整箱囤货很划算，家人都喜欢。',
        '包装严实，没有压坏。',
        '味道正宗，会回购。',
        '配料表看着挺干净，喝着放心。',
    ),
    '家用电器': (
        '安装简单，噪音比想象中小。',
        '功率够用，清洗也方便。',
        '外观简洁，和厨房风格很搭。',
        '用了一阵子没出问题，五星好评。',
        '说明书清楚，老人也能学会用。',
    ),
    '美妆个护': (
        '温和不刺激，洗完脸不紧绷。',
        '吸收快，没有闷痘。',
        '香味淡淡的，可以接受。',
        '礼盒包装精美，送人合适。',
        '和专柜试用感觉接近，正品感。',
    ),
    '图书文娱': (
        '印刷清晰，纸质手感好。',
        '孩子很喜欢，每晚都要读一本。',
        '拼图咬合紧，拼完可以整片拿起。',
        '内容实用，案例讲得清楚。',
        '物流快，边角几乎没有磕碰。',
    ),
    '服饰鞋包': (
        '面料舒服，版型正，按尺码表买的刚好。',
        '颜色和图片接近，洗涤没掉色。',
        '通勤背了一周，肩带减压还行。',
        '薄厚适合这个季节，性价比高。',
        '线头不多，拉链顺滑。',
    ),
    '居家收纳': (
        '组装五分钟搞定，承重没问题。',
        '收纳空间比想象的大。',
        '金属件表面处理得不错，没毛刺。',
        '角落刚好塞进去，省地方。',
        '布艺没异味，可水洗很方便。',
    ),
    '母婴用品': (
        '宝宝用着适应，没有过敏。',
        '材质摸起来安全，会继续买。',
        '推车收合顺滑，单手能操作。',
        '湿巾水分足，盖子密封好。',
        '米粉冲泡不结块，味道清淡。',
    ),
    '宠物用品': (
        '主子爱吃，换粮过渡期也很顺利。',
        '牵引绳结实，夜间反光条实用。',
        '猫砂结团快，除臭还行。',
        '零食颗粒大小合适，训练用刚好。',
        '封闭式真的少了很多带砂出来。',
    ),
}

_DEFAULT_REVIEW_BODIES = (
    '整体符合预期，描述和实物差别不大。',
    '客服态度好，有问题处理得快。',
    '在这个价位算满意，会考虑再买。',
    '包装仔细，没有破损。',
    '用了一段时间才来评，稳定可靠。',
)


def seed_demo_product_reviews_if_empty(db_path: Path) -> None:
    """若评论表为空，则为每件商品按类目随机插入若干条演示评论（user_id 为空）。"""
    with sqlite3.connect(db_path) as conn:
        _configure_connection(conn)
        cur = conn.execute('SELECT COUNT(*) FROM product_reviews')
        if int(cur.fetchone()[0]) > 0:
            return
        rows = conn.execute('SELECT product_id, category FROM products ORDER BY product_id').fetchall()
        bodies_by_cat = {k: list(v) for k, v in _CATEGORY_REVIEW_BODIES.items()}
        default_list = list(_DEFAULT_REVIEW_BODIES)
        names = list(_DEMO_REVIEWER_NAMES)
        for pid, cat in rows:
            pool = bodies_by_cat.get((cat or '').strip(), default_list)
            n = random.randint(2, 5)
            if len(pool) >= n:
                chosen = random.sample(pool, n)
            else:
                chosen = random.choices(pool, k=n)
            for body in chosen:
                conn.execute(
                    """
                    INSERT INTO product_reviews (product_id, user_id, reviewer_name, body)
                    VALUES (?, NULL, ?, ?)
                    """,
                    (int(pid), random.choice(names), body),
                )
        conn.commit()


def list_product_reviews(db_path: Path, product_id: int) -> list[dict]:
    with sqlite3.connect(db_path) as conn:
        _configure_connection(conn)
        rows = conn.execute(
            """
            SELECT review_id, user_id, reviewer_name, body, created_at
            FROM product_reviews
            WHERE product_id = ?
            ORDER BY review_id DESC
            """,
            (int(product_id),),
        ).fetchall()
    return [
        {
            'review_id': int(r[0]),
            'user_id': int(r[1]) if r[1] is not None else None,
            'reviewer_name': str(r[2]),
            'body': str(r[3]),
            'created_at': str(r[4]),
        }
        for r in rows
    ]


def insert_product_review(db_path: Path, product_id: int, user_id: int, reviewer_name: str, body: str) -> int:
    name = (reviewer_name or '').strip() or '用户'
    text = (body or '').strip()
    if not text:
        raise ValueError('empty body')
    if len(text) > 2000:
        text = text[:2000]
    with sqlite3.connect(db_path) as conn:
        _configure_connection(conn)
        cur = conn.execute(
            """
            INSERT INTO product_reviews (product_id, user_id, reviewer_name, body)
            VALUES (?, ?, ?, ?)
            """,
            (int(product_id), int(user_id), name, text),
        )
        conn.commit()
        return int(cur.lastrowid)
