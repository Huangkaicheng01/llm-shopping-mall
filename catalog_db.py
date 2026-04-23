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
