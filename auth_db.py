"""
注册用户与会话相关数据：与商品库共用 instance/catalog.db。
表 users、user_tags；点击/购买商品时按类目累加标签权重。
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

AUTH_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT UNIQUE NOT NULL COLLATE NOCASE,
    password_hash TEXT NOT NULL,
    name TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS user_tags (
    user_id INTEGER NOT NULL,
    tag TEXT NOT NULL,
    weight INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (user_id, tag),
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_user_tags_user ON user_tags(user_id);
"""


def init_auth_db(db_path: Path) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute('PRAGMA foreign_keys = ON')
        conn.executescript(AUTH_SCHEMA)
        conn.commit()


def get_user_by_email(db_path: Path, email: str) -> dict[str, Any] | None:
    email = (email or '').strip().lower()
    if not email:
        return None
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            'SELECT user_id, email, name, created_at FROM users WHERE lower(email) = ?',
            (email,),
        ).fetchone()
    if not row:
        return None
    return {'user_id': row[0], 'email': row[1], 'name': row[2], 'created_at': row[3]}


def get_user_by_id(db_path: Path, user_id: int) -> dict[str, Any] | None:
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            'SELECT user_id, email, name, created_at FROM users WHERE user_id = ?',
            (int(user_id),),
        ).fetchone()
    if not row:
        return None
    return {'user_id': row[0], 'email': row[1], 'name': row[2], 'created_at': row[3]}


def create_user(db_path: Path, email: str, password_hash: str, name: str) -> int:
    email = email.strip().lower()
    name = name.strip()
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(
            'INSERT INTO users (email, password_hash, name) VALUES (?, ?, ?)',
            (email, password_hash, name),
        )
        conn.commit()
        return int(cur.lastrowid)


def bump_user_tag_for_category(db_path: Path, user_id: int, category: str) -> None:
    """用户点击/购买商品后，按商品类目累加标签权重。"""
    tag = (category or '').strip()
    if not tag:
        return
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO user_tags (user_id, tag, weight) VALUES (?, ?, 1)
            ON CONFLICT(user_id, tag) DO UPDATE SET
                weight = user_tags.weight + 1,
                updated_at = datetime('now')
            """,
            (int(user_id), tag),
        )
        conn.commit()


def list_user_tags(db_path: Path, user_id: int) -> list[dict[str, Any]]:
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT tag, weight, updated_at FROM user_tags
            WHERE user_id = ?
            ORDER BY weight DESC, tag ASC
            """,
            (int(user_id),),
        ).fetchall()
    return [{'tag': r[0], 'weight': r[1], 'updated_at': r[2]} for r in rows]


def get_password_hash_for_login(db_path: Path, email: str) -> tuple[int, str] | None:
    """返回 (user_id, password_hash) 供校验密码。"""
    email = (email or '').strip().lower()
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            'SELECT user_id, password_hash FROM users WHERE lower(email) = ?',
            (email,),
        ).fetchone()
    if not row:
        return None
    return int(row[0]), str(row[1])
