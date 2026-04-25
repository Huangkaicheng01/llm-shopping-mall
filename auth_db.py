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
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    interest_onboarding_done INTEGER NOT NULL DEFAULT 0
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

CREATE TABLE IF NOT EXISTS user_profiles (
    profile_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    profile_json TEXT NOT NULL,
    summary_text TEXT NOT NULL DEFAULT '',
    source TEXT NOT NULL DEFAULT 'llm',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_user_profiles_user_created
ON user_profiles(user_id, created_at DESC);
"""


def _migrate_users_interest_onboarding(conn: sqlite3.Connection) -> None:
    """旧库补列；已有 user_tags 的用户视为已完成引导，避免老用户再被拦截。"""
    cols = {row[1] for row in conn.execute('PRAGMA table_info(users)').fetchall()}
    if 'interest_onboarding_done' not in cols:
        conn.execute(
            'ALTER TABLE users ADD COLUMN interest_onboarding_done INTEGER NOT NULL DEFAULT 0'
        )
    conn.execute(
        """
        UPDATE users SET interest_onboarding_done = 1
        WHERE EXISTS (
            SELECT 1 FROM user_tags ut WHERE ut.user_id = users.user_id
        )
        """
    )


def init_auth_db(db_path: Path) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute('PRAGMA foreign_keys = ON')
        conn.executescript(AUTH_SCHEMA)
        _migrate_users_interest_onboarding(conn)
        cols = {row[1] for row in conn.execute('PRAGMA table_info(user_profiles)').fetchall()}
        if cols and 'summary_text' not in cols:
            conn.execute("ALTER TABLE user_profiles ADD COLUMN summary_text TEXT NOT NULL DEFAULT ''")
        conn.commit()


def get_user_by_email(db_path: Path, email: str) -> dict[str, Any] | None:
    email = (email or '').strip().lower()
    if not email:
        return None
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT user_id, email, name, created_at,
                   COALESCE(interest_onboarding_done, 0) AS interest_onboarding_done
            FROM users WHERE lower(email) = ?
            """,
            (email,),
        ).fetchone()
    if not row:
        return None
    return {
        'user_id': row[0],
        'email': row[1],
        'name': row[2],
        'created_at': row[3],
        'interest_onboarding_done': int(row[4] or 0),
    }


def get_user_by_id(db_path: Path, user_id: int) -> dict[str, Any] | None:
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT user_id, email, name, created_at,
                   COALESCE(interest_onboarding_done, 0) AS interest_onboarding_done
            FROM users WHERE user_id = ?
            """,
            (int(user_id),),
        ).fetchone()
    if not row:
        return None
    return {
        'user_id': row[0],
        'email': row[1],
        'name': row[2],
        'created_at': row[3],
        'interest_onboarding_done': int(row[4] or 0),
    }


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


def set_interest_onboarding_done(db_path: Path, user_id: int, done: bool = True) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            'UPDATE users SET interest_onboarding_done = ? WHERE user_id = ?',
            (1 if done else 0, int(user_id)),
        )
        conn.commit()


def replace_user_interest_tags(db_path: Path, user_id: int, tags: list[str]) -> None:
    """用手选的商品类目覆盖 user_tags（每条初始权重 5）。"""
    uid = int(user_id)
    seen: set[str] = set()
    ordered: list[str] = []
    for raw in tags:
        t = (raw or '').strip()
        if not t or t in seen:
            continue
        seen.add(t)
        ordered.append(t)
    with sqlite3.connect(db_path) as conn:
        conn.execute('DELETE FROM user_tags WHERE user_id = ?', (uid,))
        for tag in ordered:
            conn.execute(
                'INSERT INTO user_tags (user_id, tag, weight) VALUES (?, ?, 5)',
                (uid, tag),
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


def save_user_profile(
    db_path: Path,
    user_id: int,
    profile_json: str,
    source: str = 'llm',
    summary_text: str = '',
) -> int:
    """保存一版用户画像 JSON，返回 profile_id。"""
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO user_profiles (user_id, profile_json, source, summary_text)
            VALUES (?, ?, ?, ?)
            """,
            (int(user_id), str(profile_json), str(source or 'llm'), str(summary_text or '')),
        )
        conn.commit()
        return int(cur.lastrowid)


def update_user_profile_summary(db_path: Path, profile_id: int, summary_text: str) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            'UPDATE user_profiles SET summary_text = ? WHERE profile_id = ?',
            (str(summary_text or ''), int(profile_id)),
        )
        conn.commit()


def list_recent_user_profiles(db_path: Path, user_id: int, limit: int = 10) -> list[dict[str, Any]]:
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT profile_id, profile_json, summary_text, source, created_at
            FROM user_profiles
            WHERE user_id = ?
            ORDER BY profile_id DESC
            LIMIT ?
            """,
            (int(user_id), int(limit)),
        ).fetchall()
    out: list[dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                'profile_id': int(r[0]),
                'profile_json': str(r[1]),
                'summary_text': str(r[2] or ''),
                'source': str(r[3]),
                'created_at': str(r[4]),
            }
        )
    return out


def get_latest_user_profile(db_path: Path, user_id: int) -> dict[str, Any] | None:
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT profile_id, profile_json, summary_text, source, created_at
            FROM user_profiles
            WHERE user_id = ?
            ORDER BY profile_id DESC
            LIMIT 1
            """,
            (int(user_id),),
        ).fetchone()
    if not row:
        return None
    return {
        'profile_id': int(row[0]),
        'profile_json': str(row[1]),
        'summary_text': str(row[2] or ''),
        'source': str(row[3]),
        'created_at': str(row[4]),
    }
