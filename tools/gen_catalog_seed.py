"""一次性：从 data/products.csv 生成 data/catalog_seed.sql（UTF-8）。按需运行：python tools/gen_catalog_seed.py"""
from __future__ import annotations

import csv
from pathlib import Path


def esc(s: str) -> str:
    return str(s).strip().replace("'", "''")


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    csv_path = root / 'data' / 'products.csv'
    out = root / 'data' / 'catalog_seed.sql'
    lines = [
        '-- 商品初始数据：由本文件维护；应用仅在空库或 CATALOG_RESEED=1 时执行。',
        '-- 表：products（路径见 catalog_db.py）',
        'BEGIN;',
        'DELETE FROM products;',
    ]
    with open(csv_path, encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            pid = int(row['product_id'])
            title = esc(row['title'])
            cat = esc(row['category'])
            desc = esc(row['description'])
            price = float(row['price'])
            img = esc(row['image'])
            lines.append(
                'INSERT INTO products (product_id, title, category, description, price, image) VALUES ('
                f"{pid}, '{title}', '{cat}', '{desc}', {price}, '{img}');"
            )
    lines.append('COMMIT;')
    out.write_text('\n'.join(lines), encoding='utf-8')
    print('Wrote', out, 'statements:', len(lines) - 4)


if __name__ == '__main__':
    main()
