"""为 data/products.csv 每条商品追加更完整的描述段落，并写回 CSV。"""
from __future__ import annotations

import csv
from pathlib import Path

EXTRA_BY_CAT: dict[str, str] = {
    '电子产品': '适用场景涵盖日常办公、学习与影音娱乐。选购前请核对接口、供电与系统兼容性；建议保留外包装以便售后。',
    '运动户外': '适合居家锻炼与户外轻度运动。请结合自身身体条件循序渐进使用，并注意热身与补水；尺码以商品说明为准。',
    '食品饮料': '请置于阴凉干燥处保存，开封后尽快食用。过敏人群请留意配料表；儿童与老人食用请适量。',
    '家用电器': '安装与使用前请阅读说明书，注意用电安全与接地。演示价格为参考标价，以结算页与活动规则为准。',
    '美妆个护': '首次使用建议在耳后做敏感测试。若出现不适请停用并咨询医生；避免接触眼周开放伤口。',
    '图书文娱': '纸张与印刷批次可能略有差异。拼图类商品含细小零件，儿童请在成人监护下使用。',
    '服饰鞋包': '因显示器差异颜色可能略有偏差，请以实物为准。洗涤请按洗标操作，避免长时间暴晒。',
    '居家收纳': '承重请勿超过商品说明；金属件请保持干燥防锈。塑料与布艺部件远离高温明火。',
    '母婴用品': '婴幼儿用品请在成人监护下使用。辅食与纸尿裤请按月龄与体重选择；开封后注意卫生与保质期。',
    '宠物用品': '换粮请遵循七日过渡法。零食仅作奖励不宜过量；饮水机与食盆请定期清洗消毒。',
}

DEFAULT_EXTRA = '本商城为演示环境，库存与促销以结算页为准；商品图片仅供参考，请以实物为准。'


def build_description(base: str, category: str) -> str:
    b = (base or '').strip()
    if not b.endswith(('。', '！', '？')):
        b += '。'
    extra = EXTRA_BY_CAT.get(category.strip(), DEFAULT_EXTRA)
    if extra in b:
        return b
    return b + extra


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    path = root / 'data' / 'products.csv'
    rows = []
    with open(path, encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            row['description'] = build_description(row['description'], row['category'])
            rows.append(row)
    with open(path, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator='\n')
        w.writeheader()
        w.writerows(rows)
    print('Updated', len(rows), 'rows in', path)


if __name__ == '__main__':
    main()
