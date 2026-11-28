"""生成明显语法错误案例；原文供 OCR 对照，修正供人工核验。"""
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1] / 'data/ocr_samples'
GROUPS = [
    ('05_subject_verb', '主谓一致', [
        ('She go to school every day.', 'She goes to school every day.', '第三人称单数主语 She，谓语 go 应为 goes。'),
        ('He have a red bicycle.', 'He has a red bicycle.', '第三人称单数主语 He，have 应为 has。'),
        ('They is playing football.', 'They are playing football.', '复数主语 They，is 应为 are。'),
        ('My parents likes this movie.', 'My parents like this movie.', '复数主语 parents，likes 应为 like。'),
    ]),
    ('06_verb_forms', '时态与动词形式', [
        ('Yesterday, I go to the library.', 'Yesterday, I went to the library.', 'Yesterday 表示过去，go 应为 went。'),
        ('She did not went to school.', 'She did not go to school.', 'did not 后接动词原形，went 应为 go。'),
        ('He can speaks English.', 'He can speak English.', '情态动词 can 后接动词原形，speaks 应为 speak。'),
        ('I am study English now.', 'I am studying English now.', '现在进行时使用 am + studying。'),
    ]),
    ('07_nouns_articles', '名词、冠词与代词', [
        ('I have two book.', 'I have two books.', 'two 后使用可数名词复数，book 应为 books。'),
        ('She is a excellent teacher.', 'She is an excellent teacher.', 'excellent 以元音音素开头，a 应为 an。'),
        ('This are my shoes.', 'These are my shoes.', '指代复数 shoes，应使用 These。'),
        ('Please give the book to I.', 'Please give the book to me.', '介词 to 后用宾格代词 me。'),
    ]),
]


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 44)
    metadata = []
    for name, category, cases in GROUPS:
        # 图片中仅放错误原文，避免 OCR 将修正答案一起读入。
        image = Image.new('RGB', (1600, 580), 'white')
        draw = ImageDraw.Draw(image)
        for i, (original, corrected, explanation) in enumerate(cases):
            assert draw.textbbox((0, 0), original, font=font)[2] < 1460
            draw.text((70, 70 + i * 115), original, font=font, fill=(20, 20, 20))
        image.save(ROOT / f'{name}.png')
        (ROOT / f'{name}.expected.txt').write_text('\n'.join(c[0] for c in cases) + '\n', encoding='utf-8')
        (ROOT / f'{name}.corrected.txt').write_text('\n'.join(c[1] for c in cases) + '\n', encoding='utf-8')
        metadata.append({'image': f'{name}.png', 'category': category, 'kind': 'synthetic_printed',
                         'cases': [dict(original=o, corrected=c, explanation=e) for o, c, e in cases]})
    (ROOT / 'grammar_cases.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# 明显语法错误案例', '', '共 3 张图片、12 个句子。图片仅包含错误原文；`.expected.txt` 是 OCR 应保留的原文，`.corrected.txt` 是人工参考修正。', '',
             '这些是人工编写的功能测试样例，不代表模型必然能识别全部错误。没有检测出来时，请分别核对 OCR 文本与模型输出。', '']
    for group in metadata:
        lines += [f"## {group['image']}：{group['category']}", '', '| 错误原文 | 参考修正 | 说明 |', '| --- | --- | --- |']
        for c in group['cases']:
            lines.append(f"| {c['original']} | {c['corrected']} | {c['explanation']} |")
        lines.append('')
    (ROOT / 'GRAMMAR_CASES.md').write_text('\n'.join(lines), encoding='utf-8')
    print('已生成 3 张测试图、12 个案例及原文/修正/说明。')


if __name__ == '__main__':
    main()
