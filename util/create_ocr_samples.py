"""生成可复现的 OCR 冒烟测试图；手写字体样例不是真人手写数据。"""
from pathlib import Path
import json
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1] / 'data/ocr_samples'


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    fonts = Path('/System/Library/Fonts/Supplemental')
    samples = [
        ('01_printed.png', 'Arial.ttf', ['English OCR Test', 'The quick brown fox jumps over the lazy dog.', 'I enjoy reading books and learning English.']),
        ('02_essay.png', 'Times New Roman.ttf', ['My School Life', 'Every morning, I walk to school with my friends.', 'Our teacher helps us understand new ideas.', 'After class, we play football in the playground.', 'I hope to become a scientist in the future.']),
        ('03_errors.png', 'Arial.ttf', ['She go to school every day.', 'Yesterday I buyed a new book.', 'He have two apple.', 'I am very hapy today.']),
        ('04_handwriting_font.png', 'Bradley Hand Bold.ttf', ['My Weekend', 'Last Sunday, I visited my grandparents.', 'We cooked lunch and talked about school.', 'It was a wonderful day.']),
    ]
    metadata = []
    for name, font_name, lines in samples:
        image = Image.new('RGB', (1600, 180 + 90 * len(lines)), 'white')
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype(str(fonts / font_name), 42)
        for index, line in enumerate(lines):
            draw.text((70, 65 + index * 90), line, font=font, fill=(20, 20, 20))
        image.save(ROOT / name)
        (ROOT / (Path(name).stem + '.expected.txt')).write_text('\n'.join(lines) + '\n', encoding='utf-8')
        metadata.append({'image': name, 'kind': 'synthetic_handwriting_font' if name.startswith('04') else 'synthetic_printed', 'text': '\n'.join(lines)})
    (ROOT / 'manifest.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    print(ROOT)


if __name__ == '__main__':
    main()
