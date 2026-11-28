"""macOS 本地英文 OCR：图片/目录 → TXT、JSON 和识别框预览。

默认关闭语言纠正，尽量保留作文里的拼写错误；OCR 仍可能误读。
使用 Apple Vision，不调用云服务，也不调用本项目的语法错误检测模型。
"""
import argparse
from datetime import datetime
import io
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

from PIL import Image, ImageDraw, ImageOps

PROJECT_DIR = Path(__file__).resolve().parent
IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}


def collect_images(path):
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f'输入不存在：{path}')
    if path.is_dir():
        images = sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        if not images:
            raise ValueError(f'目录中没有支持的图片：{path}')
        return images
    if path.suffix.lower() not in IMAGE_SUFFIXES:
        raise ValueError(f'不支持的图片格式：{path.suffix}')
    return [path]


def pixel_box(box, width, height):
    """Vision 左下原点归一化坐标 → 图片左上原点像素坐标。"""
    x, y, w, h = box
    return [max(0, round(x * width)), max(0, round((1 - y - h) * height)),
            min(width, round((x + w) * width)), min(height, round((1 - y) * height))]


class EnglishOCR:
    def __init__(self, language_correction=False):
        if sys.platform != 'darwin':
            raise RuntimeError('此版本使用 Apple Vision，仅支持 macOS。')
        try:
            import Vision
            import Foundation
        except ImportError as error:
            raise RuntimeError('缺少 OCR 依赖，请运行：python -m pip install -r requirements-ocr.txt') from error
        self.vision = Vision
        self.foundation = Foundation
        self.language_correction = language_correction

    def recognize(self, image_path):
        # 先按 EXIF 旋转并统一 RGB，输出框坐标对应这张校正后的图片。
        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source).convert('RGB')
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        payload = buffer.getvalue()
        data = self.foundation.NSData.dataWithBytes_length_(payload, len(payload))
        request = self.vision.VNRecognizeTextRequest.alloc().init()
        request.setRecognitionLevel_(self.vision.VNRequestTextRecognitionLevelAccurate)
        request.setRecognitionLanguages_(['en-US'])
        request.setUsesLanguageCorrection_(self.language_correction)
        handler = self.vision.VNImageRequestHandler.alloc().initWithData_options_(data, None)
        success, error = handler.performRequests_error_([request], None)
        if not success:
            raise RuntimeError(f'Vision 识别失败：{error}')
        lines = []
        for observation in request.results() or []:
            candidates = observation.topCandidates_(1)
            if not candidates:
                continue
            candidate = candidates[0]
            box = observation.boundingBox()
            lines.append({
                'text': str(candidate.string()),
                'confidence': float(candidate.confidence()),
                'box': pixel_box((box.origin.x, box.origin.y, box.size.width, box.size.height), *image.size),
            })
        # 面向单栏作文，按自上而下排列；复杂多栏版式需额外分析。
        lines.sort(key=lambda line: (line['box'][1], line['box'][0]))
        result = {
            'source': str(Path(image_path).resolve()) if isinstance(image_path, (str, Path)) else 'uploaded image',
            'engine': 'Apple Vision',
            'language': 'en-US',
            'language_correction': self.language_correction,
            'width': image.width, 'height': image.height,
            'coordinate_system': 'EXIF-oriented image, top-left origin, pixels',
            'text': '\n'.join(line['text'] for line in lines),
            'lines': lines,
        }
        return result, image


def save_result(result, image, output_dir, prefix):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f'{prefix}.txt').write_text(result['text'] + '\n', encoding='utf-8')
    (output_dir / f'{prefix}.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    preview = image.copy()
    draw = ImageDraw.Draw(preview)
    for index, line in enumerate(result['lines'], 1):
        draw.rectangle(line['box'], outline='red', width=2)
        draw.text((line['box'][0], max(0, line['box'][1] - 14)), str(index), fill='red')
    preview.save(output_dir / f'{prefix}_boxes.png')


def main():
    parser = argparse.ArgumentParser(description='本地识别图片中的英文（macOS Apple Vision）')
    parser.add_argument('input', nargs='?', type=Path, default=PROJECT_DIR / 'data/ocr_samples',
                        help='图片或图片目录；默认运行附带测试图片')
    parser.add_argument('--output', type=Path, help='结果目录；默认 result/ocr/日期/时间')
    parser.add_argument('--language-correction', action='store_true', help='开启语言纠正；作文原文识别建议保持关闭')
    args = parser.parse_args()
    try:
        images = collect_images(args.input)
        engine = EnglishOCR(args.language_correction)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f'{error}\n')
    now = datetime.now(ZoneInfo('Asia/Shanghai'))
    output_dir = args.output or PROJECT_DIR / 'result/ocr' / now.strftime('%Y-%m-%d/%H-%M-%S-%f')
    failures = 0
    for index, path in enumerate(images, 1):
        try:
            result, image = engine.recognize(path)
            # 序号+完整文件名避免同名不同扩展名图片覆盖结果。
            save_result(result, image, output_dir, f'{index:03d}_{path.name}')
            print(f'\n[{index}/{len(images)}] {path.name}\n{result["text"] or "（未识别到文字）"}')
        except (OSError, ValueError, RuntimeError) as error:
            failures += 1
            print(f'识别失败 {path}: {error}', file=sys.stderr)
    print(f'\n结果目录：{output_dir}\n成功 {len(images) - failures}，失败 {failures}')
    if failures:
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
