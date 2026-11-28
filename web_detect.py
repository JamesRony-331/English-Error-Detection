"""本地网页：图片上传 → 英文 OCR → 词级错误检测。"""
import argparse
import base64
import io
from pathlib import Path
import re
import tomllib
from threading import Lock

from flask import Flask, jsonify, render_template, request
from PIL import Image, ImageOps, UnidentifiedImageError

from OCR import EnglishOCR
from detect import Detector, find_latest_weights

ROOT = Path(__file__).resolve().parent


def detect_text(detector, text):
    """分句推理并将偏移还原到整篇 OCR 文本，避免跨句卷积。"""
    tokens = []
    for match in re.finditer(r'[^.!?\n]+[.!?]*|[.!?]+', text):
        if not match.group().strip():
            continue
        for token in detector.predict(match.group()):
            token = dict(token)
            token['start'] += match.start()
            token['end'] += match.start()
            tokens.append(token)
    return tokens


def create_app(weights=None, config=None, ocr=None, detector=None):
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = 12 * 1024 * 1024
    lock = Lock()
    services = {'ocr': ocr, 'detector': detector, 'weights': str(weights) if weights else None}

    @app.get('/')
    def index():
        return render_template('detect.html')

    @app.errorhandler(413)
    def too_large(error):
        return jsonify(error='图片不能超过 12 MB。'), 413

    @app.post('/api/recognize')
    def recognize():
        uploaded = request.files.get('image')
        if uploaded is None or not uploaded.filename:
            return jsonify(error='请上传一张图片，字段名为 image。'), 400
        try:
            with Image.open(uploaded.stream) as source:
                if source.width * source.height > 20_000_000:
                    return jsonify(error='图片不能超过 2000 万像素。'), 400
                image = ImageOps.exif_transpose(source).convert('RGB')
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
            return jsonify(error='无法读取图片，请上传有效的 PNG、JPEG 或其他常见图片。'), 400
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        buffer.seek(0)
        try:
            with lock:
                if services['ocr'] is None:
                    services['ocr'] = EnglishOCR()
                result, _ = services['ocr'].recognize(buffer)
                text = result['text']
                response = {'text': text, 'lines': result['lines'], 'tokens': [],
                            'image': 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii')}
                if not text.strip():
                    response['message'] = '未识别到英文文字，请尝试更清晰的图片。'
                    return jsonify(response)
                try:
                    if services['detector'] is None:
                        config_path = Path(config or ROOT / 'config.toml').expanduser().resolve()
                        with config_path.open('rb') as file:
                            settings = tomllib.load(file).get('detection', {})
                        backend = settings.get('backend', 'cnn')
                        if backend == 'bert':
                            from bert_detect import BertDetector
                            services['detector'] = BertDetector(weights, config_path)
                            services['weights'] = 'BERT GED · ' + str(services['detector'].weights)
                        elif backend == 'cnn':
                            path = weights or find_latest_weights(config_path)
                            services['detector'] = Detector(path, config_path=config_path)
                            services['weights'] = 'CNN · ' + str(path)
                        else:
                            raise ValueError('detection.backend 只能为 bert 或 cnn')
                    response['tokens'] = detect_text(services['detector'], text)
                    response['model'] = services['weights'] or '自定义模型'
                except Exception:
                    app.logger.exception('错误检测失败')
                    response['detection_error'] = 'OCR 已完成，但错误检测失败。请检查配置中的模型权重、分词器是否完整且与模型兼容，详情见终端。'
                return jsonify(response)
        except Exception:
            app.logger.exception('OCR 失败')
            return jsonify(error='OCR 识别失败，请确认在 macOS 上安装了 requirements-ocr.txt，详情见终端。'), 500

    return app


app = create_app()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='英文作文 OCR 与错误检测网页')
    parser.add_argument('--weights', type=Path, help='指定 best.pt，默认使用配置目录内最新的 best.pt')
    parser.add_argument('--config', type=Path, default=ROOT / 'config.toml')
    parser.add_argument('--port', type=int, default=5000)
    args = parser.parse_args()
    create_app(args.weights, args.config).run(host='127.0.0.1', port=args.port, debug=False)
