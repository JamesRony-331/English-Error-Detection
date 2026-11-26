"""英文逐词错误检测：直接运行可交互输入，也支持 --text 和 --weights。"""
import argparse
from pathlib import Path
import re
import tomllib

import torch
from model import ErrorModel

PROJECT_DIR = Path(__file__).resolve().parent


def find_latest_weights(config_path):
    """查找配置的结果目录中最近保存的 best.pt。"""
    config_path = Path(config_path).expanduser().resolve()
    with config_path.open("rb") as file:
        config = tomllib.load(file)
    root = Path(config["paths"]["output_dir"]).expanduser()
    if not root.is_absolute():
        root = config_path.parent / root
    candidates = list(root.glob("*/*/weights/best.pt"))
    if not candidates:
        raise FileNotFoundError(f"没有找到 best.pt：{root}；请先训练或用 --weights 指定模型")
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def tokenize(text):
    """保留大小写、标点和字符位置；拆分常见英文缩写（近似训练集分词）。"""
    tokens = []
    for match in re.finditer(r"[A-Za-z]+(?:['’][A-Za-z]+)?|\d+(?:\.\d+)?|[^\w\s]|\w+", text):
        word = match.group()
        suffix = re.search(r"(?:n['’]t|['’](?:s|re|ve|ll|d|m))$", word, re.I)
        spans = [(0, len(word))]
        if suffix and suffix.start() > 0:
            spans = [(0, suffix.start()), (suffix.start(), len(word))]
        for start, end in spans:
            tokens.append((word[start:end].replace('’', "'"), match.start() + start, match.start() + end))
    return tokens


class Detector:
    def __init__(self, weights):
        checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
        self.word_to_id = checkpoint["word_to_id"]
        self.epoch = checkpoint.get("epoch", "未知")
        # 从实际权重恢复架构，兼容没有保存 config 的早期检查点。
        state = checkpoint["model_state_dict"]
        model_config = checkpoint.get("config", {}).get("model", {})
        self.model = ErrorModel(
            state["embedding.weight"],
            conv_channels=state["conv.weight"].shape[0],
            kernel_size=state["conv.weight"].shape[2],
            embedding_dropout=model_config.get("embedding_dropout", 0.0),
            conv_dropout=model_config.get("conv_dropout", 0.0),
            freeze_embedding=model_config.get("freeze_embedding", False))
        self.model.load_state_dict(state)
        self.model.eval()

    @torch.inference_mode()
    def predict(self, text):
        tokens = tokenize(text)
        if not tokens:
            raise ValueError("请输入包含单词或标点的句子")
        ids = [self.word_to_id.get(word, self.word_to_id["<UNK>"]) for word, _, _ in tokens]
        output = self.model(torch.tensor([ids], dtype=torch.long))
        probabilities = output.softmax(dim=-1)[0, :, 1].tolist()
        predictions = output.argmax(dim=-1)[0].tolist()
        return [dict(token=word, start=start, end=end, error=bool(label),
                     probability=probability, unknown=word not in self.word_to_id)
                for (word, start, end), label, probability in zip(tokens, predictions, probabilities)]


def display(text, results):
    chunks = []
    cursor = 0
    for result in results:
        start, end = result["start"], result["end"]
        chunks.append(text[cursor:start])
        word = text[start:end]
        chunks.append(f"【{word}】" if result["error"] else word)
        cursor = end
    chunks.append(text[cursor:])
    print("\n检测结果：", "".join(chunks))
    print("序号  Token                  预测      错误概率")
    print("-" * 56)
    for i, result in enumerate(results, 1):
        label = "疑似错误" if result["error"] else "正确    "
        extra = "  [未登录词]" if result["unknown"] else ""
        print(f"{i:>4}  {result['token']:<22} {label}  {result['probability']:>8.2%}{extra}")
    errors = [f"{i}: {r['token']}" for i, r in enumerate(results, 1) if r["error"]]
    print("疑似错误位置：" + ("；".join(errors) if errors else "未检测到"))
    print("说明：结果是模型预测；缺词错误可能标在后一个词上，未登录词使用 UNK 表示。")


def main():
    parser = argparse.ArgumentParser(description="输入一句英文，标出模型预测的错误词")
    parser.add_argument("--weights", type=Path, help="指定 best.pt；默认使用结果目录最近保存的 best.pt")
    parser.add_argument("--config", type=Path, default=PROJECT_DIR / "config.toml")
    parser.add_argument("--text", help="直接检测一句话；省略则进入交互模式")
    args = parser.parse_args()
    try:
        weights = args.weights.expanduser().resolve() if args.weights else find_latest_weights(args.config)
        detector = Detector(weights)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.exit(1, f"模型加载失败：{error}\n")
    print(f"模型：{weights}\n模型轮次：{detector.epoch}")
    print("分词采用轻量规则，与训练集的 RASP 分词可能存在差异。")
    if args.text is not None:
        try:
            display(args.text, detector.predict(args.text))
        except ValueError as error:
            parser.exit(1, f"{error}\n")
        return
    while True:
        try:
            text = input("\n输入英文句子（输入 q 退出）：")
            if text.strip().lower() in {"q", "quit", "exit"}:
                break
            display(text, detector.predict(text))
        except ValueError as error:
            print(error)
        except (EOFError, KeyboardInterrupt):
            print("\n已退出")
            break


if __name__ == "__main__":
    main()
