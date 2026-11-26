import argparse
import csv
import json
from datetime import datetime
from zoneinfo import ZoneInfo
import math
import sys
import tomllib
from time import perf_counter
from pathlib import Path
from util.GloVe import *
from tqdm.auto import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from util.TSV import read_fce_tsv
from model import ErrorModel
from evaluate import Evaluate
from util.loss import loss_normalizer
from util.draw import draw_results


# 加载数据
def LoadData(dataset, shuffle=False, batch_size=32):
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn
    )
    return dataloader


# 补全函数
def collate_fn(batch):
    tokens, labels = zip(*batch)
    x = pad_sequence(tokens, batch_first=True, padding_value=0)
    y = pad_sequence(labels, batch_first=True, padding_value=-100)
    return x, y


# 建立词表
def BuildVocab(samples):
    word_to_id = {"<PAD>": 0, "<UNK>": 1}
    for tokens, labels in samples:
        for token in tokens:
            if token not in word_to_id:
                word_to_id[token] = len(word_to_id)
    return word_to_id


# 数据编码
def EncodeDataset(samples, word_to_id):
    dataset = []
    for tokens, labels in samples:
        token_ids = [
            word_to_id.get(token, word_to_id["<UNK>"])
            for token in tokens
        ]
        dataset.append((
            torch.tensor(token_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long)
        ))
    return dataset


# 训练一个 Epoch
def Train(model, train_loader, criterion, optimizer, description="训练", progress=None):
    model.train()
    total_loss = 0
    total_weight = 0

    bar = tqdm(train_loader, desc=description, unit="batch", **(progress or {}))
    for x, y in bar:
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output.transpose(1, 2), y)
        loss.backward()
        optimizer.step()

        batch_weight = loss_normalizer(y, criterion)
        total_loss += loss.item() * batch_weight
        total_weight += batch_weight
        bar.set_postfix(loss=f"{total_loss / total_weight:.4f}", refresh=False)

    return total_loss / total_weight


def load_config(config_path):
    config_path = Path(config_path).expanduser().resolve()
    with config_path.open("rb") as file:
        config = tomllib.load(file)
    for section, names in {
        "training": ("epochs", "batch_size"),
        "model": ("embedding_dim", "conv_channels", "kernel_size"),
    }.items():
        for name in names:
            value = config[section][name]
            if type(value) is not int or value <= 0:
                raise ValueError(f"{section}.{name} 必须是正整数")
    for name in ("embedding_dropout", "conv_dropout"):
        value = config["model"].get(name, 0.0)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value < 1:
            raise ValueError(f"model.{name} 必须满足 0 <= p < 1")
        config["model"][name] = value
    freeze_embedding = config["model"].get("freeze_embedding", False)
    if type(freeze_embedding) is not bool:
        raise ValueError("model.freeze_embedding 必须为 true 或 false")
    config["model"]["freeze_embedding"] = freeze_embedding
    if config["model"]["kernel_size"] % 2 == 0:
        raise ValueError("model.kernel_size 必须是奇数")
    lr = config["training"]["learning_rate"]
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("training.learning_rate 必须是有限正数")
    weights = config["training"].get("class_weights", [1.0, 1.0])
    if (not isinstance(weights, list) or len(weights) != 2
            or any(isinstance(w, bool) or not isinstance(w, (int, float))
                   or not math.isfinite(w) or w <= 0 for w in weights)):
        raise ValueError("training.class_weights 必须为两个有限正数：[正确词权重, 错误词权重]")
    config["training"]["class_weights"] = weights
    interval = config["progress"]["mininterval"]
    if not math.isfinite(interval) or interval < 0:
        raise ValueError("progress.mininterval 必须是有限非负数")
    for name in ("train", "dev", "test", "output_dir", "pretrained_vectors"):
        path = Path(config["paths"][name]).expanduser()
        config["paths"][name] = str(
            path if path.is_absolute() else config_path.parent / path
        )
    return config


def create_run_dir(output_dir):
    """按北京时间分日期保存，每次运行独立目录；同秒启动也不覆盖。"""
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    date_dir = Path(output_dir) / now.strftime("%Y-%m-%d")
    date_dir.mkdir(parents=True, exist_ok=True)
    name = now.strftime("%H-%M-%S")
    index = 0
    while True:
        run_dir = date_dir / (name if index == 0 else f"{name}_{index}")
        try:
            run_dir.mkdir()
            return run_dir
        except FileExistsError:
            index += 1


def main(config_path=None):
    config_path = Path(config_path or Path(__file__).with_name("config.toml")).expanduser().resolve()
    config = load_config(config_path)
    training = config["training"]
    paths = config["paths"]
    progress = {"disable": not config["progress"]["enabled"],
                "mininterval": config["progress"]["mininterval"],
                "file": sys.stdout, "ncols": 88, "leave": True,
                "ascii": True,
                "bar_format": "{desc}: {percentage:3.0f}%|{bar:12}| {n_fmt}/{total_fmt} "
                              "[{elapsed}<{remaining}{postfix}]"}
    torch.manual_seed(training["seed"])

    train_samples = read_fce_tsv(paths["train"])
    dev_samples = read_fce_tsv(paths["dev"])
    test_samples = read_fce_tsv(paths["test"])
    if not all((train_samples, dev_samples, test_samples)):
        raise ValueError("训练集、验证集和测试集均不能为空")

    # 从配置读取预训练文件，并在加载大文件前检查维度。
    glove_path = Path(paths["pretrained_vectors"])
    with glove_path.open(encoding="utf-8") as file:
        _, vector_dim = map(int, file.readline().split())
    if vector_dim != config["model"]["embedding_dim"]:
        raise ValueError(
            f"预训练词向量为 {vector_dim} 维，但配置 embedding_dim="
            f"{config['model']['embedding_dim']}，请保持一致"
        )

    # 加载预训练词向量
    word_to_id, embedding_matrix = LoadGloVe(glove_path)

    # 合并 FCE 训练集词表
    word_to_id, embedding_matrix = MergeVocab(
        train_samples,
        word_to_id,
        embedding_matrix
    )
    train_dataset = EncodeDataset(train_samples, word_to_id)
    dev_dataset = EncodeDataset(dev_samples, word_to_id)
    test_dataset = EncodeDataset(test_samples, word_to_id)

    train_loader = LoadData(train_dataset, shuffle=True, batch_size=training["batch_size"])
    dev_loader = LoadData(dev_dataset, batch_size=training["batch_size"])
    test_loader = LoadData(test_dataset, batch_size=training["batch_size"])

    print("训练集大小：", len(train_dataset))
    print("验证集大小：", len(dev_dataset))
    print("测试集大小：", len(test_dataset))
    print("词表大小：", len(word_to_id))

    model = ErrorModel(
        embedding_matrix,
        conv_channels=config["model"]["conv_channels"],
        kernel_size=config["model"]["kernel_size"],
        embedding_dropout=config["model"]["embedding_dropout"],
        conv_dropout=config["model"]["conv_dropout"],
        freeze_embedding=config["model"]["freeze_embedding"]
    )
    class_weights = torch.tensor(
        training["class_weights"], dtype=torch.float32,
        device=next(model.parameters()).device
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)
    optimizer = torch.optim.Adam(model.parameters(), lr=training["learning_rate"])

    epochs = training["epochs"]
    best_f05 = -1.0
    run_dir = create_run_dir(paths["output_dir"])
    weights_dir = run_dir / "weights"
    weights_dir.mkdir()
    model_path = weights_dir / "best.pt"
    # 原始配置保留注释，解析后的配置记录绝对路径，方便追溯。
    (run_dir / "config.toml").write_bytes(config_path.read_bytes())
    (run_dir / "config.resolved.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    results_path = run_dir / "results.csv"
    columns = ["epoch", "train_loss", "dev_loss", "dev_accuracy",
               "dev_precision", "dev_recall", "dev_f05",
               "train_eval_loss", "train_accuracy", "train_precision",
               "train_recall", "train_f05", "learning_rate", "epoch_seconds"]
    with results_path.open("w", newline="", encoding="utf-8") as file:
        csv.DictWriter(file, fieldnames=columns).writeheader()
    tqdm.write(f"本次训练结果目录：\n  {run_dir}", file=sys.stdout)
    train_eval_loader = LoadData(train_dataset, batch_size=training["batch_size"])
    for epoch in range(epochs):
        started = perf_counter()
        train_loss = Train(model, train_loader, criterion, optimizer,
                           f"训练 {epoch + 1}/{epochs}", progress)
        metrics = Evaluate(model, dev_loader, criterion,
                           f"验证 {epoch + 1}/{epochs}", progress)
        # 用本轮更新后的固定模型评估训练集，使训练/验证指标口径一致。
        train_metrics = Evaluate(model, train_eval_loader, criterion,
                                 f"训练集评估 {epoch + 1}/{epochs}", progress)
        epoch_seconds = perf_counter() - started
        tqdm.write(
            f"\nEpoch {epoch + 1}/{epochs}\n"
            f"  {'Split':<8}{'Loss':>10}{'Accuracy':>10}{'Precision':>10}"
            f"{'Recall':>10}{'F0.5':>10}\n"
            f"  {'Train':<8}{train_metrics['loss']:>10.4f}{train_metrics['accuracy']:>10.4f}"
            f"{train_metrics['precision']:>10.4f}{train_metrics['recall']:>10.4f}"
            f"{train_metrics['f05']:>10.4f}\n"
            f"  {'Dev':<8}{metrics['loss']:>10.4f}{metrics['accuracy']:>10.4f}"
            f"{metrics['precision']:>10.4f}{metrics['recall']:>10.4f}"
            f"{metrics['f05']:>10.4f}", file=sys.stdout
        )

        with results_path.open("a", newline="", encoding="utf-8") as file:
            csv.DictWriter(file, fieldnames=columns).writerow({
                "epoch": epoch + 1, "train_loss": train_loss,
                **{f"dev_{key}": value for key, value in metrics.items()},
                "train_eval_loss": train_metrics["loss"],
                **{f"train_{key}": value for key, value in train_metrics.items() if key != "loss"},
                "learning_rate": optimizer.param_groups[0]["lr"],
                "epoch_seconds": epoch_seconds
            })
        state = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "word_to_id": word_to_id,
            "epoch": epoch + 1,
            "f05": metrics["f05"],
            "config": config
        }
        # 每轮更新 last.pt；只有验证集 F0.5 提高才更新 best.pt。
        torch.save(state, weights_dir / "last.pt")
        if metrics["f05"] > best_f05:
            best_f05 = metrics["f05"]
            torch.save(state, model_path)
            tqdm.write("  最优模型已更新 → weights/best.pt", file=sys.stdout)

        print("-" * 70, flush=True)

    # 加载最优模型，进行最终测试
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = Evaluate(model, test_loader, criterion, "最终测试", progress, include_confusion=True)

    (run_dir / "test_metrics.json").write_text(
        json.dumps({"best_epoch": checkpoint["epoch"], **test_metrics},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n最优模型轮次：{checkpoint['epoch']}")
    print("最终测试集结果：")
    for key, value in test_metrics.items():
        if key == "confusion_matrix":
            continue
        print(f"  {key:<12} {value:>10.4f}", flush=True)

    for image_path in draw_results(run_dir):
        print(f"已保存图表：{image_path.relative_to(run_dir)}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FCE 英语错误检测训练")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.toml"),
                        help="TOML 配置文件路径")
    main(parser.parse_args().config)
