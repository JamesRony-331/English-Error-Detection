"""读取训练结果，用 matplotlib 绘图；也可单独运行本文件重新生成图片。"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 训练结束直接保存图片，无需桌面窗口。
import matplotlib.pyplot as plt


def draw_results(run_dir):
    """从 results.csv 和 best.pt 的测试结果生成图表，返回图片路径。"""
    run_dir = Path(run_dir)
    with (run_dir / "results.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError("results.csv 中没有训练记录")
    epochs = [int(row["epoch"]) for row in rows]
    test = json.loads((run_dir / "test_metrics.json").read_text(encoding="utf-8"))
    plots = run_dir / "plots"
    plots.mkdir(exist_ok=True)
    output_paths = []

    def save_figure(fig, name):
        path = plots / name
        fig.savefig(path, dpi=180)
        plt.close(fig)
        output_paths.append(path)
    with plt.rc_context({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False}):
        for key, title in [
            ("loss", "Loss"), ("accuracy", "Accuracy"), ("precision", "Precision"),
            ("recall", "Recall"), ("f05", "F0.5")
        ]:
            fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
            for split, color in [("train", "#2563eb"), ("dev", "#ea580c")]:
                column = f"{split}_{key}"
                if split == "train" and key == "loss" and "train_eval_loss" in rows[0]:
                    column = "train_eval_loss"
                if column in rows[0]:
                    ax.plot(epochs, [float(row[column]) for row in rows],
                            label=split, color=color, marker="." )
            if key == "loss" and "train_eval_loss" in rows[0]:
                ax.plot(epochs, [float(row["train_loss"]) for row in rows],
                        label="train (during updates)", color="#94a3b8", ls=":")
            ax.axvline(test["best_epoch"], color="#64748b", ls="--", alpha=.6)
            ax.set(title=title, xlabel="Epoch")
            if key != "loss":
                ax.set_ylim(0, 1.05)
            ax.grid(alpha=.2)
            ax.legend()
            fig.suptitle(f"Training history | best epoch {test['best_epoch']}")
            save_figure(fig, f"{key}.png")
        if "epoch_seconds" in rows[0]:
            fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
            ax.plot(epochs, [float(row["epoch_seconds"]) for row in rows], color="#059669", marker=".")
            ax.set(title="Epoch duration", xlabel="Epoch", ylabel="Seconds")
            ax.grid(alpha=.2)
            save_figure(fig, "epoch_duration.png")

        fig, ax = plt.subplots(figsize=(7, 6), layout="constrained")
        matrix = test["confusion_matrix"]  # 行是真实标签，列是预测标签。
        im = ax.imshow(matrix, cmap="Blues", vmin=0)
        maximum = max(max(row) for row in matrix)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(matrix[i][j]), ha="center", va="center", fontsize=16,
                        color="white" if matrix[i][j] > maximum / 2 else "black")
        labels = ["Correct (0)", "Error (1)"]
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=labels, yticklabels=labels,
               xlabel="Predicted label", ylabel="True label", title="Test confusion matrix")
        fig.colorbar(im, ax=ax, shrink=.8)
        fig.suptitle(f"best.pt | epoch {test['best_epoch']} | padding excluded")
        save_figure(fig, "test_confusion_matrix.png")

        fig, scores = plt.subplots(figsize=(8, 5), layout="constrained")
        names = ["accuracy", "precision", "recall", "f05"]
        bars = scores.bar(["Accuracy", "Precision", "Recall", "F0.5"],
                          [test[name] for name in names], color="#2563eb", width=.6)
        scores.bar_label(bars, fmt="%.4f", padding=4)
        scores.set(ylim=(0, 1.15), title=f"Test metrics | loss = {test['loss']:.4f}")
        scores.grid(axis="y", alpha=.2)
        scores.set_axisbelow(True)
        fig.suptitle(f"best.pt | epoch {test['best_epoch']} | padding excluded")
        save_figure(fig, "test_metrics.png")
    return output_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从一次训练结果目录重新绘图")
    parser.add_argument("run_dir", type=Path, help="包含 results.csv 和 test_metrics.json 的目录")
    for path in draw_results(parser.parse_args().run_dir):
        print(path)
