"""模型评估：供训练集评估、验证和最终测试共用。

Evaluate 返回 loss、accuracy、precision、recall、f05；
include_confusion=True 时额外返回混淆矩阵，行是真实标签、列是预测标签。
调用方应传入 CrossEntropyLoss(ignore_index=-100)，与标签补齐值一致。
"""
import torch
from tqdm.auto import tqdm
from util.loss import loss_normalizer


# 验证集和测试集评估，不更新模型参数
@torch.inference_mode()
def Evaluate(model, dataloader, criterion, description="评估", progress=None, include_confusion=False):
    model.eval()
    total_loss = 0
    total_tokens = 0
    total_weight = 0
    TP = FP = TN = FN = 0

    bar = tqdm(dataloader, desc=description, unit="batch", **(progress or {}))
    for x, y in bar:
        output = model(x)
        loss = criterion(output.transpose(1, 2), y)
        predictions = output.argmax(dim=-1)

        # 排除填充位置
        mask = y != -100
        predictions = predictions[mask]
        labels = y[mask]

        valid_tokens = labels.numel()
        batch_weight = loss_normalizer(y, criterion)
        total_loss += loss.item() * batch_weight
        total_weight += batch_weight
        total_tokens += valid_tokens
        bar.set_postfix(loss=f"{total_loss / total_weight:.4f}", refresh=False)

        TP += ((predictions == 1) & (labels == 1)).sum().item()
        FP += ((predictions == 1) & (labels == 0)).sum().item()
        TN += ((predictions == 0) & (labels == 0)).sum().item()
        FN += ((predictions == 0) & (labels == 1)).sum().item()

    accuracy = (TP + TN) / total_tokens
    precision = TP / (TP + FP) if TP + FP else 0.0
    recall = TP / (TP + FN) if TP + FN else 0.0
    f05 = (1.25 * precision * recall / (0.25 * precision + recall)
           if precision + recall else 0.0)

    metrics = {
        "loss": total_loss / total_weight,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f05": f05
    }
    if include_confusion:
        metrics["confusion_matrix"] = [[TN, FP], [FN, TP]]
    return metrics
