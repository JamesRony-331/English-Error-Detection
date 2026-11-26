"""训练和评估共用的交叉熵损失汇总工具。"""


def loss_normalizer(labels, criterion):
    """还原 mean 交叉熵的分母：有效标签数，或有效标签的类别权重之和。"""
    if criterion.reduction != "mean":
        raise ValueError("损失汇总要求 CrossEntropyLoss 使用 reduction=mean")
    valid_labels = labels[labels != criterion.ignore_index]
    if criterion.weight is None:
        return valid_labels.numel()
    return criterion.weight[valid_labels].sum().item()
