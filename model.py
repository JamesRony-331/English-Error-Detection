import torch
from torch import nn


class ErrorModel(nn.Module):

    def __init__(self, embedding_matrix, conv_channels=128, kernel_size=3,
                 embedding_dropout=0.0, conv_dropout=0.0, freeze_embedding=False):
        super().__init__()

        if type(conv_channels) is not int or conv_channels <= 0:
            raise ValueError("conv_channels 必须是正整数")
        if type(kernel_size) is not int or kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size 必须是正奇数")

        for name, value in (("embedding_dropout", embedding_dropout), ("conv_dropout", conv_dropout)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value < 1:
                raise ValueError(f"{name} 必须满足 0 <= p < 1")

        if type(freeze_embedding) is not bool:
            raise ValueError("freeze_embedding 必须为布尔值")

        # 自动获取词向量维度
        embedding_dim = embedding_matrix.shape[1]

        # 加载预训练词向量
        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix,
            freeze=freeze_embedding,
            padding_idx=0
        )

        self.embedding_dropout = nn.Dropout(embedding_dropout)
        self.conv_dropout = nn.Dropout(conv_dropout)

        self.conv = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=conv_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )

        self.relu = nn.ReLU()
        self.fc = nn.Linear(conv_channels, 2)

    def forward(self, x):

        x = self.embedding(x)
        x = self.embedding_dropout(x)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.relu(x)
        x = self.conv_dropout(x)
        x = x.transpose(1, 2)
        x = self.fc(x)

        return x