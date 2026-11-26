import torch


def LoadGloVe(path):

    word_to_id = {
        "<PAD>": 0,
        "<UNK>": 1
    }

    with open(path, encoding="utf-8") as file:

        # 读取第一行
        vocab_size, embedding_dim = map(
            int, file.readline().split()
        )

        # 初始化词向量矩阵
        embedding_matrix = torch.zeros(
            vocab_size + 2,
            embedding_dim
        )

        for line in file:

            fields = line.split()

            word = fields[0]

            vector = [float(v) for v in fields[1:]]

            idx = len(word_to_id)

            word_to_id[word] = idx

            embedding_matrix[idx] = torch.tensor(vector)

    # 使用已有词向量的均值初始化 UNK
    embedding_matrix[1] = embedding_matrix[2:].mean(dim=0)

    return word_to_id, embedding_matrix

def MergeVocab(samples, word_to_id, embedding_matrix):

    embedding_dim = embedding_matrix.shape[1]

    new_vectors = []

    exact_count = 0
    lowercase_count = 0
    unknown_count = 0

    # 获取训练集中所有不同的单词
    words = dict.fromkeys(
        token
        for tokens, labels in samples
        for token in tokens
    )

    for token in words:

        # GloVe 中已经存在
        if token in word_to_id:
            exact_count += 1
            continue

        # 分配新的 ID
        new_id = len(word_to_id)

        lower_token = token.lower()

        if lower_token in word_to_id:

            # 复制已有的小写词向量
            idx = word_to_id[lower_token]

            if idx < embedding_matrix.shape[0]:
                vector = embedding_matrix[idx].clone()
            else:
                vector = new_vectors[
                    idx - embedding_matrix.shape[0]
                ].clone()

            lowercase_count += 1

        else:

            # 未知单词随机初始化
            vector = torch.randn(embedding_dim) * 0.1

            unknown_count += 1

        word_to_id[token] = new_id
        new_vectors.append(vector)

    # 将新词向量添加到原矩阵末尾
    if new_vectors:
        embedding_matrix = torch.cat(
            [
                embedding_matrix,
                torch.stack(new_vectors)
            ],
            dim=0
        )

    print("直接匹配词数：", exact_count)
    print("小写匹配词数：", lowercase_count)
    print("随机初始化词数：", unknown_count)
    print("合并后词表大小：", len(word_to_id))
    print("合并后矩阵形状：", embedding_matrix.shape)

    assert len(word_to_id) == embedding_matrix.shape[0]

    return word_to_id, embedding_matrix
if __name__ == "__main__":

    from pathlib import Path

    path = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "glove-wiki-gigaword-100"
    )

    word_to_id, embedding_matrix = LoadGloVe(path)

    print("词表大小：", len(word_to_id))

    print("词向量矩阵：", embedding_matrix.shape)

    print("apple 的 ID：", word_to_id["apple"])

    print("apple 的词向量：",
          embedding_matrix[word_to_id["apple"]])

    print("词表前20项：",
          list(word_to_id.items())[:20])