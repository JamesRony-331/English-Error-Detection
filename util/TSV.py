

def read_fce_tsv(path):
    label_to_id = {"c": 0, "i": 1}
    samples = []
    tokens = []
    labels = []

    with open(path, encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                if tokens:
                    samples.append((tokens, labels))
                    tokens = []
                    labels = []
                continue

            fields = line.rstrip("\r\n").split("\t")
            if len(fields) != 2 or fields[1] not in label_to_id:
                raise ValueError(f"第 {line_number} 行格式错误：{line!r}")

            token, label = fields
            tokens.append(token)
            labels.append(label_to_id[label])

    # 文件末尾没有空行时，也要保存最后一句。
    if tokens:
        samples.append((tokens, labels))

    return samples
