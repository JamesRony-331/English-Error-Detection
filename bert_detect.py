"""作者 sahilnishad 的 BERT GED 模型适配：i=0、c=1、2 为额外输出。
每个原始词取首个子词，特殊符号不参与输出；长输入按完整词分块。
"""
from pathlib import Path
import tomllib
import torch
from transformers import BertConfig, BertForTokenClassification, BertTokenizerFast
from detect import tokenize

ROOT = Path(__file__).resolve().parent

class BertDetector:
    def __init__(self, weights=None, config_path=None):
        config_path = Path(config_path or ROOT / 'config.toml').expanduser().resolve()
        with config_path.open('rb') as f:
            config = tomllib.load(f)
        settings = config.get('detection', {})
        self.threshold = settings.get('threshold', .3)
        if isinstance(self.threshold, bool) or not isinstance(self.threshold, (int, float)) or not 0 <= self.threshold <= 1:
            raise ValueError('detection.threshold 必须在 [0,1] 内')
        directory = Path(settings.get('bert_dir', 'models/comparison')).expanduser()
        if not directory.is_absolute(): directory = config_path.parent / directory
        self.weights = Path(weights or directory / 'model.pt')
        self.tokenizer = BertTokenizerFast.from_pretrained(str(directory), do_lower_case=True, local_files_only=True)
        architecture = BertConfig.from_json_file(str(directory / 'config.json'))
        if len(self.tokenizer) != architecture.vocab_size or self.tokenizer.convert_tokens_to_ids("the") != 1996:
            raise ValueError("BERT 分词器词表不匹配")
        self.model = BertForTokenClassification(architecture)
        self.model.load_state_dict(torch.load(self.weights, map_location='cpu', weights_only=True), strict=True)
        self.model.eval()
        self.max_length = min(256, architecture.max_position_embeddings)

    @torch.inference_mode()
    def predict_words_batch(self, sentences, batch_size=16):
        """返回每个原始词的三类概率，批处理用于完整测试集评估。"""
        chunks = []
        outputs = [[None] * len(words) for words in sentences]
        for sentence_id, words in enumerate(sentences):
            start, pieces = 0, 0
            for index, word in enumerate(words):
                count = len(self.tokenizer.encode(word, add_special_tokens=False))
                if count > self.max_length - 2:
                    raise ValueError('单词分词后长度超过模型限制')
                if pieces + count > self.max_length - 2:
                    chunks.append((sentence_id, start, words[start:index]))
                    start, pieces = index, 0
                pieces += count
            if start < len(words): chunks.append((sentence_id, start, words[start:]))
        for begin in range(0, len(chunks), batch_size):
            group = chunks[begin:begin + batch_size]
            encoded = self.tokenizer([c[2] for c in group], is_split_into_words=True,
                                     padding=True, return_tensors='pt')
            if encoded['input_ids'].shape[1] > self.max_length:
                raise ValueError('分词结果超过最大长度，未静默截断')
            probabilities = self.model(**encoded).logits.softmax(-1).tolist()
            for row, (sentence_id, start, words) in enumerate(group):
                seen = set()
                for position, word_id in enumerate(encoded.word_ids(row)):
                    if word_id is None or word_id in seen: continue
                    seen.add(word_id)
                    outputs[sentence_id][start + word_id] = probabilities[row][position]
        if any(p is None for row in outputs for p in row):
            raise ValueError('存在无法对齐的原始词')
        return outputs

    def predict(self, text):
        spans = tokenize(text)
        if not spans: raise ValueError('请输入包含单词或标点的句子')
        probabilities = self.predict_words_batch([[word for word, _, _ in spans]])[0]
        return [dict(token=word, start=start, end=end,
                     error=p[0] >= self.threshold, probability=p[0],
                     unused_probability=p[2],
                     unknown=self.tokenizer.unk_token_id in self.tokenizer.encode(word, add_special_tokens=False))
                for (word, start, end), p in zip(spans, probabilities)]
