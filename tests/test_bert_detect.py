import unittest
import torch
from bert_detect import BertDetector

class BertTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.detector = BertDetector()

    def test_known_vocabulary_and_error_label(self):
        self.assertEqual(len(self.detector.tokenizer), 30522)
        result = self.detector.predict('She go to school every day.')
        self.assertTrue(next(t for t in result if t['token']=='go')['error'])
        self.assertFalse(any(t['unknown'] for t in result))

    def test_alignment_long_sentence_and_batch(self):
        words = ['hello'] * 520 + ['unbelievably', '.']
        output = self.detector.predict_words_batch([words])[0]
        self.assertEqual(len(output), len(words))
        self.assertTrue(all(len(p)==3 for p in output))
        batch = self.detector.predict_words_batch([['She','go','.'], ['Hello','.']])
        single = self.detector.predict_words_batch([['She','go','.']])[0]
        torch.testing.assert_close(torch.tensor(batch[0]),torch.tensor(single),atol=1e-5,rtol=1e-4)
