import tempfile
import unittest
from pathlib import Path
import torch
from model import ErrorModel
from detect import Detector


class DilatedModelTests(unittest.TestCase):
    def test_padding_and_training(self):
        for kernel in (3, 7):
            torch.manual_seed(42)
            model = ErrorModel(torch.randn(12, 4), conv_channels=6,
                               kernel_size=kernel, dilations=[1, 2, 4],
                               freeze_embedding=True)
            model.eval()
            short = torch.tensor([[1, 2, 3]])
            padded = torch.tensor([[1, 2, 3, 0, 0, 0, 0, 0]])
            torch.testing.assert_close(model(short), model(padded)[:, :3])
            self.assertEqual(model(padded).shape, (1, 8, 2))
            before = model.embedding.weight.detach().clone()
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=.01)
            loss = torch.nn.functional.cross_entropy(model(short).transpose(1, 2), torch.tensor([[0, 1, 0]]))
            loss.backward()
            for conv in model.convs:
                self.assertGreater(conv.weight.grad.abs().sum().item(), 0)
            optimizer.step()
            torch.testing.assert_close(before, model.embedding.weight)

    def test_checkpoint_roundtrip(self):
        for legacy in (False, True):
            model = ErrorModel(torch.randn(5, 4), conv_channels=6,
                               dilations=[1] if legacy else [1, 2, 4])
            model.eval()
            state = model.state_dict()
            config = {'model': {'dilations': [1, 2, 4]}}
            if legacy:
                state = {key.replace('convs.0.', 'conv.'): value for key, value in state.items()}
                config = {}
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'best.pt'
                torch.save({'model_state_dict': state, 'config': config,
                            'word_to_id': {'<PAD>': 0, '<UNK>': 1, 'hello': 2}}, path)
                detector = Detector(path)
                x = torch.tensor([[2, 1]])
                torch.testing.assert_close(model(x), detector.model(x))

    def test_invalid_dilations(self):
        for value in ([], [0], [-1], [True], [1.5]):
            with self.assertRaises(ValueError):
                ErrorModel(torch.randn(5, 4), dilations=value)


if __name__ == '__main__':
    unittest.main()
