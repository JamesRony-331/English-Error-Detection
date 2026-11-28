import tempfile
import unittest
from pathlib import Path
from PIL import Image
from OCR import collect_images, pixel_box


class OCRTests(unittest.TestCase):
    def test_coordinates(self):
        self.assertEqual(pixel_box((.1, .2, .3, .4), 100, 200), [10, 80, 40, 160])

    def test_collect_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new('RGB', (20, 20)).save(root / 'a.png')
            (root / 'notes.txt').write_text('hello')
            self.assertEqual(collect_images(root), [root / 'a.png'])
            with self.assertRaises(ValueError):
                collect_images(root / 'notes.txt')
            with self.assertRaises(FileNotFoundError):
                collect_images(root / 'missing.png')
