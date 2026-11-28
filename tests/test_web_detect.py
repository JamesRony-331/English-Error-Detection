import io
import unittest
from PIL import Image
from web_detect import create_app, detect_text

class FakeOCR:
    def recognize(self, image):
        return {'text':'She go.\nHe run.', 'lines':[]}, None
class FakeDetector:
    def predict(self, text):
        from detect import tokenize
        return [dict(token=t,start=s,end=e,error=t=='go',probability=.8,unknown=False) for t,s,e in tokenize(text)]
class WebTests(unittest.TestCase):
    def setUp(self):
        self.client=create_app(ocr=FakeOCR(),detector=FakeDetector()).test_client()
    def test_upload(self):
        image=io.BytesIO();Image.new('RGB',(20,20),'white').save(image,format='PNG');image.seek(0)
        response=self.client.post('/api/recognize',data={'image':(image,'image.png')})
        self.assertEqual(response.status_code,200)
        data=response.get_json()
        self.assertEqual(data['text'],'She go.\nHe run.')
        for token in data['tokens']:
            self.assertEqual(data['text'][token['start']:token['end']],token['token'])
        self.assertEqual(sum(t['error'] for t in data['tokens']),1)
    def test_invalid_upload(self):
        self.assertEqual(self.client.post('/api/recognize').status_code,400)
        self.assertEqual(self.client.post('/api/recognize',data={'image':(io.BytesIO(b'bad'),'image.png')}).status_code,400)
    def test_page(self):
        self.assertEqual(self.client.get('/').status_code,200)
