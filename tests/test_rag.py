import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient
from pypdf import PdfWriter
from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject

from app.rag import (CHUNK_OVERLAP, CHUNK_SIZE, Rag, RagError, clean_text, extract_chunks,
                     split_text, strip_thinking, with_prefix)
from app import main


def pdf_bytes(texts):
    writer = PdfWriter()
    for text in texts:
        page = writer.add_blank_page(width=600, height=800)
        font = DictionaryObject({NameObject('/Type'): NameObject('/Font'),
                                 NameObject('/Subtype'): NameObject('/Type1'),
                                 NameObject('/BaseFont'): NameObject('/Helvetica')})
        page[NameObject('/Resources')] = DictionaryObject({NameObject('/Font'): DictionaryObject({NameObject('/F1'): font})})
        stream = DecodedStreamObject()
        stream.set_data(f'BT /F1 12 Tf 50 700 Td ({text}) Tj ET'.encode())
        page[NameObject('/Contents')] = writer._add_object(stream)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


class RagTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.rag = Rag(root / 'paper.pdf', root / 'index.json', 'http://localhost:11434', 'test', 'embed')
        self.data = pdf_bytes(['Apples are red.', 'Bananas are yellow.'])

    def tearDown(self):
        self.tmp.cleanup()

    def build(self):
        with patch.object(self.rag, 'embed', return_value=[[1., 0.], [0., 1.]]):
            self.rag.build(self.data)

    def test_extract_real_pdf_pages(self):
        chunks, pages, empty = extract_chunks(self.data)
        self.assertEqual(pages, 2)
        self.assertEqual(empty, [])
        self.assertEqual(chunks[1]['page'], 2)
        self.assertIn('Bananas', chunks[1]['text'])

    def test_text_helpers(self):
        self.assertEqual(clean_text('摘要 . . . . . . i\n  Abstract'), '摘要 i Abstract')
        pieces = split_text('x' * 2500)
        self.assertTrue(all(len(p) <= CHUNK_SIZE for p in pieces))
        self.assertEqual(sum(len(p) for p in pieces) - CHUNK_OVERLAP * (len(pieces) - 1), 2500)
        self.assertEqual(strip_thinking('<think>\nhmm\n</think>\n答案 [1]'), '答案 [1]')
        self.assertTrue(with_prefix('embeddinggemma', 'q', 'query').startswith('task: search result'))
        self.assertEqual(with_prefix('nomic-embed-text', 'q', 'query'), 'q')

    def test_invalid_and_scan_pdf(self):
        for data in [b'invalid', pdf_bytes([''])]:
            with self.assertRaises(RagError):
                extract_chunks(data)

    def test_retrieval_and_citations(self):
        self.build()
        with patch.object(self.rag, 'embed', return_value=[[0., 1.]]), patch.object(self.rag, 'post', return_value={'message': {'content': '香蕉是黃色 [1]。'}}) as post:
            answer = self.rag.ask('Bananas?', 1)
        self.assertEqual(answer['sources'][0]['page'], 2)
        self.assertIsNone(answer['warning'])
        self.assertIn('Bananas', post.call_args.args[1]['messages'][1]['content'])

    def test_cache_and_pdf_change(self):
        self.build()
        args = (self.rag.pdf, self.rag.index, self.rag.url, 'test', 'embed')
        self.assertTrue(Rag(*args).status()['ready'])
        self.assertFalse(Rag(*args[:-1], 'other-embedding').status()['ready'])
        self.rag.pdf.write_bytes(pdf_bytes(['Different document']))
        self.assertFalse(Rag(*args).status()['ready'])

    def test_failed_rebuild_keeps_previous(self):
        self.build()
        previous = self.rag.snapshot
        with patch.object(self.rag, 'embed', side_effect=RagError('offline')):
            with self.assertRaises(RagError):
                self.rag.build(pdf_bytes(['New paper']))
        self.assertIs(self.rag.snapshot, previous)
        self.assertEqual(self.rag.pdf.read_bytes(), self.data)
        self.assertFalse(self.rag.loading)

    def test_concurrent_build_rejected(self):
        self.rag.lock.acquire()
        try:
            with self.assertRaises(RagError):
                self.rag.build(self.data)
        finally:
            self.rag.lock.release()

    def test_model_unavailable(self):
        with patch('httpx.Client.post', side_effect=httpx.ConnectError('offline')):
            with self.assertRaisesRegex(RagError, 'Ollama'):
                self.rag.embed(['test'])

    def test_invalid_citation_warning(self):
        self.build()
        with patch.object(self.rag, 'embed', return_value=[[0., 1.]]), patch.object(self.rag, 'post', return_value={'message': {'content': 'Unsupported citation [99]'}}):
            self.assertIsNotNone(self.rag.ask('Bananas?', 1)['warning'])

    def test_api(self):
        with patch.object(main, 'rag', self.rag), TestClient(main.app) as client:
            self.assertEqual(client.get('/').status_code, 200)
            self.assertEqual(client.get('/static/app.js').status_code, 200)
            self.assertFalse(client.get('/api/status').json()['ready'])
            self.assertEqual(client.get('/api/pdf').status_code, 404)
            self.assertEqual(client.post('/api/chat', json={'question': ' '}).status_code, 422)
            self.assertEqual(client.post('/api/chat', json={'question': 'hello'}).status_code, 503)
            self.assertEqual(client.post('/api/chat', json={'question': 'hello', 'top_k': 9}).status_code, 422)
            self.assertEqual(client.post('/api/index').status_code, 400)
            self.assertEqual(client.post('/api/upload', files={'file': ('x.pdf', b'bad')}).status_code, 400)
            with patch.object(self.rag, 'embed', return_value=[[1., 0.], [0., 1.]]):
                result = client.post('/api/upload', files={'file': ('../../paper.pdf', self.data, 'application/pdf')})
            self.assertEqual(result.status_code, 200)
            self.assertFalse(result.json()['indexing'])
            self.assertEqual(client.get('/api/pdf').content, self.data)


if __name__ == '__main__':
    unittest.main()
