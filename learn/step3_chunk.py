"""步驟 3：把 PDF 切成小段落（chunking）。

為什麼要切？
  1. 嵌入模型一次能讀的長度有限。
  2. 整篇論文塞不進 LLM 的提示；只放「最相關的幾段」，回答更準、更快。
相鄰段落保留一點重疊（overlap），避免一句話剛好被切斷而失去意思。

執行：python learn/step3_chunk.py [PDF路徑]
"""
import re
import sys

from pypdf import PdfReader

CHUNK_SIZE = 1100  # 每段最多幾個字元
OVERLAP = 250      # 相鄰兩段重疊幾個字元


def pdf_to_chunks(path):
    """逐頁擷取文字，再切成帶重疊的段落；每段記住自己在第幾頁。"""
    chunks = []
    for page_no, page in enumerate(PdfReader(path).pages, 1):
        text = page.extract_text() or ''
        text = re.sub(r'(?:\.\s*){4,}', ' ', text)    # 去掉目錄頁的「. . . . .」雜訊
        text = re.sub(r'\s+', ' ', text).strip()      # 合併多餘空白與換行
        start = 0
        while start < len(text):
            chunks.append({'page': page_no, 'text': text[start:start + CHUNK_SIZE]})
            if start + CHUNK_SIZE >= len(text):
                break
            start += CHUNK_SIZE - OVERLAP  # 下一段往回退 OVERLAP 個字再開始
    return chunks


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else 'data/pdfs/paper.pdf'
    chunks = pdf_to_chunks(path)
    print(f'共切出 {len(chunks)} 段，來自 {chunks[-1]["page"]} 頁。')
    for chunk in chunks[:3]:
        print(f'\n--- 第 {chunk["page"]} 頁（{len(chunk["text"])} 字）---\n{chunk["text"][:200]}…')
