"""步驟 4：語意檢索（Retrieval）—— RAG 裡的 R。

把步驟 2（向量）和步驟 3（切段）組合起來：
  1. 每個段落轉成向量，存成「索引」（只需做一次）。
  2. 問題也轉成向量。
  3. 算問題和每一段的相似度，取最高的 k 段。

執行：python learn/step4_search.py "你的問題"
"""
import json
import sys
from pathlib import Path

from step2_embed import cosine, embed
from step3_chunk import pdf_to_chunks

PDF_PATH = 'data/pdfs/paper.pdf'
INDEX_PATH = Path('learn/my_index.json')  # 練習用索引，跟網頁版的分開


def build_index():
    """切段 → 嵌入 → 存檔。已存在就直接讀，避免每次重算。"""
    if INDEX_PATH.exists():
        return json.loads(INDEX_PATH.read_text(encoding='utf-8'))
    chunks = pdf_to_chunks(PDF_PATH)
    print(f'第一次執行：正在嵌入 {len(chunks)} 段，請稍候…')
    for start in range(0, len(chunks), 16):  # 分批送，避免一次太大
        batch = chunks[start:start + 16]
        vectors = embed([f'title: none | text: {c["text"]}' for c in batch])
        for chunk, vector in zip(batch, vectors):
            chunk['vector'] = vector
    INDEX_PATH.write_text(json.dumps(chunks, ensure_ascii=False), encoding='utf-8')
    return chunks


def search(index, question, top_k=3):
    """回傳最相關的 top_k 段，每段附上相似度分數。"""
    query_vector = embed([f'task: search result | query: {question}'])[0]
    scored = [{**c, 'score': cosine(query_vector, c['vector'])} for c in index]
    return sorted(scored, key=lambda c: c['score'], reverse=True)[:top_k]


if __name__ == '__main__':
    question = sys.argv[1] if len(sys.argv) > 1 else '這篇論文使用了哪些資料？'
    for hit in search(build_index(), question):
        print(f'\n[{hit["score"]:.3f}] 第 {hit["page"]} 頁\n{hit["text"][:250]}…')
