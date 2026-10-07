"""步驟 2：把文字變成向量（embedding），用來比較「意思」是否相近。

嵌入模型把一段文字轉成一串數字（這裡是 768 個）。
意思相近的文字，向量的方向也相近；用 cosine similarity 衡量，越接近 1 越相似。

執行：python learn/step2_embed.py
"""
import math
import os

import httpx

OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://127.0.0.1:11434')
EMBED_MODEL = os.getenv('EMBED_MODEL', 'embeddinggemma')


def embed(texts):
    """呼叫 Ollama 的 /api/embed，一次把多段文字轉成向量。"""
    response = httpx.post(f'{OLLAMA_URL}/api/embed', timeout=300,
                          json={'model': EMBED_MODEL, 'input': texts})
    response.raise_for_status()
    return response.json()['embeddings']


def cosine(a, b):
    """cosine similarity = 兩向量內積 ÷ 兩向量長度相乘。"""
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


if __name__ == '__main__':
    # embeddinggemma 建議在「問題」和「文件」前加上不同的前綴，檢索效果較好。
    query = 'task: search result | query: 怎麼預測明天會不會下雨？'
    documents = [
        '氣象模型利用歷史雨量資料預測未來降水。',
        '圖書館週末延長開放時間。',
        '這家拉麵店的湯頭非常濃郁。',
        'Weather forecasting uses past rainfall data to predict precipitation.',
    ]
    vectors = embed([query] + [f'title: none | text: {d}' for d in documents])
    query_vector, doc_vectors = vectors[0], vectors[1:]

    print(f'每個向量有 {len(query_vector)} 個數字，前 5 個：{[round(x, 3) for x in query_vector[:5]]}\n')
    print('與問題的相似度（越高越相關）：')
    for doc, vector in sorted(zip(documents, doc_vectors), key=lambda p: -cosine(query_vector, p[1])):
        print(f'  {cosine(query_vector, vector):.3f}  {doc}')
    # 注意：英文句子也會排在前面——向量比較的是「意思」，不是關鍵字。
