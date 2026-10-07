"""步驟 5：完整的 RAG = 檢索（Retrieval）+ 增強提示（Augmented）+ 生成（Generation）。

  1. Retrieval  ：用步驟 4 找出最相關的段落。
  2. Augmented  ：把段落貼進提示，並要求模型「只根據這些資料回答」。
  3. Generation ：用步驟 1 的 chat() 產生答案，並以 [1]、[2] 引用來源。

程式會先讓模型「不看論文」直接回答，再用 RAG 回答，讓你比較差異。

執行：python learn/step5_rag.py "你的問題"
"""
import sys

from step1_chat import chat
from step4_search import build_index, search

SYSTEM_PROMPT = (
    '你是論文閱讀助手，使用繁體中文回答。'
    '只能根據提供的摘錄回答，每個重點用 [1] 這類編號標示來源。'
    '摘錄裡找不到答案時，直接說「論文摘錄中沒有提到」，不要編造。'
)


def rag_answer(question, top_k=4):
    hits = search(build_index(), question, top_k)                      # 1. 檢索
    context = '\n\n'.join(f'[{i}] 第 {h["page"]} 頁：{h["text"]}'
                          for i, h in enumerate(hits, 1))
    prompt = f'摘錄：\n{context}\n\n問題：{question}'                  # 2. 組合提示
    answer = chat([{'role': 'system', 'content': SYSTEM_PROMPT},       # 3. 生成
                   {'role': 'user', 'content': prompt}], temperature=0.1)
    return answer, hits


if __name__ == '__main__':
    question = sys.argv[1] if len(sys.argv) > 1 else '這篇論文的研究目的與使用的資料是什麼？'

    print('=== 沒有 RAG（模型只能靠自己的記憶猜）===')
    print(chat([{'role': 'user', 'content': f'用繁體中文簡短回答：{question}'}]))

    print('\n=== 使用 RAG（模型看著論文摘錄回答）===')
    answer, hits = rag_answer(question)
    print(answer)
    print('\n來源：')
    for i, hit in enumerate(hits, 1):
        print(f'  [{i}] 第 {hit["page"]} 頁（相似度 {hit["score"]:.3f}）')
