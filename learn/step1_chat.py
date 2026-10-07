"""步驟 1：跟 LLM 對話。

LLM（大型語言模型）本質上是一個 HTTP 服務：
送出「訊息列表」，拿回「模型的回覆」。這裡用本機的 Ollama。

執行：python learn/step1_chat.py
"""
import os

import httpx

OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://127.0.0.1:11434')
LLM_MODEL = os.getenv('LLM_MODEL', 'qwen3:4b-instruct')


def chat(messages, temperature=0.2):
    """呼叫 Ollama 的 /api/chat，回傳模型的文字回覆。"""
    response = httpx.post(f'{OLLAMA_URL}/api/chat', timeout=300, json={
        'model': LLM_MODEL,
        'messages': messages,
        'stream': False,                          # 一次拿到完整回覆（較簡單）
        'options': {'temperature': temperature},  # 越低越穩定，越高越有創意
    })
    response.raise_for_status()
    return response.json()['message']['content']


if __name__ == '__main__':
    # 訊息有三種角色：
    #   system    — 給模型的「角色設定與規則」
    #   user      — 使用者說的話
    #   assistant — 模型之前說過的話（用來延續多輪對話）
    messages = [
        {'role': 'system', 'content': '你是簡潔的助教，用繁體中文、兩三句話回答。'},
        {'role': 'user', 'content': '什麼是 RAG？'},
    ]
    answer = chat(messages)
    print('第一輪回答：\n', answer)

    # LLM 本身「沒有記憶」。要多輪對話，就把之前的內容一起送回去。
    messages.append({'role': 'assistant', 'content': answer})
    messages.append({'role': 'user', 'content': '用一個生活中的比喻再說一次。'})
    print('\n第二輪回答：\n', chat(messages))
