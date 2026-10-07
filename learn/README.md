# 從零學 LLM 與 RAG：5 個步驟

這個資料夾用 5 支短程式（每支 30–60 行），一步步拆解 `app/` 網頁版背後的原理。
建議**照順序執行、讀程式、做練習**，每一步只學一個觀念。

```
步驟 1  LLM 對話          問 → 答
步驟 2  文字變向量         「意思」可以用數字比較
步驟 3  PDF 切段          長文件 → 小段落
步驟 4  語意檢索（R）      問題 → 找出最相關的段落
步驟 5  完整 RAG          檢索 + 把段落放進提示 + 讓 LLM 回答
```

## 準備

Ollama 裝在 WSL 裡，所以程式也在 WSL 執行。在 PowerShell 輸入 `wsl -d Ubuntu-24.04` 進入 Linux，然後：

```bash
cd /mnt/d/claude/RAG/outputs/paper-rag
python3 -m venv .venv-linux                       # 只需做一次
.venv-linux/bin/pip install -r requirements.lock.txt
source .venv-linux/bin/activate                   # 之後每次開新終端都要執行
```

> 若已經跑過 `start-wsl.ps1`，`.venv-linux` 已經建好，直接 `source` 即可。

所有指令都在專案根目錄（`paper-rag/`）執行。

---

## 步驟 1：跟 LLM 對話 — [step1_chat.py](step1_chat.py)

```bash
python learn/step1_chat.py
```

**觀念**

- LLM 就是一個 HTTP API：送出 `messages`，拿回文字。
- 訊息有三種角色：`system`（規則）、`user`（使用者）、`assistant`（模型說過的話）。
- **LLM 沒有記憶。** 第二輪之所以「記得」第一輪，是因為程式把整段對話重新送了一次。
- `temperature`：越低越穩定，越高越隨機。

**練習**

1. 把 `system` 改成「你是海盜，用海盜口吻回答」，看語氣怎麼變。
2. 把 `temperature` 改成 `1.5`，同一個問題跑三次，比較答案差異。
3. 刪掉 `messages.append({'role': 'assistant', ...})` 那行，第二輪會發生什麼事？

---

## 步驟 2：文字變向量 — [step2_embed.py](step2_embed.py)

```bash
python learn/step2_embed.py
```

**觀念**

- **Embedding（嵌入）**：把一段文字轉成一串數字（embeddinggemma 是 768 個）。
- 意思相近 → 向量方向相近。用 **cosine similarity** 衡量，越接近 1 越相似。
- 比的是「語意」不是「關鍵字」：英文的天氣預報句子雖然一個中文字都沒有，分數仍然很高。
- embeddinggemma 建議問題加 `task: search result | query:`、文件加 `title: none | text:` 前綴。

**練習**

1. 在 `documents` 加入「明天降雨機率 80%」，猜猜它會排第幾？
2. 拿掉兩種前綴再跑一次，分數有變化嗎？

---

## 步驟 3：PDF 切段 — [step3_chunk.py](step3_chunk.py)

```bash
python learn/step3_chunk.py
```

**觀念**

- 整份論文太長，塞不進模型，所以要切成小段，**只把相關的幾段**交給 LLM。
- 每段記住自己的**頁碼**，回答時才能附上出處。
- **重疊（overlap）**：相鄰段落共用 250 字，避免重要句子剛好被切成兩半。
- **清理很重要**：目錄頁的 `. . . . .` 會被誤認為「跟什麼都有點像」，所以先移除。

**練習**

1. 把 `CHUNK_SIZE` 改成 `300`，段落數變多少？
2. 試著用你自己的 PDF：`python learn/step3_chunk.py 路徑/檔案.pdf`。

---

## 步驟 4：語意檢索 — [step4_search.py](step4_search.py)

```bash
python learn/step4_search.py "這篇論文提出的模型架構是什麼？"
```

第一次會花幾十秒建索引（存成 `learn/my_index.json`），之後就很快。
**改了步驟 3 的切段方式後，請刪掉 `my_index.json` 重建。**

**觀念**

- 索引 = 每一段的「文字 + 向量」。只需建一次。
- 檢索 = 問題轉成向量 → 跟每段算相似度 → 取前 k 名。
- 這就是 RAG 的 **R（Retrieval）**。

**練習（重要！）**

1. 問一個籠統的問題：`"這篇論文使用了哪些資料？"`。你會發現排第一的是**目錄**和**致謝**，這兩段剛好「什麼都提一點」。
2. 換成具體、帶有論文專有名詞的問法（例如「用哪些資料集訓練 ○○ 模型？」），結果好很多。
   → 學到：**問題越具體，檢索越準。** 這也是 RAG 最常見的失敗原因。
3. 想想看：如果要讓籠統問題也能找到好結果，可以怎麼做？（提示：過濾掉目錄頁、讓 LLM 先改寫問題）

---

## 步驟 5：完整 RAG — [step5_rag.py](step5_rag.py)

```bash
python learn/step5_rag.py "這篇論文的研究目的與使用的資料是什麼？"
```

程式會先讓模型**不看論文**回答，再用 RAG 回答。作者用自己的論文實測的結果（內容因保密省略）：

| | 回答 |
|---|---|
| 沒有 RAG | 憑一般知識猜測，把論文研究的**目標變數猜錯** ❌ |
| 有 RAG | 正確說出使用的資料與目標變數，並以 `[1]` 引用摘要頁 ✅ |

**觀念**

- **R**etrieval：步驟 4 找段落。
- **A**ugmented：把段落貼進提示，並在 system prompt 規定「只根據摘錄回答、要標來源、不知道就說不知道」。
- **G**eneration：步驟 1 的 `chat()` 產生答案。
- RAG **沒有訓練模型**，只是在提問時「把參考資料一起給它看」。所以換一份 PDF，重建索引就能用。

**練習**

1. 問論文裡沒有的事（「作者喜歡吃什麼？」），模型有沒有乖乖說不知道？
2. 把 `top_k` 改成 `1` 和 `8`，回答品質怎麼變？（太少：資訊不足；太多：雜訊變多、變慢）
3. 刪掉 system prompt 裡「不要編造」那句，再問一次第 1 題。

---

## 下一步：對照網頁版

讀懂 5 個步驟後，打開 [../app/rag.py](../app/rag.py)，你會發現它是同樣的流程，只多了實務需要的部分：

| 教學版 | 網頁版 `app/rag.py` | 多做了什麼 |
|---|---|---|
| `pdf_to_chunks()` | `extract_chunks()` | 檢查加密、空白頁、檔案大小上限 |
| `embed()` + `cosine()` | `embed()` + `normalize()` + `dot()` | 先把向量長度縮成 1，內積就等於 cosine，計算更快 |
| `build_index()` | `build()` | 用 SHA-256 判斷 PDF 是否換過；全部成功才替換舊索引 |
| `search()` | `retrieve()` | 檢查向量維度是否一致 |
| `rag_answer()` | `build_messages()` + `generate()` + `ask()` | 防提示注入、移除 `<think>`、檢查引用編號是否有效 |

再往後可以學的主題：

- **向量資料庫**（Chroma、Qdrant）：段落多到幾十萬筆時才需要。
- **混合檢索**：向量檢索搭配關鍵字檢索（BM25），專有名詞會找得更準。
- **Reranker**：先粗撈 20 段，再用另一個模型精挑 5 段。
- **多輪對話 RAG**：讓 LLM 把「它呢？」這類問題改寫成完整問題後再檢索。
