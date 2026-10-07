# Paper RAG 論文閱讀室

FastAPI 網頁 + 本機 Ollama LLM + 向量檢索。PDF 依頁擷取文字，切成帶重疊的段落，使用 embeddinggemma 建立向量，以 cosine similarity 選取相關段落，再交由 qwen3:4b 回答。這是 RAG，並未訓練或微調新模型。

本 repo 不含任何 PDF 或索引檔；請自行放入要閱讀的論文（見下方啟動步驟）。

## 想學原理？

- [learn/README.md](learn/README.md)：5 支短程式一步步拆解 LLM 對話、向量、切段、檢索與完整 RAG。
- [docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md)：完整實作過程、設計取捨、遇到的問題與驗證方法。

## 專案結構

```
app/rag.py          RAG 核心：切段、嵌入、檢索、生成
app/main.py         FastAPI 路由
app/static/         網頁前端（純 HTML/JS/CSS）
learn/              5 步驟教學腳本
tests/test_rag.py   單元測試（不需啟動 Ollama）
docs/               實作說明
```

## 啟動方式 A：Ollama 安裝在 WSL

Ollama 位於 WSL（預設發行版 Ubuntu-24.04）時，Windows 的 localhost 可能無法直接連到它，請在專案目錄執行：

```powershell
powershell -ExecutionPolicy Bypass -File .\start-wsl.ps1
```

此腳本會建立獨立 `.venv-linux` 環境、安裝鎖定套件並印出網頁網址。首次執行需等待套件安裝。網頁綁定 WSL 虛擬網卡，埠號 8001；WSL 重啟後 IP 可能變更，請以腳本顯示的新網址為準。Ollama 的既有 systemd 服務須保持啟動。同一埠已在執行時不需重複啟動；在啟動終端按 Ctrl+C 可停止服務。

首次使用請先 `cp .env.example .env` 並確認模型名稱（例如 `qwen3:4b-instruct` 與 `embeddinggemma`）。

## 啟動方式 B：Ollama 安裝在 Windows

1. 安裝 Python 3.11 或以上，以及 [Ollama](https://ollama.com/download)。啟動 Ollama（桌面版通常會在背景執行；若未啟動，另開終端執行 `ollama serve`）。
2. 下載本機模型，下載容量及 RAM 需求取決於模型：

   ```powershell
   ollama pull qwen3:4b
   ollama pull embeddinggemma
   ```

3. 在此專案目錄執行：

   ```powershell
   Copy-Item .env.example .env
   powershell -ExecutionPolicy Bypass -File .\start.ps1
   ```

4. 開啟 <http://127.0.0.1:8000>，選擇 PDF 並按「上傳並建立索引」。也可放入 `data/pdfs/paper.pdf` 後按「讀取 data/pdfs/paper.pdf」。
5. 索引就緒後提問，展開回答下方摘錄，點選「開啟原文頁面」核對。

已有其他位置的 PDF 時，在 `.env` 指定 `PDF_PATH=C:/完整路徑/data/pdfs/paper.pdf`。**上傳新檔會取代該路徑的 PDF**；若需保留原始檔，請使用預設路徑並透過上傳匯入副本。未指定絕對路徑時，一律相對於本專案根目錄解析。

模型名稱與服務位置可在 `.env` 修改；變更 EMBED_MODEL 或 PDF 後須重新建立索引。索引儲存在 `data/index/index.json`，重啟時會檢查文件 SHA-256 與嵌入模型名稱後才重用。更新同名模型的權重後也應手動重建。

## API

互動文件：<http://127.0.0.1:8000/docs>

| 路徑 | 用途 |
| --- | --- |
| `GET /api/status` | PDF、索引、模型設定與解析空白頁狀態；不代表模型服務在線 |
| `POST /api/index` | 讀取指定 paper.pdf 並重建索引 |
| `POST /api/upload` | multipart `file` 上傳並重建索引 |
| `POST /api/chat` | JSON `{"question":"主要結論是什麼？","top_k":5}` |
| `GET /api/pdf` | 瀏覽目前的原始 PDF |

## 限制與設計

- 適合單一使用者、本機、單篇論文；請使用單一 uvicorn worker。未加入登入機制，預設只綁定 127.0.0.1。若要上線公開使用，需另加身分驗證、HTTPS、代理上傳限制與工作佇列。
- 每題獨立檢索，不會將先前對話自動加入提示。問題應明確指稱主題。
- 支援具有文字層的 PDF。無文字頁會顯示警告；掃描檔需先 OCR。圖表圖像、複雜公式和多欄閱讀順序可能無法準確還原。
- 最多 30 MB、1000 頁、12000 段；小型文件使用記憶體中的向量比較，不需額外向量資料庫。
- 索引建立失敗時保留原本記憶體索引。索引採完整建置後替換，並以 lock 防止同時重建。
- 索引內容包含原文，請與 PDF 一樣保管。預設僅將文件與問題送至本機 Ollama；更改 OLLAMA_URL 會改變資料接收位置。
- 頁碼是 PDF 實體頁序，未必等於論文印刷頁碼。列出的來源是檢索候選，引用編號檢查不能保證模型敘述受原文支持；語意相關度分數也不是答案可信度。
- 跨分頁上傳新文件後，其他分頁先前回答的原文連結將指向目前文件。請重新載入並重新提問。

## 測試

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

測試無需啟動 Ollama，涵蓋 PDF 頁碼、索引持久化與失效、檢索來源、模型失敗、無文字 PDF 與 API 驗證。

## 官方介面參考

- [Ollama embeddings API](https://docs.ollama.com/api/embed)
- [Ollama chat API](https://docs.ollama.com/api/chat)
- [FastAPI 檔案上傳](https://fastapi.tiangolo.com/tutorial/request-files/)
