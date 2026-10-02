# 汽車維修手冊 GraphRAG 平台

本專案將汽車維修 PDF 手冊轉換為 Microsoft GraphRAG 索引，提供可追溯原始 PDF、頁碼與 Chunk 的維修問答，以及題目集、人工審查、Retrieval 評估、自動出題與 LLM Judge 等功能。

系統提供兩個獨立入口：

- `automotive-graphrag`：一般使用者入口，只顯示車型選擇、維修問答、Evidence 與案例紀錄。
- `automotive-graphrag-admin`：管理後台，負責專案、API 連線、PDF、建圖、測試及評估。

## 目錄

- [專案介紹](#專案介紹)
- [實作原理](#實作原理)
- [基本使用流程](#基本使用流程)
- [Docker：使用 Dockerfile 建立執行環境](#docker使用-dockerfile-建立執行環境)
- [部署後如何啟動](#部署後如何啟動)
- [常見問題](#常見問題)
- [非 Docker 啟動方式](#非-docker-啟動方式)
- [測試](#測試)
- [限制與注意事項](#限制與注意事項)

## 專案介紹

主要功能如下：

- 建立、刪除、啟用或停用多個車型／手冊專案。
- 共用同一組 GraphRAG API Base URL、API Key 與模型設定。
- 匯入或移除 PDF，並設定要忽略的頁首／頁尾高度百分比。
- 將 PDF 逐頁轉成帶有 `project_id`、PDF、頁碼、章節及 Chunk ID 的 JSONL。
- 初始化及執行 Microsoft GraphRAG 建圖。
- 執行單題或題目集批次問答，顯示可回連原始手冊的 Evidence。
- 建立 Gold Evidence，計算 Recall@K、MRR 等 Retrieval 指標。
- 從處理後原文取樣，生成繁體中文候選題目及參考答案。
- 使用 LLM Judge 評估答案正確性、完整性及證據支持程度。
- 保存人工審查、案例紀錄及 JSON／CSV 報告。

目前允許的模型：

| 用途 | 模型 |
|---|---|
| Chat | `gpt-4o-mini`、`gpt-4.1-mini` |
| Embedding | `text-embedding-3-small`、`text-embedding-3-large` |

開發及測試建議使用 `gpt-4o-mini` 與 `text-embedding-3-small` 以降低成本。

## 實作原理

資料流程：

```text
PDF
  → PyPDF 逐頁擷取文字
  → 排除指定比例的頁首／頁尾
  → processed/input.jsonl
  → Microsoft GraphRAG index
  → GraphRAG local/global/drift/basic query
  → 回答 + Query Context
  → Evidence 對應 PDF、頁碼與 Chunk ID
```

題目與評估流程：

```text
處理後原文
  → 依章節、頁碼、內容類型取樣
  → 生成繁體中文問題、參考答案與 Gold Evidence
  → 人工核准
  → 批次回答
  → Retrieval 評估 + LLM Judge + 人工抽查
  → JSON／CSV 報告
```

每個專案都有隔離的工作目錄：

```text
projects/<project_id>/
├── project.json
├── source/          # 原始 PDF
├── processed/       # PDF 前處理結果
├── graphrag/        # GraphRAG 設定、輸入、索引與日誌
├── question_sets/   # 題目集
├── runs/            # 查詢、取樣、評測與案例紀錄
└── exports/         # JSON／CSV 匯出
```

API 連線設定保存在 `PROJECTS_ROOT/.connection.json`，由該根目錄下所有專案共用。專案選擇只使用已登錄的 `project_id`，不接受任意檔案路徑。

## 基本使用流程

1. 啟動管理後台 `automotive-graphrag-admin`。
2. 在「連線設定」輸入 API Base URL、API Key，選擇 Chat 與 Embedding 模型，再測試連線。
3. 在「專案設定」建立專案，例如 `L33-SM3E`。
4. 在「文件與建圖」匯入 PDF。
5. 視手冊版面設定忽略頁首／頁尾百分比，執行前處理。
6. 確認文件及頁數後建立 Graph。建圖會使用 API Token，且可能需要較長時間。
7. 建圖成功後，將專案保持為「啟用一般使用者查詢」。
8. 啟動一般入口 `automotive-graphrag`，選擇車型／專案並輸入問題。
9. 檢查回答下方的 PDF、頁碼、Chunk 及 Evidence 全文；必要時加入案例紀錄。

題目生成需先完成 PDF 前處理。建議先掃描及預覽原文，再建立取樣批次，最後呼叫 API 生成候選題。生成內容固定為繁體中文，但零件名稱、DTC、單位及原廠術語可保留英文。

## Docker：使用 Dockerfile 建立執行環境

需求：

- Docker Engine 24 或更新版本
- 可連線至設定的 OpenAI 相容 API
- 足夠的磁碟空間保存 PDF、Parquet 與 GraphRAG 索引

建立映像：

Linux／macOS：

```bash
DOCKER_BUILDKIT=1 docker build --progress=plain -t automotive-graphrag:local .
```

Windows PowerShell（單行）：

```powershell
$env:DOCKER_BUILDKIT="1"; docker build --progress=plain -t automotive-graphrag:local .
```

Dockerfile 會直接從 Astral 官方容器映像複製 `uv` 與 `uvx` 二進位檔，不依賴 PyPI 鏡像是否收錄 `uv`；接著以 `uv pip install --system` 並行下載及安裝 `requirements.docker.txt` 中已驗證的固定版本，最後才複製應用程式碼。如此可避免 pip 在 GraphRAG 與 Pandas 版本間大量回溯；後續只修改原始碼或 README 時，也會直接沿用依賴 layer 與 `/root/.cache/uv` BuildKit 快取。

預設使用清華 TUNA PyPI 鏡像。若所在網路使用阿里雲較快，可在建置時覆寫：

Linux／macOS：

```bash
DOCKER_BUILDKIT=1 docker build --build-arg PYPI_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ --progress=plain -t automotive-graphrag:local .
```

Windows PowerShell（單行）：

```powershell
$env:DOCKER_BUILDKIT="1"; docker build --build-arg PYPI_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ --progress=plain -t automotive-graphrag:local .
```

也可將 `PYPI_INDEX_URL` 改回 `https://pypi.org/simple`。請只使用信任且支援 HTTPS 的鏡像站。

第一次建置仍需下載 GraphRAG、PyArrow 等大型套件，實際時間取決於鏡像站、網路與 CPU；第二次之後應明顯加快。請勿在一般重建時使用 `--no-cache` 或 `--no-cache-filter`。若只想重新下載基礎映像，可使用：

```bash
DOCKER_BUILDKIT=1 docker build --pull --progress=plain -t automotive-graphrag:local .
```

Windows PowerShell（單行）：

```powershell
$env:DOCKER_BUILDKIT="1"; docker build --pull --progress=plain -t automotive-graphrag:local .
```

若建置看似停滯，`--progress=plain` 會顯示目前正在下載或安裝的套件，方便判斷是網路速度還是套件解析問題。

建立持久化 Volume：

```bash
docker volume create automotive-graphrag-data
```

Windows PowerShell 使用相同的單行指令：

```powershell
docker volume create automotive-graphrag-data
```

不要將 API Key 寫入 Dockerfile 或映像。可建立不提交至版本控制的 `.env`：

```dotenv
GRAPHRAG_API_KEY=your-api-key
GRAPHRAG_API_BASE=https://api.openai.com/v1
GRAPHRAG_CHAT_MODEL=gpt-4o-mini
GRAPHRAG_EMBEDDING_MODEL=text-embedding-3-small
```

若選擇由管理後台儲存連線設定，設定會寫入持久化 Volume 內的 `.connection.json`。請將該 Volume 視為敏感資料並限制存取。

## 部署後如何啟動

先啟動管理後台：

Linux／macOS：

```bash
docker run --rm \
  --name automotive-graphrag-admin \
  --env-file .env \
  -p 7861:7860 \
  -v automotive-graphrag-data:/app/projects \
  automotive-graphrag:local \
  automotive-graphrag-admin
```

Windows PowerShell（單行）：

```powershell
docker run --rm --name automotive-graphrag-admin --env-file .env -p 7861:7860 -v automotive-graphrag-data:/app/projects automotive-graphrag:local automotive-graphrag-admin
```

瀏覽 `http://localhost:7861` 完成專案、PDF 與建圖設定。

再啟動一般使用者入口：

Linux／macOS：

```bash
docker run --rm \
  --name automotive-graphrag-portal \
  --env-file .env \
  -p 7860:7860 \
  -v automotive-graphrag-data:/app/projects \
  automotive-graphrag:local
```

Windows PowerShell（單行）：

```powershell
docker run --rm --name automotive-graphrag-portal --env-file .env -p 7860:7860 -v automotive-graphrag-data:/app/projects automotive-graphrag:local
```

瀏覽 `http://localhost:7860` 使用維修問答。

兩個容器必須掛載同一個 `/app/projects` Volume。正式部署時應由反向代理提供 TLS、身分驗證及存取控制，不應直接公開包含原廠手冊的 Gradio 服務。

若需要保存容器，可移除 `--rm`，並搭配 `--restart unless-stopped`。更新版本時重新建立映像及容器，但不要刪除資料 Volume。

## 常見問題

### 下拉選單顯示找不到專案或 `is not in the list of choices`

請確認已完全停止舊版 Gradio 程序並重新啟動。新版會容許瀏覽器短暫送出舊選擇，再由後端安全清除。若同時執行管理與一般入口，兩者必須使用相同的 `PROJECTS_ROOT` 或 Docker Volume。

### GraphRAG Logs 已出現回答，但網頁沒有顯示

新版會先回傳查詢中狀態，完成後只將回答、Evidence 與精簡 Context 摘要送至前端。即使 Evidence 解析失敗，已產生的答案仍會回傳。請確認瀏覽器連線未被反向代理提前逾時，並檢查 `projects/<project_id>/runs/queries.jsonl`。

### 一般入口沒有任何可選專案

只有狀態為 `INDEXED` 且已啟用一般使用者查詢的專案會顯示。請至管理後台確認建圖狀態與啟用選項。

### 建圖失敗

檢查以下項目：

- API Base URL 與 API Key 是否正確。
- 模型是否在允許清單內。
- PDF 是否已完成前處理並產生 `processed/input.jsonl`。
- `graphrag/indexing.log` 與 `graphrag/last_index_run.json` 的錯誤訊息。
- API 額度、網路及磁碟空間是否足夠。

建圖失敗時，系統會嘗試保留上一次成功的索引。

### 題目生成為英文

目前 Prompt 與後端驗證要求問題及參考答案包含繁體中文；純英文結果不會保存。技術名詞、DTC 及單位仍可能保留英文。更新程式後請重新啟動服務。

### PDF 內文混入頁首頁尾

重新執行前處理，增加「忽略頁首高度 (%)」或「忽略頁尾高度 (%)」。兩者合計必須小於 100%。修改 PDF 或前處理結果後需重新建圖。

### 如何降低開發測試成本

- Chat 使用 `gpt-4o-mini`。
- Embedding 使用 `text-embedding-3-small`。
- 題目生成先縮小頁碼、章節及取樣數量。
- 自動評測使用既有回答，不要勾選重新執行系統回答。
- 系統的題目生成與 Judge 均以每批一次 API 呼叫為設計目標。

## 非 Docker 啟動方式

需求：Python 3.11 以上，建議 Python 3.12。

建立虛擬環境並安裝：

Linux／macOS：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

Windows PowerShell（單行）：

```powershell
py -3.12 -m venv .venv; .\.venv\Scripts\python.exe -m pip install --upgrade pip; .\.venv\Scripts\python.exe -m pip install -e ".[test]"
```

設定專案資料位置及 API：

Linux／macOS：

```bash
export PROJECTS_ROOT="$PWD/projects"
export GRAPHRAG_API_KEY="your-api-key"
export GRAPHRAG_API_BASE="https://api.openai.com/v1"
export GRAPHRAG_CHAT_MODEL="gpt-4o-mini"
export GRAPHRAG_EMBEDDING_MODEL="text-embedding-3-small"
```

Windows PowerShell（單行）：

```powershell
$env:PROJECTS_ROOT=(Join-Path $PWD "projects"); $env:GRAPHRAG_API_KEY="your-api-key"; $env:GRAPHRAG_API_BASE="https://api.openai.com/v1"; $env:GRAPHRAG_CHAT_MODEL="gpt-4o-mini"; $env:GRAPHRAG_EMBEDDING_MODEL="text-embedding-3-small"
```

啟動一般入口：

```bash
automotive-graphrag
```

Windows PowerShell（單行）：

```powershell
$env:GRADIO_SERVER_PORT="7860"; .\.venv\Scripts\automotive-graphrag.exe
```

啟動管理後台：

```bash
automotive-graphrag-admin
```

Windows PowerShell（請在另一個視窗執行，單行）：

```powershell
$env:GRADIO_SERVER_PORT="7861"; .\.venv\Scripts\automotive-graphrag-admin.exe
```

Gradio 預設只監聽本機。若需讓同一網路中的其他主機連線：

```bash
export GRADIO_SERVER_NAME=0.0.0.0
export GRADIO_SERVER_PORT=7860
automotive-graphrag
```

Windows PowerShell（單行）：

```powershell
$env:GRADIO_SERVER_NAME="0.0.0.0"; $env:GRADIO_SERVER_PORT="7860"; .\.venv\Scripts\automotive-graphrag.exe
```

管理後台與一般入口同時執行時，請使用不同連接埠，但必須指向同一個絕對 `PROJECTS_ROOT`。

## 測試

安裝測試依賴後執行完整測試：

```bash
.venv/bin/pytest -q
```

執行特定模組：

```bash
.venv/bin/pytest -q tests/test_querying.py
.venv/bin/pytest -q tests/test_question_generation.py
.venv/bin/pytest -q tests/test_automatic_evaluation.py
```

測試使用 Fake Runner／Client，不會呼叫實際模型 API，因此不會產生 Token 費用。

## 限制與注意事項

- LLM 回答、生成題目與 Judge 評分都可能出錯；正式用途必須檢查 Evidence 並進行人工抽查。
- 系統只能從文字型 PDF 擷取內容；掃描影像 PDF 尚未內建 OCR。
- 表格、跨欄版面、圖片及複雜公式的文字順序可能不完整。
- GraphRAG 建圖需要 API Token、記憶體、磁碟及時間，請先用少量文件驗證設定。
- 不同專案的資料互相隔離，但共用 API 連線設定及模型設定。
- 不要在建圖、查詢或評測執行期間刪除其專案，也不要讓多個管理程序同時修改同一專案。
- `.connection.json`、原始 PDF、索引、查詢紀錄及匯出報告可能含敏感資料，不應提交至 Git 或放入公開映像。
- 一般使用者入口不包含驗證與授權；正式部署必須在反向代理或平台層加入身分驗證。
- 自動評測不能取代人工真值，低分、Retrieval 失敗與低信心案例應進入人工審查。
- 目前主要驗證 Linux 環境；Windows 建議使用 WSL2 或 Docker。
