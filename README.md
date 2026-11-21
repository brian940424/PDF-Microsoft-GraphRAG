# 汽車維修手冊 GraphRAG 平台

本專案將 PDF 文件轉換為 Microsoft GraphRAG 索引，提供來源可追溯問答、自動問答測試及多策略檢索實驗。管理後台以目前開啟專案為操作範圍，題目集、設定與執行結果會保存於該專案。

系統提供兩個獨立入口：

- `automotive-graphrag`：一般使用者入口，選擇已啟用專案進行問答及檢視 Evidence。
- `automotive-graphrag-admin`：管理後台，管理專案、API、PDF、建圖、問答測試與檢索實驗。

新版操作說明：

- [功能總覽（管理後台分頁 0 至 5）](docs/01_功能總覽.md)
- [安裝與啟動](docs/02_安裝與啟動.md)
- [操作指南](docs/04_使用指南.md)
- [檢索實驗策略與參數](docs/08_檢索實驗與策略參數.md)
- [回答與圖譜快取機制（含語意快取觸發方式與設定）](docs/14_回答與圖譜快取機制說明.md)
- [語意快取改善計畫（目前暫不實作）](docs/15_快取機制之語意快取改善計畫.md)
- [題目集 JSON 格式](docs/05_題目集格式.md)
- [開發進度指揮中心（3D 進度儀表板）](progress-dashboard/README.md)

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

- 在「0 專案與連線設定」建立、刪除、選取並開啟專案；管理頁面共用目前開啟的專案。
- 在「0 專案與連線設定」共用設定 LiteLLM proxy 的 Base URL、Key 與模型；所有模型呼叫都經由此 proxy。
- 在「1 文件與建圖」匯入 PDF、設定各文件前處理頁面範圍與忽略比例（可選用 Vision 模型轉寫表格、圖片與掃描頁），執行建圖並即時查看日誌。
- 在「2 問答測試」使用四種 GraphRAG 策略（或由 LLM 自動選擇策略並說明原因）單次問答並檢視可取得的來源 Evidence。
- 對可可靠解析的檢索來源計算 Evidence Hit Rate、Evidence Precision、Document Hit Rate；無法可靠比對時明確標示不可計算。
- 在「3 自動問答測試」分開生成題目、回答及評測，並可人工修正結果。
- 以 LLM Judge 比較系統回答和標準答案，並提供評判理由。
- 在「4 檢索實驗」比較多個檢索實驗組，保存設定及結果，匯出精簡摘要和逐題 JSON。
- 在「5 3D 儀表板」以明亮等角積木風格的 3D 地圖瀏覽知識圖譜：社群為分區、實體為建築、最近問答為道路上的卡車，並顯示 KPI、建圖流程與社群／類型／問答清單。

目前允許的模型：

| 用途 | 模型 |
|---|---|
| Chat | LiteLLM proxy 提供並在第 0 頁勾選的模型（預設 `gpt-6-luna`） |
| Embedding | `text-embedding-3-small`、`text-embedding-3-large` |

Chat 模型預設為 `gpt-6-luna`，Embedding 模型預設為 `text-embedding-3-small`。

## 實作原理

資料流程：

```text
PDF
  → PyMuPDF 逐頁擷取文字（含圖表或無文字層的頁面可改以 Vision 模型轉寫）
  → 排除指定比例的頁首／頁尾與左右兩側
  → processed/input.jsonl
  → Microsoft GraphRAG index
  → GraphRAG local/global/drift/basic query
  → 回答 + Query Context
  → Evidence 對應 PDF、頁碼與 Chunk ID
```

題目與評估流程：

```text
處理後原文
  →「3 自動問答測試」生成／編輯題目與來源
  → 生成回答
  → 獨立執行答案評測，並可人工修正結果
  →「4 檢索實驗」跨實驗組比較答案品質與可計算的來源指標
  → LLM Judge 評測答案，並按來源頁面計算可用的檢索指標
  → 實驗組摘要與逐題 JSON 結果
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

1. 啟動管理後台並設定 API 連線。
2. 在「0 專案與連線設定」建立、選取並開啟專案。
3. 在「1 文件與建圖」匯入 PDF、設定逐份文件的前處理範圍、執行前處理及建圖。
4. 在「2 問答測試」進行單次問答；或在「3 自動問答測試」、「4 檢索實驗」載入題目集進行自動問答與策略比較。
5. 將已建圖且啟用查詢的專案提供給一般使用者入口。

完整操作方式及題目來源格式請見[新版使用指南](docs/04_使用指南.md)與[題目集 JSON 格式](docs/05_題目集格式.md)。

## Docker：使用 Dockerfile 建立執行環境

需求：

- Docker Engine 24 或更新版本
- 可連線至 LiteLLM proxy（OpenAI 相容 API），設定方式見 [docs/03_LiteLLM模型設定.md](docs/03_LiteLLM模型設定.md)
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
GRAPHRAG_API_KEY=your-litellm-key
GRAPHRAG_API_BASE=http://localhost:4000/v1
GRAPHRAG_CHAT_MODEL=gpt-6-luna
GRAPHRAG_EMBEDDING_MODEL=text-embedding-3-small
# Optional: set these only when the LiteLLM proxy is protected by Cloudflare Access
CF_ACCESS_CLIENT_ID=your-cloudflare-access-client-id
CF_ACCESS_CLIENT_SECRET=your-cloudflare-access-client-secret
```

`CF_ACCESS_CLIENT_ID` and `CF_ACCESS_CLIENT_SECRET` are credentials for the **LiteLLM proxy accessed by this app**. They are sent as `CF-Access-Client-Id` and `CF-Access-Client-Secret` headers for every LiteLLM request, including GraphRAG indexing and queries. Do not put them in the LiteLLM proxy's own `.env`; configure them in the environment passed to the GraphRAG admin container/process. They are not needed when the proxy is not behind Cloudflare Access.

以 `--env-file` 傳入時，上述設定都會成為環境變數。非 Docker 的本機 `.env` 只提供 API Key 與 Cloudflare Access 憑證，Base URL 與模型請改用環境變數或管理後台設定。

若選擇由管理後台儲存連線設定，設定會寫入持久化 Volume 內的 `.connection.json`，並優先於環境變數與 `.env`；儲存後再改這些值不會生效。請將該 Volume 視為敏感資料並限制存取。

### 查詢與語意快取環境變數

下列變數由管理後台及一般使用者入口各自的**服務程序環境**讀取。Docker 可把它們放入 `--env-file` 使用的 `.env`；本機直接執行時，專案的 `.env` 只會自動讀取 API Key 與 Cloudflare Access 憑證，快取變數需先在啟動 shell 設定／export，再啟動服務。變更後需重啟對應服務。

| 變數 | 預設值 | 用途 |
|---|---:|---|
| `GRAPHRAG_ENABLE_QUERY_CACHE` | `true` | 磁碟回答快取總開關。設 `false`／`0`／`no`／`off` 時 exact 與 semantic 結果快取都不會執行。 |
| `GRAPHRAG_QUERY_CACHE_TTL_SECONDS` | `2592000`（30 天） | 每專案 SQLite exact／semantic entries 的存活秒數。 |
| `GRAPHRAG_QUERY_CACHE_MAX_BYTES` | `524288000`（500 MiB） | 每專案 exact 與 semantic entries 共用的容量上限；超限時淘汰較舊／較少使用的項目。 |
| `GRAPHRAG_ENABLE_INDEX_CACHE` | `true` | 程序內 GraphRAG 索引快取開關；與回答快取分開控制。 |
| `GRAPHRAG_MAX_LOADED_PROJECTS` | 管理後台 `1`；一般入口 `2` | 每個服務程序可常駐 RAM 的索引專案數。 |
| `GRAPHRAG_ENABLE_SEMANTIC_QUERY_CACHE` | `false` | 語意快取全域 master switch。只有設為 `true`／`1`／`yes`／`on` 才允許 Shadow／Serving；打開它**不會**自動啟用所有專案。 |
| `GRAPHRAG_SEMANTIC_QUERY_CACHE_THRESHOLD` | 無 | Serving 必填，校準後的 cosine 相似度門檻，範圍 `0 < threshold <= 1`。未設定或無效時 Serving 停用；不要使用未校準的猜測值。 |
| `GRAPHRAG_SEMANTIC_QUERY_CACHE_MIN_MARGIN` | 無 | Serving 必填，最佳候選與次佳候選的校準分數差，範圍 `0 <= margin <= 1`。未設定或無效時 Serving 停用。 |
| `GRAPHRAG_SEMANTIC_QUERY_CACHE_TOP_K` | `5` | 語意候選數，允許 `1`～`20`。無效值會停用該次語意查找。 |

語意是否實際觸發由**全域開關、專案模式及單次查詢條件共同決定**：全域開關必須開啟、`GRAPHRAG_ENABLE_QUERY_CACHE` 必須啟用、專案在第 0 頁設為 `shadow` 或 `serving`，且此問題 exact miss、沒有 bypass。專案模式寫在專案根目錄 `query_cache_settings.json`，新專案預設 `off`；因此其他仍為 `off` 的專案不會因全域 master switch 打開而呼叫 Embedding。若某專案已是 Shadow／Serving，打開全域開關後它才會開始依該模式運作。管理後台與 portal 是不同程序；要讓兩個入口都使用相同模式，必須讓兩個程序取得相同環境變數。

快取觸發順序：

1. Exact cache 先查；命中就直接回傳，不呼叫 Embedding。這是預期行為，測 Shadow 時應用未曾精確查詢過的改寫問題。
2. `off`、全域開關關閉、回答快取總開關關閉、bypass，或不支援的查詢種類，都不呼叫 Embedding。自動問答／檢索實驗預設 bypass 時亦略過 semantic lookup。
3. `shadow` 在 exact miss 時呼叫一次 Embedding，記錄候選判斷後仍執行 fresh GraphRAG 查詢；有無 threshold 不改變 Shadow 不回傳候選的行為。每筆 miss 會增加 Embedding API 延遲與可能費用。
4. `serving` 除以上條件外，threshold 和 margin 必須都有效；候選還需通過 scope、hard guards、相似度與 margin，才會重用舊結果。未校準前不要設 serving 或填入臆測門檻。

範例：Docker `--env-file` 使用的設定應保持語意總開關關閉：

```dotenv
GRAPHRAG_ENABLE_QUERY_CACHE=true
GRAPHRAG_QUERY_CACHE_TTL_SECONDS=2592000
GRAPHRAG_QUERY_CACHE_MAX_BYTES=524288000
GRAPHRAG_ENABLE_INDEX_CACHE=true
GRAPHRAG_ENABLE_SEMANTIC_QUERY_CACHE=false
GRAPHRAG_SEMANTIC_QUERY_CACHE_TOP_K=5
```

若要在本機測 Shadow，先在啟動兩個服務的 shell 設定全域開關，例如 `export GRAPHRAG_ENABLE_SEMANTIC_QUERY_CACHE=true`；再於後台只把指定測試專案切成 `shadow`，其他專案維持 `off`，然後重啟要觀察的服務。Serving 的 threshold／margin 必須先完成獨立 holdout 校準，並依專案改善計畫驗收，不能把本文件中的預設值當成門檻建議。本機 Shadow 測試流程與紀錄欄位見[快取機制說明](docs/14_回答與圖譜快取機制說明.md)及[語意快取改善計畫](docs/15_快取機制之語意快取改善計畫.md)。

Linux／macOS 本機啟動範例（管理後台和 portal 是不同程序，因此在兩個終端機都設定開關）：

```bash
# Terminal 1: admin
export GRAPHRAG_ENABLE_SEMANTIC_QUERY_CACHE=true
export PROJECTS_ROOT="$PWD/projects"
GRADIO_SERVER_PORT=7861 automotive-graphrag-admin

# Terminal 2: portal
export GRAPHRAG_ENABLE_SEMANTIC_QUERY_CACHE=true
export PROJECTS_ROOT="/absolute/path/to/the/project/projects"
automotive-graphrag
```

Windows PowerShell 在啟動每個服務的視窗執行 `$env:GRAPHRAG_ENABLE_SEMANTIC_QUERY_CACHE="true"`，並設定相同的 `$env:PROJECTS_ROOT`；管理後台使用連接埠 7861，portal 使用 7860。重啟服務才會讀取新的 process environment。全域開關開啟後，只有後台設定成 Shadow／Serving 的專案會嘗試語意查找；其他專案維持 `off`。

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

- LiteLLM Base URL 與 LiteLLM Key 是否正確。
- 模型是否在允許清單內。
- PDF 是否已完成前處理並產生 `processed/input.jsonl`。
- `graphrag/indexing.log` 與 `graphrag/last_index_run.json` 的錯誤訊息。
- API 額度、網路及磁碟空間是否足夠。

建圖失敗時，系統會嘗試保留上一次成功的索引。

### 題目生成為英文

目前 Prompt 與後端驗證要求問題及參考答案包含繁體中文；純英文結果不會保存。技術名詞、DTC 及單位仍可能保留英文。更新程式後請重新啟動服務。

### PDF 內文混入頁首頁尾或頁緣索引籤

在「前處理設定與逐頁檢查」的表格調高「忽略 Header (%)」或「忽略 Footer (%)」；頁緣的章節索引籤等側邊文字則用「忽略左側 (%)」「忽略右側 (%)」。上下、左右各自合計必須小於 100%。下方逐頁檢查會以紅色標出忽略範圍並列出被裁掉的文字，確認後重新前處理；修改 PDF 或前處理結果後需重新建圖。

### 如何降低開發測試成本

- Chat 預設使用 `gpt-6-luna`；也可在連線設定中勾選 proxy 提供的其他模型。
- Embedding 使用 `text-embedding-3-small`。
- 題目生成先縮小頁碼、章節及取樣數量。
- 回答與評測分開執行：「3 自動問答測試」的「③ 評測回答」與「4 檢索實驗」的評測都只使用既有回答，調整評審設定時不必重新產生回答。
- 系統的題目生成與 Judge 均以每批一次 API 呼叫為設計目標。

## 非 Docker 啟動方式

需求：Python 3.11 以上，建議 Python 3.12。

建立虛擬環境並安裝：

Linux／macOS：

```bash
python3.12 --version   # 需 3.11 以上；沒有 3.12 時可改用 python3.11
python3.12 -m venv .venv
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
export GRAPHRAG_API_KEY="your-litellm-key"
export GRAPHRAG_API_BASE="http://localhost:4000/v1"
export GRAPHRAG_CHAT_MODEL="gpt-6-luna"
export GRAPHRAG_EMBEDDING_MODEL="text-embedding-3-small"
# Optional, only for a LiteLLM proxy protected by Cloudflare Access
export CF_ACCESS_CLIENT_ID="your-cloudflare-access-client-id"
export CF_ACCESS_CLIENT_SECRET="your-cloudflare-access-client-secret"
```

Windows PowerShell（單行）：

```powershell
$env:PROJECTS_ROOT=(Join-Path $PWD "projects"); $env:GRAPHRAG_API_KEY="your-litellm-key"; $env:GRAPHRAG_API_BASE="http://localhost:4000/v1"; $env:GRAPHRAG_CHAT_MODEL="gpt-6-luna"; $env:GRAPHRAG_EMBEDDING_MODEL="text-embedding-3-small"; $env:CF_ACCESS_CLIENT_ID="your-cloudflare-access-client-id"; $env:CF_ACCESS_CLIENT_SECRET="your-cloudflare-access-client-secret"
```

啟動一般入口：

```bash
automotive-graphrag
```

Windows PowerShell（單行）：

```powershell
$env:GRADIO_SERVER_PORT="7860"; .\.venv\Scripts\automotive-graphrag.exe
```

啟動管理後台（與一般入口同時執行時需用不同連接埠）：

```bash
GRADIO_SERVER_PORT=7861 automotive-graphrag-admin
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
- 停用 Vision 時只能擷取 PDF 文字層，掃描影像 PDF 無法取得內容；啟用 Vision 會依送出頁數額外計費，轉寫品質仍需以「逐頁檢查」抽查。
- 表格、跨欄版面、圖片及複雜公式的文字順序可能不完整（啟用 Vision 可改善，但不保證正確）。
- GraphRAG 建圖需要 API Token、記憶體、磁碟及時間，請先用少量文件驗證設定。
- 不同專案的資料互相隔離，但共用 API 連線設定及模型設定。
- 不要在建圖、查詢或評測執行期間刪除其專案，也不要讓多個管理程序同時修改同一專案。
- `.connection.json`、原始 PDF、索引、查詢紀錄及匯出報告可能含敏感資料，不應提交至 Git 或放入公開映像。
- 一般使用者入口不包含驗證與授權；正式部署必須在反向代理或平台層加入身分驗證。
- 自動評測不能取代人工真值，低分、Retrieval 失敗與低信心案例應進入人工審查。
- 目前主要驗證 Linux 環境；Windows 建議使用 WSL2 或 Docker。
