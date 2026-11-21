# LiteLLM 模型設定

本系統所有模型呼叫（GraphRAG 建圖與查詢、問題生成、Vision、LLM Judge）都經由自行架設的 [LiteLLM proxy](https://docs.litellm.ai/docs/simple_proxy)。LiteLLM 提供 OpenAI 相容端點，本系統只需要知道 proxy 的網址與一把 LiteLLM Key；各家供應商的 API Key 只存在 LiteLLM 的設定中。

## 1. 準備 `config.yaml`

在專案以外的目錄（例如 `~/litellm/`）建立 `config.yaml`。`model_name` 就是本系統選單顯示的模型名稱，可以自由命名：

```yaml
model_list:
  - model_name: gemini-flash
    litellm_params:
      model: gemini/gemini-2.5-flash
      api_key: os.environ/GEMINI_API_KEY
  # 建圖與 Local／Global／DRIFT／Basic 查詢都需要 Embedding 模型
  - model_name: text-embedding-3-small
    litellm_params:
      model: openai/text-embedding-3-small
      api_key: os.environ/OPENAI_API_KEY
    model_info:
      mode: embedding
  - model_name: claude-judge
    litellm_params:
      model: anthropic/claude-sonnet-5-5
      api_key: os.environ/ANTHROPIC_API_KEY

general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY

litellm_settings:
  # 模型不支援某個參數時自動略過，而不是整個請求失敗
  drop_params: true
```

供應商的模型 ID 依 LiteLLM 文件填寫；上面的模型名稱只是範例，請換成你帳號可用的模型。

**Embedding 模型必須與建立索引時相同**。既有索引的向量由該模型產生，換模型後向量維度與語意空間都不同，必須重新建圖。`model_info.mode: embedding` 讓系統把它歸入 Embedding 選單；未設定時，名稱含 `embed` 的模型也會歸入。

## 2. 用 Docker 啟動 proxy

把 Key 放在 `~/litellm/.env`（不要放進本專案，也不要提交到 Git）：

```bash
GEMINI_API_KEY=...
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
LITELLM_MASTER_KEY=sk-litellm-自訂一組長字串
```

啟動：

```bash
docker run -d --name litellm --env-file ~/litellm/.env \
  -v ~/litellm/config.yaml:/app/config.yaml \
  -p 4000:4000 \
  ghcr.io/berriai/litellm:main-stable --config /app/config.yaml
```

確認可用：

```bash
python3 -c "import urllib.request as u; r=u.Request('http://localhost:4000/v1/models', headers={'Authorization':'Bearer sk-litellm-...'}); print(u.urlopen(r).read().decode())"
```

## 3. 在管理後台設定

1. 開啟「0 專案與連線設定」的「LiteLLM 連線設定」區塊。
2. 填入 **LiteLLM Base URL** 與 **LiteLLM Key**（master key 或 LiteLLM 發的 virtual key）。網址可含或不含 `/v1`，系統會統一補上。
   - 若 LiteLLM 網址受 Cloudflare Access 保護，在**執行本系統**的專案根目錄 `.env` 加入以下兩項（或在管理後台程序環境設定）：
     ```dotenv
     CF_ACCESS_CLIENT_ID=your-cloudflare-access-client-id
     CF_ACCESS_CLIENT_SECRET=your-cloudflare-access-client-secret
     ```
     兩者會作為 `CF-Access-Client-Id`、`CF-Access-Client-Secret` 標頭送出，不會保存到 `projects/.connection.json`。
   - 管理後台在本機執行：`http://localhost:4000`。
   - 管理後台在 Docker 執行、proxy 在主機：`http://host.docker.internal:4000`（Linux 需在 `docker run` 加 `--add-host=host.docker.internal:host-gateway`）。
   - 兩個容器在同一個 Docker network：`http://litellm:4000`。
3. 按「測試連線」。系統呼叫 proxy 的 `/v1/models` 列出模型，並從 `/model/info` 判斷支援推理深度（`supports_reasoning`）與 Embedding（`mode`）的模型。
4. 勾選可用模型，選擇預設 Chat 模型與 Embedding 模型，按「儲存連線設定」。各頁的模型選單會立即更新，只列出勾選的模型名稱。

## 行為說明

- **保存位置**：`projects/.connection.json`（權限 600）。`projects/` 已列在 `.gitignore`。
- **舊設定移轉**：原本的 `.connection.json` 與 `.judge_connection.json` 會在讀取時自動合併，Base URL 與 Key 取自舊的評測模型設定；第一次儲存後只使用 `.connection.json`，舊檔不再讀取，可自行刪除。
- **舊結果相容**：先前保存的 `litellm:<別名>` 模型 ID 讀取時會去掉前綴，顯示為模型名稱。
- **Key 讀取順序**：已儲存的 LiteLLM Key → 環境變數 `LITELLM_API_KEY` → 環境變數 `GRAPHRAG_API_KEY` → `.env`。儲存時 Key 欄留空會沿用已儲存的 Key。
- **Cloudflare Access**：可選環境變數 `CF_ACCESS_CLIENT_ID`、`CF_ACCESS_CLIENT_SECRET`，支援本機專案根目錄 `.env` 與程序環境（程序環境優先）。所有直接請求與 GraphRAG 請求都會帶入標頭；未設定時不送出。GraphRAG 的 `settings.yaml` 只寫入 `${CF_ACCESS_CLIENT_ID}` 這類變數參照，不保存憑證值。管理後台狀態會顯示是否偵測到 Cloudflare Access 設定。
- **推理深度**：只有測試連線時回報 `supports_reasoning` 的模型才顯示推理深度選單並送出 `reasoning_effort`；其他模型不送此參數，改送 `temperature: 0` 與 `max_tokens`。GraphRAG 內建的 LiteLLM 用戶端會丟棄它不認得的參數，因此設定檔同時寫入 `allowed_openai_params`。
- **JSON 輸出**：一律先送 `response_format: json_object`。評測若收到 HTTP 400（模型不支援 JSON mode），會不帶此參數重送一次；模型回覆純文字時，從內容中擷取第一個 JSON 物件再驗證。
- 在 LiteLLM 改了 `config.yaml` 的模型後，回到第 0 頁重新「測試連線」並儲存，模型清單與推理深度支援資訊才會更新。
