# LiteLLM 評測模型設定

「3 自動問答測試」的「③ 評測回答」與「4 檢索實驗」可以改用 OpenAI 以外的評測模型（Gemini、Claude 等），做法是經由自行架設的 [LiteLLM proxy](https://docs.litellm.ai/docs/simple_proxy)。LiteLLM 提供 OpenAI 相容端點，本系統只需要知道 proxy 的網址與一把 LiteLLM Key；各家供應商的 API Key 只存在 LiteLLM 的設定中。

## 1. 準備 `config.yaml`

在專案以外的目錄（例如 `~/litellm/`）建立 `config.yaml`。`model_name` 是本系統看到的模型別名，可以自由命名：

```yaml
model_list:
  - model_name: gemini-flash
    litellm_params:
      model: gemini/gemini-2.5-flash
      api_key: os.environ/GEMINI_API_KEY
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

## 2. 用 Docker 啟動 proxy

把 Key 放在 `~/litellm/.env`（不要放進本專案，也不要提交到 Git）：

```bash
GEMINI_API_KEY=...
ANTHROPIC_API_KEY=...
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

1. 開啟「0 專案與連線設定」的「評測模型設定（LiteLLM proxy）」區塊。
2. 填入 **LiteLLM Base URL** 與 **LiteLLM Key**（master key 或 LiteLLM 發的 virtual key）。
   - 管理後台在本機執行：`http://localhost:4000`。
   - 管理後台在 Docker 執行、proxy 在主機：`http://host.docker.internal:4000`（Linux 需在 `docker run` 加 `--add-host=host.docker.internal:host-gateway`）。
   - 兩個容器在同一個 Docker network：`http://litellm:4000`。
3. 按「測試連線」。系統呼叫 proxy 的 `/v1/models` 列出模型別名，並從 `/model/info` 的 `model_info.supports_reasoning` 判斷哪些模型支援推理深度。
4. 勾選要加入評測選單的模型，按「儲存評測模型設定」。

儲存後，評測模型選單會出現「LiteLLM／claude-judge」這類選項；結果檔中保存的模型 ID 為 `litellm:claude-judge`，與 OpenAI 模型名稱分開。OpenAI 模型仍使用「GraphRAG 共用連線設定」。

## 行為說明

- **保存位置**：`projects/.judge_connection.json`（權限 600），與 GraphRAG 共用連線分開。`projects/` 已列在 `.gitignore`。
- **Key 讀取順序**：已儲存的 LiteLLM Key → 環境變數 `LITELLM_API_KEY`。儲存時 Key 欄留空會沿用已儲存的 Key；proxy 未設定 master key 時可以不填。
- **推理深度**：只有測試連線時回報 `supports_reasoning` 的模型才顯示推理深度選單並送出 `reasoning_effort`；其他模型不送此參數，改送 `temperature: 0` 與 `max_tokens`。
- **JSON 輸出**：一律先送 `response_format: json_object`。若 proxy 回 HTTP 400（模型不支援 JSON mode），會不帶此參數重送一次；模型回覆純文字時，從內容中擷取第一個 JSON 物件再驗證，仍失敗才判定該題評測失敗。
- 在 LiteLLM 改了 `config.yaml` 的模型後，回到第 0 頁重新「測試連線」並儲存，推理深度支援資訊才會更新。
