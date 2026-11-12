# Hypura macOS App & Model Management CLI Guide

Hypura now provides an Ollama-grade experience on macOS: a **global CLI**, an **Ollama-compatible HTTP API**, and a native **macOS Menu Bar Application (`Hypura.app`)**.

---

## 1. Global CLI Usage

Once installed, `hypura` is accessible from any terminal window:

### Discovered Models
```bash
# List all models discovered in your models library and ~/.ollama/models
hypura list

# Scan a custom models directory
hypura list --models-dir /Volumes/RealtekData/models
```

### Pulling Models (`hypura pull`)
Downloads model weights from Hugging Face with automatic resume (`curl -C -`), live progress reporting (speed, percentage, ETA), and atomic `.part` verification:

```bash
# Pull by 2026/2025 alias:
hypura pull gemma-4:26b       # Google Gemma 4 26B-A4B (~14.8 GB, ~22-28 tok/s)
hypura pull qwen3.8:27b       # Qwen 3.8 27B Instruct (~15.8 GB, ~17-19 tok/s)
hypura pull qwen2.5-coder:32b # Qwen 2.5 Coder 32B Instruct (~15.4 GB)
hypura pull mistral-small:24b # Mistral Small 24B Instruct 2501 (~14.3 GB)

# Pull any HuggingFace repository:quant
hypura pull bartowski/google_gemma-4-26B-A4B-it-GGUF:Q4_K_M

# Pull a direct GGUF URL
hypura pull https://example.com/models/my-model-q4_k_m.gguf

# Specify target models destination directory
hypura pull gemma-4:26b --models-dir /Volumes/RealtekData/models
```

### Removing Models (`hypura rm` / `delete`)
Safely unlinks model files from the library and cleans up parent directories:

```bash
hypura rm gemma-4:26b
# or
hypura delete qwen3.8:27b
```
* **Multi-Shard Aware**: Automatically detects and deletes all associated split shards (`*-00001-of-*`, `*-00002-of-*`, etc.).
* **Active Unload**: If the model is currently resident in unified memory, it is unallocated cleanly before removal.

### Serving Models (`hypura serve`)
```bash
# Start background server with custom context and idle timeout
hypura serve --models-dir /Volumes/RealtekData/models --context 32768 --idle-timeout 1800 --kv-quant q8_0 --port 6000

# Pre-warm a specific model on startup
hypura serve gemma-4:26b -c 32768 -p 6000
```

### Checking Active Processes (`hypura ps`)
```bash
hypura ps --port 6000
```

---

## 2. Ollama-Compatible HTTP API

Hypura exposes full parity with Ollama client tools, web UIs (Open-WebUI), and agentic frameworks:

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/api/pull` | `POST` | Pull a model with streaming NDJSON progress chunks (`{"status": "downloading", "total": ..., "completed": ...}`) |
| `/api/delete` | `DELETE` | Remove a model from the library |
| `/api/tags` | `GET` | List all discovered models and quantization tags |
| `/api/ps` | `GET` | List running models, context sizes, and VRAM residency |
| `/api/show` | `POST` | Inspect model architecture, parameters, and metadata |
| `/api/chat` | `POST` | Multi-turn chat completions (Ollama format) |
| `/api/generate` | `POST` | Raw text completions with streaming keep-alive heartbeats |
| `/v1/chat/completions` | `POST` | OpenAI-compatible chat and native tool calling |

### Example: Pull via API
```bash
curl http://localhost:6000/api/pull -d '{
  "name": "gemma-4:26b"
}'
```

### Example: Delete via API
```bash
curl -X DELETE http://localhost:6000/api/delete -d '{
  "name": "gemma-4:26b"
}'
```

---

## 3. Native macOS Menu Bar App (`Hypura.app`)

The native macOS application provides a background menu bar tray icon (`⚡`), identical to Ollama:

* **Location**: `/Applications/Hypura.app`
* **Agent App**: Runs in the top menu bar (`LSUIElement = true`) without docking.
* **Live Status**: Shows green indicator when the server is healthy (`🟢 Running on port 6000`).
* **One-Click Actions**:
  * Copy API URL (`http://localhost:6000`)
  * Open Models Folder in Finder
  * Install / Update global `hypura` CLI symlink
  * Restart Server
  * Quit

### Building & Installing `Hypura.app`
To rebuild the app bundle from source:
```bash
./scripts/package_macos_app.sh
cp -R dist/Hypura.app /Applications/
open /Applications/Hypura.app
```

---

## 4. Runtime Models Library Overrides

Hypura defaults to standard model discovery (`~/.ollama/models`, `/Volumes/RealtekData/models`, and `~/.hypura/models`). You can override this dynamically at any time:

1. **CLI Flag**: Pass `--models-dir <PATH>` to any CLI command (`serve`, `list`, `run`, `estimate`, `pull`, `rm`).
2. **Environment Variable**: Set `export HYPURA_MODELS_DIR=/Volumes/RealtekData/models`.
3. **HTTP API Request**: Include `"models_dir": "/path/to/models"` in your `/api/pull` or `/api/delete` JSON body.
