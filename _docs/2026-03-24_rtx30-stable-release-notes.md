## Hypura RTX30 Stable (Windows)

### 日本語
- 対象: Windows 11 + NVIDIA RTX 30 系 (sm_86)
- 同梱成果物: `hypura-rtx30-windows-stable-2026-03-24.tar.gz`
- 同梱内容: `hypura.exe`, `README.md`, `2026-03-24_rtx30-stable-release-guide.md`

#### 導入方法
1. リリース資産 `hypura-rtx30-windows-stable-2026-03-24.tar.gz` をダウンロード
2. 展開して `hypura.exe` を配置
3. GGUF モデルを指定して起動

```powershell
.\hypura.exe serve "F:\path\to\model.gguf" --port 8080 --context 1024
```

#### 使用方法 (API smoke)
```powershell
Invoke-WebRequest http://127.0.0.1:8080/
Invoke-WebRequest http://127.0.0.1:8080/api/tags
Invoke-WebRequest -Uri http://127.0.0.1:8080/api/generate -Method POST -ContentType "application/json" -Body '{"model":"<model-name>","prompt":"hello","stream":false}'
```

### English
- Target: Windows 11 + NVIDIA RTX 30 series (sm_86)
- Artifact: `hypura-rtx30-windows-stable-2026-03-24.tar.gz`
- Bundle: `hypura.exe`, `README.md`, `2026-03-24_rtx30-stable-release-guide.md`

#### Installation
1. Download release asset `hypura-rtx30-windows-stable-2026-03-24.tar.gz`
2. Extract and place `hypura.exe`
3. Start server with your GGUF model

```powershell
.\hypura.exe serve "F:\path\to\model.gguf" --port 8080 --context 1024
```

#### Usage (API smoke)
```powershell
Invoke-WebRequest http://127.0.0.1:8080/
Invoke-WebRequest http://127.0.0.1:8080/api/tags
Invoke-WebRequest -Uri http://127.0.0.1:8080/api/generate -Method POST -ContentType "application/json" -Body '{"model":"<model-name>","prompt":"hello","stream":false}'
```
