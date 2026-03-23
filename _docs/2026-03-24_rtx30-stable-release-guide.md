# Hypura RTX30 Stable Release Guide / リリースガイド

## 1) Artifact / 成果物

- File: `hypura-rtx30-windows-stable-2026-03-24.tar.gz`
- Target: Windows 11 + NVIDIA RTX 30 series (sm_86)
- Contents:
  - `hypura.exe`
  - `README.md`
  - `_docs/2026-03-24_rtx30-stable-release-guide.md`

## 2) Install (JA)

1. `tar.gz` を展開
2. `hypura.exe` を任意ディレクトリに配置
3. モデル (`.gguf`) のパスを用意
4. 実行:

```powershell
.\hypura.exe serve "F:\path\to\model.gguf" --port 8080 --context 1024
```

## 3) Usage (JA)

```powershell
Invoke-WebRequest http://127.0.0.1:8080/
Invoke-WebRequest http://127.0.0.1:8080/api/tags
Invoke-WebRequest -Uri http://127.0.0.1:8080/api/generate -Method POST -ContentType "application/json" -Body '{"model":"<model-name>","prompt":"hello","stream":false}'
```

## 6) Twitter / X Intro (<=139 chars)

`RTX30+CUDA12でローカルLLMを安定運用。HypuraならGGUFをOllama互換APIで即サーブ、GPU/RAM/NVMe階層配置で巨大モデルも攻められる。https://github.com/t8/hypura`

## 4) Install (EN)

1. Extract the `tar.gz`
2. Place `hypura.exe` in your preferred directory
3. Prepare a `.gguf` model path
4. Run:

```powershell
.\hypura.exe serve "F:\path\to\model.gguf" --port 8080 --context 1024
```

## 5) Usage (EN)

```powershell
Invoke-WebRequest http://127.0.0.1:8080/
Invoke-WebRequest http://127.0.0.1:8080/api/tags
Invoke-WebRequest -Uri http://127.0.0.1:8080/api/generate -Method POST -ContentType "application/json" -Body '{"model":"<model-name>","prompt":"hello","stream":false}'
```
