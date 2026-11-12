#!/usr/bin/env bash
set -e

# ==============================================================================
# Hypura Startup Script: Qwen 3.8 Flash Next (180B Ultra-Sparse Hybrid)
# ==============================================================================
# Optimized for Apple Silicon Unified Memory (M4 Pro / Max)
# - Routes 51B sparse N-gram lookup tables to Host RAM / NVMe
# - Keeps 6B active compute layers resident in Metal GPU
# - Quantizes 32k KV Cache to Q8_0 (~4.1 GB instead of 8.2 GB)
# - Auto-unloads after inactivity to free unified memory
# ==============================================================================

# Default Configuration (can be overridden via environment variables or CLI flags)
MODEL="${HYPURA_MODEL:-qwen3.8-flash-next}"
MODELS_DIR="${HYPURA_MODELS_DIR:-/Volumes/RealtekData/models}"
CONTEXT="${HYPURA_CONTEXT:-32768}"
IDLE_TIMEOUT="${HYPURA_IDLE_TIMEOUT:-1800}"
KV_QUANT="${HYPURA_KV_QUANT:-q8_0}"
PORT="${HYPURA_PORT:-6000}"
HOST="${HYPURA_HOST:-0.0.0.0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$SCRIPT_DIR"

if [ -f "$SCRIPT_DIR/../target/release/hypura" ]; then
    ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
fi

HYPURA_BIN="$ROOT_DIR/target/release/hypura"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        --models-dir)
            MODELS_DIR="$2"
            shift 2
            ;;
        -c|--context|--ctx-size)
            CONTEXT="$2"
            shift 2
            ;;
        --idle-timeout)
            IDLE_TIMEOUT="$2"
            shift 2
            ;;
        --kv-quant)
            KV_QUANT="$2"
            shift 2
            ;;
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options] [extra hypura flags...]"
            echo ""
            echo "Options:"
            echo "  -m, --model <NAME>         Model name or GGUF path (default: $MODEL)"
            echo "      --models-dir <DIR>     Models directory to scan (default: $MODELS_DIR)"
            echo "  -c, --context <TOKENS>     Context window size (default: $CONTEXT)"
            echo "      --idle-timeout <SECS>  Idle unload timeout in seconds (default: $IDLE_TIMEOUT)"
            echo "      --kv-quant <TYPE>      KV cache quantization: f16, q8_0, q4_0 (default: $KV_QUANT)"
            echo "  -p, --port <PORT>          HTTP port (default: $PORT)"
            echo "      --host <HOST>          Bind host (default: $HOST)"
            echo ""
            echo "Environment variables:"
            echo "  HYPURA_MODEL, HYPURA_MODELS_DIR, HYPURA_CONTEXT, HYPURA_IDLE_TIMEOUT, HYPURA_KV_QUANT, HYPURA_PORT"
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# Build release binary if missing
if [ ! -f "$HYPURA_BIN" ]; then
    echo "Building Hypura release binary..."
    (cd "$ROOT_DIR" && cargo build --release)
fi

# Stop any previous server instance on this port
echo "Stopping any existing Hypura instance..."
pkill -f "hypura serve" || true
sleep 1

# Setup Tailscale Funnel in background if available
if command -v tailscale &> /dev/null; then
    echo "Configuring Tailscale Funnel on port $PORT..."
    tailscale funnel --bg --yes "$PORT" 2>/dev/null || true
fi

# Network IP resolution
LOCAL_IP=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo "127.0.0.1")
TAILSCALE_IP=$(tailscale ip -4 2>/dev/null || echo "")
FUNNEL_URL=$(tailscale funnel status 2>/dev/null | grep -o 'https://[a-zA-Z0-9.-]*\.ts\.net' | head -n 1 || echo "")

echo ""
echo "=========================================================="
echo " Hypura Server: Qwen 3.8 Flash Next (Sparse MoE + N-gram)"
echo "=========================================================="
echo " Model:          $MODEL"
echo " Models Dir:     $MODELS_DIR"
echo " Context:        $CONTEXT tokens"
echo " Idle Timeout:   ${IDLE_TIMEOUT}s (auto-unload unified memory)"
echo " KV Quant:       $KV_QUANT"
echo " Port:           $PORT (Host: $HOST)"
echo "----------------------------------------------------------"
echo " Local URL:      http://localhost:$PORT"
echo " LAN URL:        http://$LOCAL_IP:$PORT"
if [ -n "$TAILSCALE_IP" ]; then
    echo " Tailscale IP:   http://$TAILSCALE_IP:$PORT"
fi
if [ -n "$FUNNEL_URL" ]; then
    echo " Public HTTPS:   $FUNNEL_URL"
fi
echo " APIs:           Ollama (/api/*) & OpenAI (/v1/*)"
echo "=========================================================="
echo ""

exec "$HYPURA_BIN" serve \
  --models-dir "$MODELS_DIR" \
  --model "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --context "$CONTEXT" \
  --idle-timeout "$IDLE_TIMEOUT" \
  --kv-quant "$KV_QUANT" \
  "${EXTRA_ARGS[@]}"

