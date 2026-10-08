#!/usr/bin/env bash
set -e

# ==============================================================================
# Hypura Startup Script: Dynamic Multi-Model Serve & Tailscale Funnel
# ==============================================================================

# Configuration with environment variable overrides
PORT="${HYPURA_PORT:-6000}"
CONTEXT="${HYPURA_CONTEXT:-32768}"  # Default 32k context window (client can override per request via num_ctx)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$SCRIPT_DIR"

if [ -f "$SCRIPT_DIR/../target/release/hypura" ]; then
    ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
fi

HYPURA_BIN="$ROOT_DIR/target/release/hypura"

# Parse CLI arguments: allow passing model or arbitrary additional flags
# e.g.: ./start_hypura_funnel.sh --context 65536
#       ./start_hypura_funnel.sh my_model -c 65536
MODEL=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -c|--context|--ctx-size)
            CONTEXT="$2"
            shift 2
            ;;
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [model_name] [-c|--context <tokens>] [-p|--port <port>] [extra hypura flags...]"
            echo ""
            echo "Environment variables:"
            echo "  HYPURA_CONTEXT   Default context length (default: 32768)"
            echo "  HYPURA_PORT      Server port (default: 6000)"
            echo "  HYPURA_EXTRA_ARGS Extra flags passed to hypura serve"
            exit 0
            ;;
        -*)
            EXTRA_ARGS+=("$1")
            shift
            ;;
        *)
            if [ -z "$MODEL" ]; then
                MODEL="$1"
            else
                EXTRA_ARGS+=("$1")
            fi
            shift
            ;;
    esac
done

if [ -n "$HYPURA_EXTRA_ARGS" ]; then
    EXTRA_ARGS+=($HYPURA_EXTRA_ARGS)
fi

# Build binary if not present
if [ ! -f "$HYPURA_BIN" ]; then
    echo "Building Hypura release binary..."
    (cd "$ROOT_DIR" && cargo build --release)
fi

# Stop any existing server on this port
echo "Stopping any existing Hypura instance..."
pkill -f "hypura serve" || true
sleep 1

# Setup Tailscale Funnel in background
if command -v tailscale &> /dev/null; then
    echo "Configuring Tailscale Funnel on port $PORT..."
    tailscale funnel --bg --yes $PORT || true
    echo "Tailscale Funnel active."
else
    echo "Warning: tailscale CLI not found in PATH."
fi

# Display Network Endpoints
LOCAL_IP=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo "127.0.0.1")
TAILSCALE_IP=$(tailscale ip -4 2>/dev/null || echo "")
FUNNEL_URL=$(tailscale funnel status 2>/dev/null | grep -o 'https://[a-zA-Z0-9.-]*\.ts\.net' | head -n 1)

echo ""
echo "=========================================================="
echo " Hypura Ollama & OpenAI Compatible Multi-Model Server"
if [ -n "$MODEL" ]; then
    echo " Mode:           Pre-warmed ($MODEL)"
else
    echo " Mode:           Dynamic On-Demand (All Local + Ollama Models)"
fi
echo " Context:        $CONTEXT default tokens (client adjustable)"
echo " Port:           $PORT"
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

# Start Hypura Server (with or without pre-warmed model)
if [ -n "$MODEL" ]; then
    exec "$HYPURA_BIN" serve "$MODEL" --host 0.0.0.0 --port "$PORT" --context "$CONTEXT" "${EXTRA_ARGS[@]}"
else
    exec "$HYPURA_BIN" serve --host 0.0.0.0 --port "$PORT" --context "$CONTEXT" "${EXTRA_ARGS[@]}"
fi
