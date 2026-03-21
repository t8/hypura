#!/usr/bin/env bash
#
# Generate benchmark chart images from Hypura JSON results.
#
# Usage:
#   ./benchmarks/gen_charts.sh                    # auto-detect latest results
#   ./benchmarks/gen_charts.sh results/*.json     # specific files
#
# Output:
#   benchmarks/charts/*.png  — chart images
#   benchmarks/CHARTS.md     — markdown with embedded images
#
# Requirements: python3, matplotlib (pip3 install matplotlib)
#
# Each machine's best result per model is used. Run benchmarks first:
#   cargo run --release -- bench --max-tokens 30 ./test-models/model.gguf

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
CHARTS_DIR="$SCRIPT_DIR/charts"
OUTPUT="$SCRIPT_DIR/CHARTS.md"

mkdir -p "$CHARTS_DIR"

# Collect input files
if [ $# -gt 0 ]; then
    FILES=("$@")
else
    FILES=("$RESULTS_DIR"/*.json)
fi

if [ ${#FILES[@]} -eq 0 ]; then
    echo "No benchmark results found in $RESULTS_DIR"
    exit 1
fi

python3 - "$CHARTS_DIR" "$OUTPUT" "${FILES[@]}" <<'PYTHON'
import json, sys, os
from datetime import datetime

charts_dir = sys.argv[1]
output_md = sys.argv[2]
files = sys.argv[3:]

# --- Load and deduplicate results ---

results = []
for path in files:
    try:
        with open(path) as f:
            data = json.load(f)
        if data.get("hypura") and data["hypura"].get("tok_per_sec", 0) > 0:
            results.append(data)
    except Exception:
        continue

if not results:
    print("No valid benchmark results found.")
    sys.exit(1)

best = {}
for r in results:
    hw = r["hardware"]["cpu"]
    ram = r["hardware"]["ram_gb"]
    model = r["model"]["name"]
    tps = r["hypura"]["tok_per_sec"]
    key = (hw, ram, model)
    if key not in best or tps > best[key]["hypura"]["tok_per_sec"]:
        best[key] = r

entries = sorted(best.values(), key=lambda r: r["hypura"]["tok_per_sec"])

# --- Helpers ---

def short_model(name):
    name = name.replace("-instruct-v0.1", "")
    name = name.replace("-instruct", "")
    name = name.replace(".Q5_K_M", " Q5")
    name = name.replace(".Q4_K_M", " Q4")
    name = name.replace("-q4_k_m", " Q4")
    name = name.replace("-q5_k_m", " Q5")
    name = name.replace("Q4_K_M", "Q4")
    name = name.replace("Q5_K_M", "Q5")
    return name

def short_hw(cpu, ram_gb):
    chip = cpu.replace("Apple ", "")
    return f"{chip} {ram_gb:.0f}GB"

def format_size(gb):
    return f"{gb:.0f} GB" if gb >= 10 else f"{gb:.1f} GB"

def fits_in_gpu(r):
    """Model fits entirely in GPU (no NVMe/RAM spill)."""
    return r["placement"]["nvme_gb"] < 0.01 and r["placement"]["ram_gb"] < 0.01

def model_base_name(name):
    """Extract base model family name for color grouping."""
    n = name.lower()
    if "tinyllama" in n: return "tinyllama"
    if "qwen3-coder-next" in n or "qwen3-coder-next" in n.replace(" ", "-"): return "qwen3-coder-next"
    if "qwen" in n and "coder" in n and "32b" in n: return "qwen-coder-32b"
    if "qwen" in n and "32b" in n: return "qwen-32b"
    if "qwen" in n and "14b" in n: return "qwen-14b"
    if "qwen" in n: return "qwen"
    if "mixtral" in n: return "mixtral"
    if "llama" in n and "70b" in n: return "llama-70b"
    if "llama" in n: return "llama"
    return n.split("-")[0]

# Distinct colors per model family (colorblind-friendly palette)
MODEL_COLORS = {
    "tinyllama":         "#7ee787",   # green
    "qwen-14b":          "#58a6ff",   # blue
    "qwen-32b":          "#bc8cff",   # purple
    "qwen-coder-32b":    "#d2a8ff",   # light purple
    "qwen3-coder-next":  "#79c0ff",   # light blue
    "mixtral":           "#f0883e",   # orange
    "llama-70b":         "#f778ba",   # pink
}
MODEL_COLOR_DEFAULT = "#8b949e"  # gray fallback

def get_model_color(r):
    base = model_base_name(r["model"]["name"])
    return MODEL_COLORS.get(base, MODEL_COLOR_DEFAULT)

# --- Chart styling ---

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

BG = "#0d1117"
FG = "#c9d1d9"
ACCENT = "#58a6ff"
GREEN = "#3fb950"
ORANGE = "#d29922"
RED = "#f85149"
GRID = "#21262d"
BORDER = "#30363d"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "axes.edgecolor": BORDER,
    "axes.labelcolor": FG,
    "text.color": FG,
    "xtick.color": FG,
    "ytick.color": FG,
    "grid.color": GRID,
    "font.family": "monospace",
    "font.size": 12,
})

# --- Chart 1: Generation speed — only models that need NVMe streaming ---

from matplotlib.patches import Patch

# Filter to models that actually need Hypura (exceed GPU)
streaming_entries = [r for r in entries if not fits_in_gpu(r)]

if streaming_entries:
    fig, ax = plt.subplots(figsize=(10, max(3, len(streaming_entries) * 1.1 + 1.5)))

    models = []
    hypura_vals = []
    baseline_vals = []
    model_colors = []
    for r in streaming_entries:
        model = short_model(r["model"]["name"])
        size = format_size(r["model"]["size_gb"])
        hw = short_hw(r["hardware"]["cpu"], r["hardware"]["ram_gb"])
        models.append(f"{model}\n({size}) — {hw}")
        hypura_vals.append(r["hypura"]["tok_per_sec"])
        bl = r.get("baseline")
        baseline_vals.append(bl["tok_per_sec"] if bl and bl.get("tok_per_sec", 0) > 0 else 0)
        model_colors.append(get_model_color(r))

    y = range(len(models))
    bar_h = 0.35

    # Baseline bars — show "impossible" when OOM
    for i in y:
        if baseline_vals[i] > 0:
            ax.barh(i + bar_h/2, baseline_vals[i], height=bar_h,
                    color="#484f58", edgecolor=BORDER, zorder=1)
            offset = max(baseline_vals[i] * 0.02, 0.3)
            ax.text(baseline_vals[i] + offset, i + bar_h/2, f"{baseline_vals[i]:.1f}",
                    va="center", fontsize=9, color="#8b949e")
        else:
            ax.text(0.5, i + bar_h/2, "llama.cpp: crashes (model exceeds RAM)",
                    va="center", fontsize=9, color=RED, fontstyle="italic")

    # Hypura bars (colored by model)
    for i in y:
        ax.barh(i - bar_h/2, hypura_vals[i], height=bar_h,
                color=model_colors[i], edgecolor=BORDER, zorder=2)
        offset = max(hypura_vals[i] * 0.02, 0.3)
        ax.text(hypura_vals[i] + offset, i - bar_h/2, f"{hypura_vals[i]:.1f}",
                va="center", fontsize=10, fontweight="bold", color=FG)

    ax.set_yticks(list(y))
    ax.set_yticklabels(models, fontsize=10)
    for i in y:
        ax.get_yticklabels()[i].set_color(model_colors[i])
    ax.set_xlabel("tok/s (higher is better)", fontsize=11)
    ax.set_title("NVMe-Streaming Models: Hypura vs llama.cpp", fontsize=14, fontweight="bold", pad=12)
    ax.grid(axis="x", alpha=0.3)
    all_vals = hypura_vals + [v for v in baseline_vals if v > 0]
    ax.set_xlim(0, max(all_vals) * 1.18)

    seen = {}
    legend_elements = []
    for r, c in zip(streaming_entries, model_colors):
        base = model_base_name(r["model"]["name"])
        if base not in seen:
            seen[base] = True
            legend_elements.append(Patch(facecolor=c, edgecolor=BORDER, label=base))
    legend_elements.append(Patch(facecolor="#484f58", edgecolor=BORDER, label="llama.cpp baseline"))
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9,
              facecolor=BG, edgecolor=BORDER, labelcolor=FG)

    fig.tight_layout()
    fig.savefig(os.path.join(charts_dir, "generation_speed.png"), dpi=150, bbox_inches="tight")
    plt.close()

# --- Chart 1b: GPU-resident models speed ---

gpu_entries = [r for r in entries if fits_in_gpu(r)]

if gpu_entries:
    fig, ax = plt.subplots(figsize=(10, max(3, len(gpu_entries) * 1.1 + 1.5)))

    models = []
    hypura_vals = []
    baseline_vals = []
    model_colors = []
    for r in gpu_entries:
        model = short_model(r["model"]["name"])
        size = format_size(r["model"]["size_gb"])
        hw = short_hw(r["hardware"]["cpu"], r["hardware"]["ram_gb"])
        models.append(f"{model}\n({size}) — {hw}")
        hypura_vals.append(r["hypura"]["tok_per_sec"])
        bl = r.get("baseline")
        baseline_vals.append(bl["tok_per_sec"] if bl and bl.get("tok_per_sec", 0) > 0 else 0)
        model_colors.append(get_model_color(r))

    y = range(len(models))
    bar_h = 0.35

    for i in y:
        if baseline_vals[i] > 0:
            ax.barh(i + bar_h/2, baseline_vals[i], height=bar_h,
                    color="#484f58", edgecolor=BORDER, zorder=1)
            offset = max(baseline_vals[i] * 0.02, 0.3)
            ax.text(baseline_vals[i] + offset, i + bar_h/2, f"{baseline_vals[i]:.1f}",
                    va="center", fontsize=9, color="#8b949e")

    for i in y:
        ax.barh(i - bar_h/2, hypura_vals[i], height=bar_h,
                color=model_colors[i], edgecolor=BORDER, zorder=2)
        offset = max(hypura_vals[i] * 0.02, 0.3)
        ax.text(hypura_vals[i] + offset, i - bar_h/2, f"{hypura_vals[i]:.1f}",
                va="center", fontsize=10, fontweight="bold", color=FG)

    ax.set_yticks(list(y))
    ax.set_yticklabels(models, fontsize=10)
    for i in y:
        ax.get_yticklabels()[i].set_color(model_colors[i])
    ax.set_xlabel("tok/s (higher is better)", fontsize=11)
    ax.set_title("GPU-Resident Models (no NVMe needed)", fontsize=14, fontweight="bold", pad=12)
    ax.grid(axis="x", alpha=0.3)
    all_vals = hypura_vals + [v for v in baseline_vals if v > 0]
    ax.set_xlim(0, max(all_vals) * 1.18)

    seen = {}
    legend_elements = []
    for r, c in zip(gpu_entries, model_colors):
        base = model_base_name(r["model"]["name"])
        if base not in seen:
            seen[base] = True
            legend_elements.append(Patch(facecolor=c, edgecolor=BORDER, label=base))
    legend_elements.append(Patch(facecolor="#484f58", edgecolor=BORDER, label="llama.cpp baseline"))
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9,
              facecolor=BG, edgecolor=BORDER, labelcolor=FG)

    fig.tight_layout()
    fig.savefig(os.path.join(charts_dir, "generation_speed_gpu.png"), dpi=150, bbox_inches="tight")
    plt.close()

# --- Chart 2: Memory placement ---

fig, ax = plt.subplots(figsize=(10, max(3, len(entries) * 0.9 + 1.2)))

GPU_COLOR = "#58a6ff"
RAM_COLOR = "#d29922"
NVME_COLOR = "#8b949e"

models_p = []
for i, r in enumerate(entries):
    model = short_model(r["model"]["name"])
    hw = short_hw(r["hardware"]["cpu"], r["hardware"]["ram_gb"])
    models_p.append(f"{model}\n{hw}")
    gpu = r["placement"]["gpu_gb"]
    ram = r["placement"]["ram_gb"]
    nvme = r["placement"]["nvme_gb"]

    ax.barh(i, gpu, color=GPU_COLOR, height=0.6, edgecolor=BORDER)
    ax.barh(i, ram, left=gpu, color=RAM_COLOR, height=0.6, edgecolor=BORDER)
    ax.barh(i, nvme, left=gpu + ram, color=NVME_COLOR, height=0.6, edgecolor=BORDER)

    total = gpu + ram + nvme
    parts = []
    if gpu > 0.01: parts.append(f"{format_size(gpu)} GPU")
    if ram > 0.01: parts.append(f"{format_size(ram)} RAM")
    if nvme > 0.01: parts.append(f"{format_size(nvme)} NVMe")
    ax.text(total + 0.3, i, " | ".join(parts), va="center", fontsize=9, color=FG)

ax.set_yticks(range(len(models_p)))
ax.set_yticklabels(models_p, fontsize=10)
# Color y-axis labels by model
for i, r in enumerate(entries):
    ax.get_yticklabels()[i].set_color(get_model_color(r))
ax.set_xlabel("Size (GB)", fontsize=11)
ax.set_title("Memory Placement by Tier", fontsize=14, fontweight="bold", pad=12)
ax.grid(axis="x", alpha=0.3)
ax.set_xlim(0, max(r["model"]["size_gb"] for r in entries) * 1.45)

has_ram = any(r["placement"]["ram_gb"] > 0.01 for r in entries)
legend_elements = [
    Patch(facecolor=GPU_COLOR, edgecolor=BORDER, label="GPU (Metal)"),
]
if has_ram:
    legend_elements.append(Patch(facecolor=RAM_COLOR, edgecolor=BORDER, label="RAM"))
has_nvme = any(r["placement"]["nvme_gb"] > 0.01 for r in entries)
if has_nvme:
    legend_elements.append(Patch(facecolor=NVME_COLOR, edgecolor=BORDER, label="NVMe"))
ax.legend(handles=legend_elements, loc="upper right", fontsize=9,
          facecolor=BG, edgecolor=BORDER, labelcolor=FG)

fig.tight_layout()
fig.savefig(os.path.join(charts_dir, "memory_placement.png"), dpi=150, bbox_inches="tight")
plt.close()

# --- Chart 3: Prompt eval vs generation — only NVMe-streaming models ---

streaming_entries_3 = [r for r in entries if not fits_in_gpu(r)]

if streaming_entries_3:
    fig, ax = plt.subplots(figsize=(10, max(3, len(streaming_entries_3) * 0.9 + 1.2)))

    models_e = []
    prompt_vals = []
    gen_vals = []
    for r in streaming_entries_3:
        model = short_model(r["model"]["name"])
        hw = short_hw(r["hardware"]["cpu"], r["hardware"]["ram_gb"])
        models_e.append(f"{model}\n{hw}")
        prompt_ms = r["hypura"]["prompt_eval_ms"]
        prompt_token_est = int(len(r["config"]["prompt"].split()) * 1.3)
        prompt_tps = prompt_token_est / (prompt_ms / 1000) if prompt_ms > 0 else 0
        prompt_vals.append(prompt_tps)
        gen_vals.append(r["hypura"]["tok_per_sec"])

    y = range(len(models_e))
    bar_h = 0.3
    ax.barh([i + bar_h/2 for i in y], prompt_vals, height=bar_h,
            color=ACCENT, edgecolor=BORDER, label="Prompt eval")
    ax.barh([i - bar_h/2 for i in y], gen_vals, height=bar_h,
            color=GREEN, edgecolor=BORDER, label="Generation")

    for i in y:
        offset_p = max(prompt_vals[i] * 0.02, 0.2)
        offset_g = max(gen_vals[i] * 0.02, 0.2)
        ax.text(prompt_vals[i] + offset_p, i + bar_h/2, f"{prompt_vals[i]:.1f}",
                va="center", fontsize=9, color=FG)
        ax.text(gen_vals[i] + offset_g, i - bar_h/2, f"{gen_vals[i]:.1f}",
                va="center", fontsize=9, color=FG)

    ax.set_yticks(list(y))
    ax.set_yticklabels(models_e, fontsize=10)
    ax.set_xlabel("tok/s", fontsize=11)
    ax.set_title("NVMe-Streaming: Prompt Eval vs Generation", fontsize=14, fontweight="bold", pad=12)
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlim(0, max(max(prompt_vals), max(gen_vals)) * 1.15)
    ax.legend(loc="lower right", fontsize=9, facecolor=BG, edgecolor=BORDER, labelcolor=FG)

    fig.tight_layout()
    fig.savefig(os.path.join(charts_dir, "prompt_vs_generation.png"), dpi=150, bbox_inches="tight")
    plt.close()

# --- Generate CHARTS.md ---

with open(output_md, "w") as f:
    f.write("# Hypura Benchmark Charts\n\n")
    f.write("*Auto-generated by `benchmarks/gen_charts.sh`*\n\n")

    f.write("## NVMe-Streaming Models\n\n")
    f.write("![Generation Speed](charts/generation_speed.png)\n\n")

    f.write("## GPU-Resident Models\n\n")
    f.write("![GPU-Resident Speed](charts/generation_speed_gpu.png)\n\n")

    f.write("## Memory Placement\n\n")
    f.write("![Memory Placement](charts/memory_placement.png)\n\n")

    f.write("## Prompt Eval vs Generation\n\n")
    f.write("![Prompt vs Generation](charts/prompt_vs_generation.png)\n\n")

    f.write("## Hardware\n\n")
    machines = set()
    for r in best.values():
        hw = r["hardware"]
        machines.add((hw["cpu"], hw["ram_gb"], hw.get("nvme_seq_gbps", 0)))
    for cpu, ram, nvme in sorted(machines):
        f.write(f"- **{cpu}**, {ram:.0f} GB unified, {nvme:.1f} GB/s NVMe seq read\n")
    f.write(f"\n*Generated {datetime.now().strftime('%Y-%m-%d')}*\n")

print(f"Charts written to {charts_dir}/")
print(f"Markdown written to {output_md}")
PYTHON
