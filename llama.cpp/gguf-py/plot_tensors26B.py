import sys
import os
import re
import csv
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from gguf.gguf_reader import GGUFReader


# ============================================================
# 1. Tensor classification
# ============================================================

def get_layer_id(name: str):
    """
    Extract layer id from tensor name like:
      blk.0.attn_q.weight -> 0
      blk.29.ffn_down_exps.weight -> 29

    Return None for global tensors.
    """
    m = re.match(r"blk\.(\d+)\.", name)
    if m:
        return int(m.group(1))
    return None


def classify_tensor(name: str):
    """
    Return:
      family: coarse group for high-level visualization
      role:   detailed role for analysis
    """
    n = name.lower()

    # -----------------------------
    # Global tensors
    # -----------------------------
    if n == "token_embd.weight":
        return "global", "token_embedding"

    if n == "rope_freqs.weight":
        return "global", "rope_freqs"

    if n == "output_norm.weight":
        return "global", "output_norm"

    # -----------------------------
    # MoE expert tensors
    # Main target for hot/cold analysis
    # -----------------------------
    if "ffn_gate_up_exps.weight" in n:
        return "moe_expert", "expert_gate_up"

    if "ffn_down_exps.weight" in n:
        return "moe_expert", "expert_down"

    if "ffn_down_exps.scale" in n:
        return "moe_expert", "expert_down_scale"

    # -----------------------------
    # MoE router
    # Always-hot baseline
    # -----------------------------
    if "ffn_gate_inp.weight" in n:
        return "moe_router", "router_weight"

    if "ffn_gate_inp.scale" in n:
        return "moe_router", "router_scale"

    # -----------------------------
    # Shared / dense FFN
    # Usually hot baseline
    # -----------------------------
    if "ffn_gate.weight" in n:
        return "shared_ffn", "shared_ffn_gate"

    if "ffn_up.weight" in n:
        return "shared_ffn", "shared_ffn_up"

    if "ffn_down.weight" in n:
        return "shared_ffn", "shared_ffn_down"

    # -----------------------------
    # Attention projection
    # -----------------------------
    if "attn_q.weight" in n:
        return "attention", "attn_q"

    if "attn_k.weight" in n:
        return "attention", "attn_k"

    if "attn_v.weight" in n:
        return "attention", "attn_v"

    if "attn_output.weight" in n:
        return "attention", "attn_output"

    # -----------------------------
    # Norms and residual scales
    # -----------------------------
    if "norm" in n:
        return "norm_scale", "norm"

    if "layer_output_scale" in n:
        return "norm_scale", "layer_output_scale"

    return "other", "other"


def safe_shape(tensor):
    try:
        return "x".join(str(int(x)) for x in tensor.shape)
    except Exception:
        return ""


def safe_type(tensor):
    for attr in ["tensor_type", "type"]:
        if hasattr(tensor, attr):
            return str(getattr(tensor, attr))
    return ""


def bytes_to_gib(x):
    return x / (1024 ** 3)


def bytes_to_mib(x):
    return x / (1024 ** 2)


# ============================================================
# 2. Read GGUF tensors
# ============================================================

def read_tensor_records(path):
    reader = GGUFReader(path)
    records = []

    for idx, t in enumerate(reader.tensors):
        name = str(t.name)
        layer = get_layer_id(name)
        family, role = classify_tensor(name)

        offset = int(t.data_offset)
        size = int(t.n_bytes)
        end = offset + size

        records.append({
            "index": idx,
            "name": name,
            "layer": layer,
            "family": family,
            "role": role,
            "offset": offset,
            "end": end,
            "size_bytes": size,
            "size_mib": bytes_to_mib(size),
            "size_gib": bytes_to_gib(size),
            "shape": safe_shape(t),
            "type": safe_type(t),
        })

    records.sort(key=lambda r: r["offset"])
    return records


# ============================================================
# 3. CSV reports
# ============================================================

def write_detail_csv(records, out_dir):
    path = os.path.join(out_dir, "tensor_detail.csv")

    fields = [
        "index",
        "name",
        "layer",
        "family",
        "role",
        "offset",
        "end",
        "size_bytes",
        "size_mib",
        "size_gib",
        "shape",
        "type",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in records:
            writer.writerow(r)

    return path


def write_layer_family_csv(records, out_dir):
    path = os.path.join(out_dir, "layer_family_summary.csv")

    summary = defaultdict(lambda: defaultdict(int))

    for r in records:
        if r["layer"] is None:
            continue
        summary[r["layer"]][r["family"]] += r["size_bytes"]

    families = [
        "attention",
        "shared_ffn",
        "moe_router",
        "moe_expert",
        "norm_scale",
        "other",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["layer"] + [f"{fam}_gib" for fam in families] + ["total_gib"])

        for layer in sorted(summary.keys()):
            row = [layer]
            total = 0
            for fam in families:
                b = summary[layer][fam]
                total += b
                row.append(bytes_to_gib(b))
            row.append(bytes_to_gib(total))
            writer.writerow(row)

    return path


def write_top_tensors_csv(records, out_dir, k=80):
    path = os.path.join(out_dir, "top_tensors_by_size.csv")

    top = sorted(records, key=lambda r: r["size_bytes"], reverse=True)[:k]

    fields = [
        "rank",
        "name",
        "layer",
        "family",
        "role",
        "size_gib",
        "size_mib",
        "offset",
        "end",
        "shape",
        "type",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for rank, r in enumerate(top, 1):
            writer.writerow({
                "rank": rank,
                "name": r["name"],
                "layer": r["layer"],
                "family": r["family"],
                "role": r["role"],
                "size_gib": r["size_gib"],
                "size_mib": r["size_mib"],
                "offset": r["offset"],
                "end": r["end"],
                "shape": r["shape"],
                "type": r["type"],
            })

    return path


# ============================================================
# 4. Visualization 1: Layer x Family heatmap
# ============================================================

def plot_layer_family_heatmap(records, out_dir):
    families = [
        "attention",
        "shared_ffn",
        "moe_router",
        "moe_expert",
        "norm_scale",
        "other",
    ]

    layers = sorted({r["layer"] for r in records if r["layer"] is not None})

    matrix = np.zeros((len(families), len(layers)), dtype=np.float64)

    layer_to_col = {layer: i for i, layer in enumerate(layers)}
    family_to_row = {fam: i for i, fam in enumerate(families)}

    for r in records:
        if r["layer"] is None:
            continue

        fam = r["family"]
        if fam not in family_to_row:
            fam = "other"

        row = family_to_row[fam]
        col = layer_to_col[r["layer"]]
        matrix[row, col] += bytes_to_gib(r["size_bytes"])

    fig, ax = plt.subplots(figsize=(18, 5))

    im = ax.imshow(matrix, aspect="auto")

    ax.set_title("Layer × Tensor Family Size Heatmap")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Tensor family")

    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers, rotation=90)

    ax.set_yticks(range(len(families)))
    ax.set_yticklabels(families)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Size per layer-family (GiB)")

    # Annotate only large values to avoid clutter
    max_val = matrix.max() if matrix.size else 0
    threshold = max_val * 0.25

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if val >= threshold and val > 0:
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7)

    plt.tight_layout()

    path = os.path.join(out_dir, "01_layer_family_heatmap.png")
    plt.savefig(path, dpi=200)
    plt.close(fig)

    return path


# ============================================================
# 5. Visualization 2: Stacked bar by layer
# ============================================================

def plot_layer_stacked_bar(records, out_dir):
    families = [
        "attention",
        "shared_ffn",
        "moe_router",
        "moe_expert",
        "norm_scale",
        "other",
    ]

    colors = {
        "attention": "#4C78A8",
        "shared_ffn": "#59A14F",
        "moe_router": "#F28E2B",
        "moe_expert": "#E15759",
        "norm_scale": "#9C755F",
        "other": "#BAB0AC",
    }

    layers = sorted({r["layer"] for r in records if r["layer"] is not None})
    layer_to_idx = {layer: i for i, layer in enumerate(layers)}

    data = {fam: np.zeros(len(layers), dtype=np.float64) for fam in families}

    for r in records:
        if r["layer"] is None:
            continue

        fam = r["family"]
        if fam not in data:
            fam = "other"

        idx = layer_to_idx[r["layer"]]
        data[fam][idx] += bytes_to_gib(r["size_bytes"])

    fig, ax = plt.subplots(figsize=(18, 6))

    bottom = np.zeros(len(layers), dtype=np.float64)

    for fam in families:
        ax.bar(
            layers,
            data[fam],
            bottom=bottom,
            label=fam,
            color=colors[fam],
            width=0.85,
        )
        bottom += data[fam]

    ax.set_title("Per-layer Weight Composition")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Tensor size per layer (GiB)")
    ax.set_xticks(layers)
    ax.set_xticklabels(layers, rotation=90)
    ax.legend(ncol=3, loc="upper right")

    plt.tight_layout()

    path = os.path.join(out_dir, "02_layer_stacked_bar.png")
    plt.savefig(path, dpi=200)
    plt.close(fig)

    return path


# ============================================================
# 6. Visualization 3: Better offset timeline with lanes
# ============================================================

def plot_offset_lanes(records, out_dir):
    """
    Better replacement for your current single-line colored bar.

    Instead of drawing every tensor at y=0,
    we draw different families on different y lanes.
    """
    families = [
        "global",
        "attention",
        "shared_ffn",
        "moe_router",
        "moe_expert",
        "norm_scale",
        "other",
    ]

    colors = {
        "global": "#222222",
        "attention": "#4C78A8",
        "shared_ffn": "#59A14F",
        "moe_router": "#F28E2B",
        "moe_expert": "#E15759",
        "norm_scale": "#9C755F",
        "other": "#BAB0AC",
    }

    lane_y = {fam: i for i, fam in enumerate(families)}

    fig, ax = plt.subplots(figsize=(22, 7))

    for r in records:
        fam = r["family"]
        if fam not in lane_y:
            fam = "other"

        y = lane_y[fam]
        start_gib = bytes_to_gib(r["offset"])
        width_gib = bytes_to_gib(r["size_bytes"])

        ax.broken_barh(
            [(start_gib, width_gib)],
            (y - 0.35, 0.7),
            facecolors=colors[fam],
            edgecolors="none",
            alpha=0.9,
        )

    # Layer boundary markers
    layer_min_offset = {}
    for r in records:
        if r["layer"] is None:
            continue
        layer = r["layer"]
        layer_min_offset[layer] = min(layer_min_offset.get(layer, r["offset"]), r["offset"])

    for layer, offset in sorted(layer_min_offset.items()):
        x = bytes_to_gib(offset)
        ax.axvline(x, color="black", linewidth=0.25, alpha=0.25)

        # Label every 5 layers to avoid clutter
        if layer % 5 == 0:
            ax.text(
                x,
                len(families) - 0.2,
                f"blk.{layer}",
                rotation=90,
                fontsize=8,
                va="bottom",
                ha="center",
            )

    ax.set_yticks(range(len(families)))
    ax.set_yticklabels(families)

    ax.set_xlabel("File offset (GiB)")
    ax.set_title("GGUF Tensor Layout by File Offset, Split into Semantic Lanes")

    ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.3)

    plt.tight_layout()

    path = os.path.join(out_dir, "03_offset_lanes.png")
    plt.savefig(path, dpi=200)
    plt.close(fig)

    return path


# ============================================================
# 7. Visualization 4: MoE-focused offset timeline
# ============================================================

def plot_moe_offset_focus(records, out_dir):
    """
    Focus only on tensors related to MoE.
    This is the most useful figure for mmap / page fault study.
    """
    roles = [
        "router_weight",
        "router_scale",
        "expert_gate_up",
        "expert_down",
        "expert_down_scale",
    ]

    colors = {
        "router_weight": "#F28E2B",
        "router_scale": "#FFBE7D",
        "expert_gate_up": "#E15759",
        "expert_down": "#B07AA1",
        "expert_down_scale": "#FF9DA7",
    }

    role_y = {role: i for i, role in enumerate(roles)}

    moe_records = [r for r in records if r["role"] in role_y]

    fig, ax = plt.subplots(figsize=(22, 6))

    for r in moe_records:
        role = r["role"]
        y = role_y[role]

        start_gib = bytes_to_gib(r["offset"])
        width_gib = bytes_to_gib(r["size_bytes"])

        ax.broken_barh(
            [(start_gib, width_gib)],
            (y - 0.35, 0.7),
            facecolors=colors[role],
            edgecolors="none",
            alpha=0.95,
        )

    # Mark layer boundaries for MoE tensors
    layer_min_offset = {}
    for r in moe_records:
        layer = r["layer"]
        if layer is None:
            continue
        layer_min_offset[layer] = min(layer_min_offset.get(layer, r["offset"]), r["offset"])

    for layer, offset in sorted(layer_min_offset.items()):
        x = bytes_to_gib(offset)
        ax.axvline(x, color="black", linewidth=0.35, alpha=0.25)

        if layer % 5 == 0:
            ax.text(
                x,
                len(roles) - 0.15,
                f"blk.{layer}",
                rotation=90,
                fontsize=8,
                va="bottom",
                ha="center",
            )

    ax.set_yticks(range(len(roles)))
    ax.set_yticklabels(roles)

    ax.set_xlabel("File offset (GiB)")
    ax.set_title("MoE-focused GGUF Layout: Router and Expert Tensors")

    ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.3)

    plt.tight_layout()

    path = os.path.join(out_dir, "04_moe_offset_focus.png")
    plt.savefig(path, dpi=200)
    plt.close(fig)

    return path


# ============================================================
# 8. Text summary
# ============================================================

def write_text_summary(records, out_dir):
    path = os.path.join(out_dir, "summary.txt")

    total_bytes = sum(r["size_bytes"] for r in records)

    by_family = defaultdict(int)
    by_role = defaultdict(int)
    by_layer = defaultdict(int)

    for r in records:
        by_family[r["family"]] += r["size_bytes"]
        by_role[r["role"]] += r["size_bytes"]
        if r["layer"] is not None:
            by_layer[r["layer"]] += r["size_bytes"]

    layers = sorted(by_layer.keys())

    with open(path, "w", encoding="utf-8") as f:
        f.write("GGUF Tensor Layout Summary\n")
        f.write("==========================\n\n")

        f.write(f"Total tensors: {len(records)}\n")
        f.write(f"Total tensor bytes: {bytes_to_gib(total_bytes):.3f} GiB\n")

        if layers:
            f.write(f"Transformer layers: {min(layers)} ~ {max(layers)}\n")
            f.write(f"Number of layers: {len(layers)}\n")

        f.write("\nBy family:\n")
        for fam, b in sorted(by_family.items(), key=lambda x: x[1], reverse=True):
            pct = b / total_bytes * 100 if total_bytes else 0
            f.write(f"  {fam:14s} {bytes_to_gib(b):8.3f} GiB  {pct:6.2f}%\n")

        f.write("\nTop roles:\n")
        for role, b in sorted(by_role.items(), key=lambda x: x[1], reverse=True)[:20]:
            pct = b / total_bytes * 100 if total_bytes else 0
            f.write(f"  {role:20s} {bytes_to_gib(b):8.3f} GiB  {pct:6.2f}%\n")

        f.write("\nResearch interpretation:\n")
        f.write("- moe_expert tensors are the main target for hot/cold page analysis.\n")
        f.write("- moe_router tensors are expected to be always-hot because routing is needed every token.\n")
        f.write("- attention and shared_ffn tensors are useful hot baselines.\n")
        f.write("- norm_scale tensors are usually smaller and less important for page-fault dominance.\n")

    return path


# ============================================================
# 9. Main
# ============================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_gguf_layout.py <model.gguf> [out_dir]")
        sys.exit(1)

    model_path = sys.argv[1]

    if len(sys.argv) >= 3:
        out_dir = sys.argv[2]
    else:
        out_dir = "gguf_layout_report"

    os.makedirs(out_dir, exist_ok=True)

    print(f"[INFO] Reading GGUF: {model_path}")
    records = read_tensor_records(model_path)

    print(f"[INFO] Number of tensors: {len(records)}")
    print(f"[INFO] Writing report to: {out_dir}")

    outputs = []

    outputs.append(write_detail_csv(records, out_dir))
    outputs.append(write_layer_family_csv(records, out_dir))
    outputs.append(write_top_tensors_csv(records, out_dir))
    outputs.append(write_text_summary(records, out_dir))

    outputs.append(plot_layer_family_heatmap(records, out_dir))
    outputs.append(plot_layer_stacked_bar(records, out_dir))
    outputs.append(plot_offset_lanes(records, out_dir))
    outputs.append(plot_moe_offset_focus(records, out_dir))

    print("\nGenerated files:")
    for p in outputs:
        print(f"  {p}")


if __name__ == "__main__":
    main()