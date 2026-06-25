import sys
from gguf.gguf_reader import GGUFReader
import matplotlib.pyplot as plt

def categorize(name):
    n = name.lower()

    if n == "token_embd.weight":
        return "embedding"

    if n.startswith("per_layer_token_embd"):
        return "ple"

    if n.startswith("per_layer_model_proj"):
        return "ple_projection"

    if n.startswith("per_layer_proj_norm"):
        return "ple_norm"

    if n.startswith("rope_freqs"):
        return "rope"

    if n.startswith("output_norm"):
        return "output_norm"

    if ".attn_q.weight" in n or ".attn_k.weight" in n or ".attn_v.weight" in n:
        return "attention_core"

    if ".attn_output.weight" in n:
        return "attention_output"

    if ".attn_norm.weight" in n or ".attn_q_norm.weight" in n or ".attn_k_norm.weight" in n:
        return "attention_norm"

    if ".ffn_gate.weight" in n or ".ffn_up.weight" in n or ".ffn_down.weight" in n:
        return "ffn_core"

    if ".ffn_norm.weight" in n:
        return "ffn_norm"

    if ".post_attention_norm.weight" in n or ".post_ffw_norm.weight" in n or ".post_norm.weight" in n:
        return "post_norm"

    if ".inp_gate.weight" in n or ".layer_output_scale.weight" in n:
        return "control"

    if ".proj.weight" in n:
        return "projection"

    return "other"

COLOR_MAP = {
    # ===== Embedding / global =====
    "embedding": "#2ecc71",        # 綠
    "ple": "#27ae60",              # 深綠（per-layer embedding）
    "ple_projection": "#1abc9c",   # 青綠
    "ple_norm": "#16a085",         # 深青
    "rope": "#a3e4d7",             # 淺綠藍
    "output_norm": "#8e44ad",      # 紫

    # ===== Attention =====
    "attention_core": "#e74c3c",   # 紅（QKV）
    "attention_output": "#ff6b6b", # 淺紅
    "attention_norm": "#f5b7b1",   # 淡紅

    # ===== FFN =====
    "ffn_core": "#3498db",         # 藍
    "ffn_norm": "#85c1e9",         # 淡藍

    # ===== Norm / control / projection =====
    "post_norm": "#f39c12",        # 橘
    "control": "#d35400",          # 深橘（gate / scale）
    "projection": "#9b59b6",       # 紫藍（proj.weight）

    # ===== fallback =====
    "other": "#7f8c8d"             # 灰
}

def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_tensors.py <model.gguf>")
        sys.exit(1)

    path = sys.argv[1]
    reader = GGUFReader(path)

    fig, ax = plt.subplots(figsize=(24, 2))

    for t in reader.tensors:
        offset = t.data_offset
        size = t.n_bytes
        print(t.name)
        category = categorize(t.name)

        color = COLOR_MAP.get(category, "gray")

        ax.barh(
            y=0,
            width=size,
            left=offset,
            color=color,
            height=0.5
        )

    ax.set_xlabel("File Offset (bytes)")
    ax.set_yticks([])

    # legend
    handles = []    
    for k, v in COLOR_MAP.items():
        handles.append(plt.Rectangle((0,0),1,1,color=v,label=k))
    ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.title("GGUF Tensor Layout")
    plt.tight_layout()
    plt.savefig("tensor_layout.png", dpi=200)
    print("Saved to tensor_layout.png")


if __name__ == "__main__":
    main()

