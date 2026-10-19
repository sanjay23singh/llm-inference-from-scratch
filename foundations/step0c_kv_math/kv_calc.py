"""Step 0c: KV cache calculator. Your formula, applied to real models.

KV bytes = 2 * n_layers * kv_heads * head_dim * bytes * tokens * batch
"""
from dataclasses import dataclass


@dataclass
class Model:
    name: str
    params_b: float   # billions of parameters
    layers: int
    kv_heads: int
    head_dim: int
    bytes: int        # bytes per number (fp32=4, fp16=2)

    def kv_per_token(self):                      # bytes
        return 2 * self.layers * self.kv_heads * self.head_dim * self.bytes

    def kv_total(self, tokens, batch=1):         # bytes
        return self.kv_per_token() * tokens * batch

    def weights_gb(self):
        return self.params_b * self.bytes        # params (billions) * bytes = GB


MODELS = [
    Model("GPT-2 small (MHA)", 0.124, 12, 12, 64, 4),
    Model("Llama-2-7B (MHA)", 7, 32, 32, 128, 2),
    Model("Llama-3-8B (GQA)", 8, 32, 8, 128, 2),
    Model("Llama-3-70B (GQA)", 70, 80, 8, 128, 2),
]


def max_users(m, gpu_gb, tokens):
    """How many users fit: (GPU memory - weights) / KV per user. Ignores activations and overhead."""
    free_gb = gpu_gb - m.weights_gb()
    return max(0, int(free_gb * 1e9 // m.kv_total(tokens)))


if __name__ == "__main__":
    TOKENS, GPU_GB = 8192, 80
    print(f"context = {TOKENS} tokens, GPU = {GPU_GB} GB (Llama-3-70B: 2 GPUs = {2 * GPU_GB} GB)\n")
    print(f"{'model':20s} {'KB/token':>9s} {'GB/user':>8s} {'weights GB':>11s} {'max users':>10s}")
    for m in MODELS:
        gpu = GPU_GB * 2 if m.params_b > 40 else GPU_GB
        print(f"{m.name:20s} {m.kv_per_token() / 1024:9.0f} {m.kv_total(TOKENS) / 1e9:8.2f} "
              f"{m.weights_gb():11.1f} {max_users(m, gpu, TOKENS):10d}")
