import torch
import transformers
import transformers.models.llama.modeling_llama
from einops import rearrange


class XposRotaryEmbedding(torch.nn.Module):
    def __init__(
        self,
        dim,
        max_position_embeddings=4096,
        base=10000,
        device=None,
        scale_base=4096,
        use_xpos=True,
    ):
        super().__init__()
        print("Using XPos")
        self.max_seq_len_cached = max_position_embeddings
        self.scale_base = scale_base

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(self.max_seq_len_cached, device=device).type_as(inv_freq)
        freqs = torch.einsum("i , j -> i j", t, inv_freq)
        freqs = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer("inv_freq", inv_freq, persistent=False)

        if not use_xpos:
            self.register_buffer("cos_cached", freqs.cos().to(device=t.device))
            self.register_buffer("sin_cached", freqs.sin().to(device=t.device))
            self.register_buffer("k_cos_cached", freqs.cos().to(device=t.device))
            self.register_buffer("k_sin_cached", freqs.sin().to(device=t.device))
            self.register_buffer("scale", None)
            self.register_buffer("scale_cached", torch.ones(1))
            return

        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        power = (t - (self.max_seq_len_cached // 2)) / self.scale_base
        scale_cached = scale ** rearrange(power, "n -> n 1")
        print(scale_cached)
        scale_cached = torch.cat((scale_cached, scale_cached), dim=-1)

        self.register_buffer("scale", scale, persistent=False)
        self.register_buffer("scale_cached", scale_cached, persistent=False)
        q_cos_cached = freqs.cos() * scale_cached
        q_sin_cached = freqs.sin() * scale_cached
        k_cos_cached = freqs.cos() * (scale_cached**-1)
        k_sin_cached = freqs.sin() * (scale_cached**-1)
        self.register_buffer("q_cos_cached", q_cos_cached.to(device=t.device))
        self.register_buffer("q_sin_cached", q_sin_cached.to(device=t.device))
        self.register_buffer("k_cos_cached", k_cos_cached.to(device=t.device))
        self.register_buffer("k_sin_cached", k_sin_cached.to(device=t.device))

    def forward(
        self,
        x,
        seq_len,
    ):
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=x.device).type_as(
                self.inv_freq
            )
            freqs = torch.einsum("i , j -> i j", t, self.inv_freq)
            freqs = torch.cat((freqs, freqs), dim=-1).to(dtype=x.dtype)

            self.register_buffer("freqs_cached", freqs)

            if self.scale is None:
                self.register_buffer(
                    "scale_cached", torch.ones(1, device=x.device).to(dtype=x.dtype)
                )
                self.register_buffer("cos_cached", freqs.cos().to(dtype=x.dtype))
                self.register_buffer("sin_cached", freqs.sin().to(dtype=x.dtype))
                self.register_buffer("k_cos_cached", freqs.cos().to(dtype=x.dtype))
                self.register_buffer("k_sin_cached", freqs.sin().to(dtype=x.dtype))
                return self.freqs_cached.to(dtype=x.dtype), self.scale_cached

            power = (t - (seq_len // 2)) / self.scale_base
            scale = self.scale ** rearrange(power, "n -> n 1")
            scale = torch.cat((scale, scale), dim=-1).to(dtype=x.dtype)
            self.register_buffer("scale_cached", scale)
            q_cos_cached = freqs.cos() * scale
            q_sin_cached = freqs.sin() * scale
            k_cos_cached = freqs.cos() * (scale**-1)
            k_sin_cached = freqs.sin() * (scale**-1)
            self.register_buffer("q_cos_cached", q_cos_cached.to(dtype=x.dtype))
            self.register_buffer("q_sin_cached", q_sin_cached.to(dtype=x.dtype))
            self.register_buffer("k_cos_cached", k_cos_cached.to(dtype=x.dtype))
            self.register_buffer("k_sin_cached", k_sin_cached.to(dtype=x.dtype))

        return (self.q_cos_cached, self.q_sin_cached), (
            self.k_cos_cached,
            self.k_sin_cached,
        )


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, q_freqs, k_freqs, position_ids=None):
    q_cos, q_sin = q_freqs
    q_cos = q_cos[position_ids, :]
    q_sin = q_sin[position_ids, :]

    k_cos, k_sin = k_freqs
    k_cos = k_cos[position_ids, :]
    k_sin = k_sin[position_ids, :]

    q_embed = (q * q_cos) + (rotate_half(q) * q_sin)
    k_embed = (k * k_cos) + (rotate_half(k) * k_sin)

    return q_embed, k_embed


def replace_llama_rope_with_xpos_rope():
    transformers.models.llama.modeling_llama.LlamaRotaryEmbedding = XposRotaryEmbedding
    transformers.models.llama.modeling_llama.apply_rotary_pos_emb = apply_rotary_pos_emb
