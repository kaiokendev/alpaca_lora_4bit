import torch
from transformers.models.llama.modeling_llama import LlamaConfig, LlamaRotaryEmbedding, apply_rotary_pos_emb

class xPosRotaryEmbedding(torch.nn.Module):
    """
    Modified LLaMa RoPE positional embeddings based on this code:
    https://github.com/lucidrains/x-transformers/blob/82ba6aa9526dd14a032a3b4e8bc2111b45c2249f/x_transformers/x_transformers.py#L416
    which in turn is based on this paper https://arxiv.org/abs/2212.10554v1 "A Length-Extrapolatable Transformer"
    
    The goal is to add a scale to the embeddings to allow RoPE to extrapolate positional embeddings beyond the original training dimensions
    """
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scale_base=512):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        
        self.dim = dim
        self.scale_base = scale_base
        self.register_buffer('scale', scale)
        self.register_buffer('freqs_cached', freqs)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if not torch.exists(self.scale):
                return freqs, 1.
            
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            scale = (torch.arange(0, self.dim, 2) + 0.4 * self.dim) / (1.4 * self.dim)
            
            self.register_buffer('scale', scale)
            self.register_buffer('freqs_cached', freqs)
            
        power = (torch.arange(seq_len, device = x.device) - (seq_len // 2)) / self.scale_base
        scale = self.scale ** torch.rearrange(power, 'n -> n 1')
        scale = torch.cat((scale, scale), dim = -1)
            
        return self.freqs.to(dtype=x.dtype), scale.to(dtype=x.dtype)
    
def replace_llama_rope_with_xpos_rope():
    transformers.models.llama.modeling_llama.LlamaRotaryEmbedding = xPosRotaryEmbedding