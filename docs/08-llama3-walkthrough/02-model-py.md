---
title: model.py kiến trúc Llama-3
---

# `model.py`, kiến trúc Llama-3

File `model.py` định nghĩa kiến trúc Transformer Llama-3 thuần PyTorch, không có bất kỳ phần TP/FSDP nào. Đây là một trong những điểm thiết kế tốt: kiến trúc và parallelism tách rời, mỗi file một việc.

## `ModelArgs` dataclass

```python
@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 10000
    max_batch_size: int = 32
    max_seq_len: int = 2048
    depth_init: bool = True
```

Đây là cấu hình một model Llama:

- `dim`: model dim $K$. Llama-3 8B dùng $4096$.
- `n_layers`: số Transformer block. 32 cho 8B, 80 cho 70B.
- `n_heads`: số Q head. Thường 32 cho 8B.
- `n_kv_heads`: số K/V head cho GQA. 8 cho Llama-3 8B.
- `vocab_size`: kích thước vocab. -1 mặc định, được tokenizer điền vào.
- `multiple_of`, `ffn_dim_multiplier`: dùng để tính FFN hidden dim sao cho là bội số đẹp.
- `norm_eps`: $\epsilon$ cho RMSNorm.
- `rope_theta`: base frequency cho RoPE.
- `max_batch_size`, `max_seq_len`: giới hạn shape của precomputed buffers.
- `depth_init`: heuristic khởi tạo weight theo độ sâu layer.

## RMSNorm

```python
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
```

Implementation chuẩn: normalize theo chiều cuối, nhân với scale weight.

Quan trọng: `_norm` được tính trên `x.float()` rồi cast lại `type_as(x)`. Đây là trick để giữ độ chính xác fp32 trong norm dù model chạy bf16. Norm rất nhạy với numerical precision.

Với TP: `weight` là vector $K$ chiều, replicate trên các rank. Khi áp `SequenceParallel()`, PyTorch sẽ xử lý placement tự động.

## RoPE (Rotary Positional Embedding)

```python
def precompute_freqs_cis(dim, end, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    return torch.polar(torch.ones_like(freqs), freqs)
```

`precompute_freqs_cis` tính trước tensor complex $\mathbb{C}^{S_{max} \times d_k/2}$ chứa $e^{i \theta_{s, j}}$ với $\theta_{s, j} = s / \theta^{2j/d_k}$.

Đây là buffer, không phải parameter. Trong forward, tensor này được multiply phức vào Q và K:

```python
def apply_rotary_emb(xq, xk, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)
```

Trick: dùng complex multiplication để áp xoay. Mỗi cặp dim $(2i, 2i+1)$ của Q được nhân với $\cos \theta + i \sin \theta$.

Với TP: rotary chỉ chạm chiều `head_dim` của mỗi head, không trộn head. Hoàn toàn commute với head shard. PyTorch không cần xử lý đặc biệt.

## `Attention` class

```python
class Attention(nn.Module):
    def __init__(self, model_args):
        super().__init__()
        self.n_heads = model_args.n_heads
        self.n_kv_heads = ... # default to n_heads nếu không có
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = model_args.dim // model_args.n_heads

        self.wq = nn.Linear(model_args.dim, model_args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(model_args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(model_args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(model_args.n_heads * self.head_dim, model_args.dim, bias=False)
```

Bốn linear, không bias. `wq` output dim $n_{heads} \cdot d_k$, `wk, wv` output dim $n_{kv\_heads} \cdot d_k$. Đây là chuẩn GQA.

`forward` đã walkthrough ở Phần 4. Lưu ý là `Attention` không biết gì về TP. Mọi sự thông minh nằm trong `parallelism.py`.

## `FeedForward` class

```python
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, multiple_of, ffn_dim_multiplier):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
```

SwiGLU FFN với hidden dim được tính kỹ:

- Bắt đầu với `hidden_dim` truyền vào (thường $4 K$).
- Cắt còn $2/3$ (vì SwiGLU có 3 ma trận thay vì 2, để giữ tổng tham số tương đương).
- Nhân với `ffn_dim_multiplier` nếu có (Llama-3 dùng $\sim 1.3$ để tăng quality).
- Round lên multiple of `multiple_of` (256) để align với CUDA kernel.

Forward đúng công thức SwiGLU đã derive ở Phần 3.

## `TransformerBlock` class

```python
class TransformerBlock(nn.Module):
    def __init__(self, layer_id, model_args):
        super().__init__()
        self.attention = Attention(model_args)
        self.feed_forward = FeedForward(
            dim=model_args.dim,
            hidden_dim=4 * model_args.dim,
            multiple_of=model_args.multiple_of,
            ffn_dim_multiplier=model_args.ffn_dim_multiplier,
        )
        self.attention_norm = RMSNorm(dim=model_args.dim, eps=model_args.norm_eps)
        self.ffn_norm = RMSNorm(dim=model_args.dim, eps=model_args.norm_eps)

    def forward(self, x, freqs_cis):
        h = x + self.attention(self.attention_norm(x), freqs_cis)
        return h + self.feed_forward(self.ffn_norm(h))
```

Đây là Pre-Norm block (norm trước attention/FFN, residual sau). Cấu trúc cổ điển của Llama.

Lưu ý mẫu: `attention_norm` trước attention, `ffn_norm` trước feed_forward. Trong plan TP, hai norm này được wrap với `SequenceParallel()`, hai sub-module `attention` và `feed_forward` được wrap với `PrepareModuleInput`.

## `Transformer` class

```python
class Transformer(nn.Module):
    def __init__(self, model_args):
        super().__init__()
        self.tok_embeddings = nn.Embedding(model_args.vocab_size, model_args.dim)
        self.register_buffer("freqs_cis", self._precompute_freqs_cis(), persistent=True)
        self.layers = torch.nn.ModuleDict()
        for layer_id in range(model_args.n_layers):
            self.layers[str(layer_id)] = TransformerBlock(layer_id, model_args)
        self.norm = RMSNorm(dim=model_args.dim, eps=model_args.norm_eps)
        self.output = nn.Linear(model_args.dim, model_args.vocab_size, bias=False)

    def forward(self, tokens):
        h = self.tok_embeddings(tokens)
        for layer in self.layers.values():
            h = layer(h, self.freqs_cis)
        h = self.norm(h)
        return self.output(h).float()
```

Toàn bộ kiến trúc Llama-3, gọn trong một class.

Chú ý: `self.layers` là `ModuleDict` chứ không phải `ModuleList`. Lý do: keys là string `"0", "1", ...`, dễ map khi áp parallelism (đặc biệt pipeline). Forward duyệt `.values()` để giữ thứ tự.

`return self.output(h).float()` cast logits sang fp32 ở cuối. Quan trọng cho `loss_parallel` chính xác.

`init_weights` (đã có nhưng không hiển thị): khởi tạo Gaussian với std phụ thuộc layer depth. Áp dụng heuristic của paper Llama 3.

Chương sau ta sang `parallelism.py`.
