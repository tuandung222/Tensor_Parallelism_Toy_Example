---
title: Walkthrough Attention.forward
---

# Walkthrough `Attention.forward`

Đọc kỹ `Attention.forward` trong `02_large_language_model/model.py`, đối chiếu với plan TP để xem mỗi tensor đang ở placement nào.

## Code đầy đủ

```python
def forward(self, x, freqs_cis):
    bs, seqlen, _ = x.shape
    xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

    xq = xq.view(bs, seqlen, self.n_heads, self.head_dim)
    xk = xk.view(bs, seqlen, self.n_kv_heads, self.head_dim)
    xv = xv.view(bs, seqlen, self.n_kv_heads, self.head_dim)

    xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

    keys = repeat_kv(xk, self.n_rep)
    values = repeat_kv(xv, self.n_rep)

    xq = xq.transpose(1, 2)
    xk = keys.transpose(1, 2)
    xv = values.transpose(1, 2)

    output = F.scaled_dot_product_attention(xq, xk, xv, is_causal=True)
    output = output.transpose(1, 2).contiguous()
    output = output.view(bs, seqlen, -1)
    return self.wo(output)
```

## Đầu vào và QKV projection

Trước khi vào `forward`, `PrepareModuleInput` đã biến $x$ thành DTensor replicate trên TP mesh, local shape $(B, S, K)$. Sau ba dòng `self.wq, wk, wv`, các tensor là DTensor `Shard(-1)`, local shape lần lượt $(B, S, n_{heads}^{local} \cdot d_k)$ và $(B, S, n_{kv\_heads}^{local} \cdot d_k)$.

Vì `parallelism.py` đã chỉnh `self.n_heads //= P` và `self.n_kv_heads //= P`, các attribute này khớp đúng với kích thước local.

## View thành dạng head

Ba dòng `view` áp dụng trên tensor local. Sau view, `xq` có shape $(B, S, n_{heads}^{local}, d_k)$, shard trên chiều index 2 (chiều head). Chiều `head_dim` ($d_k$) ở vị trí cuối nguyên vẹn, không bị shard. Đây chính là điểm cốt lõi của TP cho attention: shard theo head, không bao giờ shard theo head_dim.

## Rotary embedding

`apply_rotary_emb` là chuỗi view + complex multiply + view ngược. Tất cả là element-wise theo cặp `(2i, 2i+1)` trong chiều `head_dim`, không trộn các head. `freqs_cis` là buffer global (replicate). Sau dòng này, `xq, xk` vẫn shard theo head. Không collective.

## `repeat_kv`

Hàm này `expand` chiều `n_kv_heads` lên $n_{rep}$ lần để khớp với $n_{heads}$. Vì cả hai cùng chia $P$, $n_{kv\_heads}^{local} \cdot n_{rep} = n_{heads}^{local}$. `expand` không sao chép memory thực, chỉ thay stride. Sau dòng này, `keys, values` shape local $(B, S, n_{heads}^{local}, d_k)$, khớp với `xq`.

## Transpose và scaled dot product attention

Ba dòng `transpose(1, 2)` đảo chiều `seqlen` và chiều `head` để khớp với convention của `F.scaled_dot_product_attention`: `(B, n_heads, S, d_k)`.

`scaled_dot_product_attention` tính:

$$
\mathrm{output} = \mathrm{softmax}\!\left( \frac{Q K^\top}{\sqrt{d_k}} + M \right) V
$$

với $M$ là causal mask. Phép này áp trên trục batch của chiều head, mỗi head độc lập. Vì các head đã shard, mỗi rank tính các head của mình mà không cần collective. PyTorch coi đây như batched matmul nội bộ, không quan tâm rằng các head đang shard ở mức DTensor.

Output shape local: $(B, n_{heads}^{local}, S, d_k)$.

## Transpose ngược và view

Hai dòng cuối transpose lại để đưa chiều seqlen về vị trí 1, rồi `view` gộp `(n_heads^local, d_k)` thành một chiều cuối duy nhất. Output local: $(B, S, n_{heads}^{local} \cdot d_k)$. Đây vẫn là DTensor `Shard(-1)`.

## `self.wo(output)`

Đây là Row Parallel. Input shard cuối khớp đúng, mỗi rank tính local matmul:

$$
Y^{(r)} = O^{(r)} W_O^{(r)} \in \mathbb{R}^{B \times S \times K}
$$

Tổng $\sum_r Y^{(r)}$ cho ra $Y$ đầy đủ (partial sum trên TP mesh). Cuối cùng, PyTorch chèn AllReduce (hoặc ReduceScatter nếu Sequence Parallel) để biến partial thành output cuối.

## Tổng kết placement trên một forward

| Vị trí | Placement | Local shape |
|--------|-----------|--------------|
| Sau `PrepareModuleInput` | Replicate | $(B, S, K)$ |
| Sau `wq` | Shard cuối | $(B, S, n_h^{loc} d_k)$ |
| Sau view + rotary + transpose | Shard chiều head | $(B, n_h^{loc}, S, d_k)$ |
| Sau `scaled_dot_product_attention` | Shard chiều head | $(B, n_h^{loc}, S, d_k)$ |
| Sau view ngược | Shard cuối | $(B, S, n_h^{loc} d_k)$ |
| Sau `wo` (local matmul) | Partial | $(B, S, K)$ |
| Sau AllReduce | Replicate (hoặc Shard(1) nếu SP) | $(B, S, K)$ |

Một forward = một collective duy nhất. Backward đối ngẫu, cũng một collective. Tổng chi phí giao tiếp mỗi block attention mỗi step: hai AllReduce trên tensor $(B, S, K)$. Khớp đúng với lý thuyết.

Phần 4 kết thúc tại đây. Phần 5 sẽ chuyển sang Embedding và LM head, hai khối còn lại của Transformer LLM.
