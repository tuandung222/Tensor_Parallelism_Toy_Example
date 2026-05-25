---
title: SequenceParallel cho Norm
---

# `SequenceParallel()` cho RMSNorm và LayerNorm

Trái tim của SP là khả năng tính normalization khi chiều sequence đang shard. Chương này giải thích vì sao RMSNorm và LayerNorm "tự nhiên" tương thích với SP, và cài đặt SequenceParallel hoạt động ra sao.

## Norm là phép per-position

RMSNorm và LayerNorm áp dụng trên chiều cuối (hidden dim $K$). Với tensor input shape $(B, S, K)$:

$$
\mathrm{RMSNorm}(x)_{b,s,:} = \frac{x_{b,s,:}}{\sqrt{\frac{1}{K} \sum_k x_{b,s,k}^2 + \epsilon}} \cdot \gamma
$$

Chú ý ký hiệu: phép normalize chỉ chạm vào chiều $K$ tại từng position $(b, s)$. Hai position khác nhau hoàn toàn độc lập. Không có sum, mean, hoặc reduce gì trên chiều $S$.

Vì vậy nếu ta shard chiều $S$ giữa các rank, mỗi rank giữ một subset position, mỗi rank có thể tính norm cho các position của mình mà không cần giao tiếp với rank khác. **Không collective**.

Đây là điểm cốt lõi: **norm commute với shard sequence**.

## `SequenceParallel()` style

PyTorch cài đặt SP cho norm thông qua một style:

```python
from torch.distributed.tensor.parallel import SequenceParallel

plan = {
    "attention_norm": SequenceParallel(),
    "ffn_norm": SequenceParallel(),
    "norm": SequenceParallel(),  # final norm
}
parallelize_module(model, tp_mesh, plan)
```

`SequenceParallel()` style làm các việc sau cho một module norm:

1. Shard `weight` (parameter $\gamma$ của norm) theo chiều cuối. Vì $\gamma \in \mathbb{R}^K$ là vector 1D, shard nó tương đương với replicate (PyTorch tự xử lý). Trong thực tế, $\gamma$ replicate, nhỏ nên không vấn đề.
2. Khẳng định input của norm phải có placement `Shard(1)` (sequence shard).
3. Output của norm cũng là `Shard(1)`.

Tóm lại, norm trở thành "pass-through" về sequence shard: input shard, output shard, không collective.

## So sánh với norm không SP

Nếu ta không dùng SP cho norm, norm sẽ hoạt động trên tensor replicate. Tensor $(B, S, K)$ replicate chiếm $B \cdot S \cdot K \cdot 4$ bytes mỗi rank. Với $S = 8K$, $K = 4096, B = 1$: 128 MB chỉ riêng norm input mỗi rank.

Với SP, tensor $(B, S/P, K)$ chiếm $B \cdot S \cdot K \cdot 4 / P$ bytes. Tiết kiệm $P$ lần. Đây chính là phần activation tiết kiệm chính của SP.

## Đảm bảo zone nhất quán

SP chỉ hoạt động khi cấu trúc zone nhất quán:

- Tensor đi vào norm phải `Shard(1)`.
- Tensor đi ra norm vẫn `Shard(1)`.

Nếu tensor đầu vào không đúng placement, PyTorch sẽ trả lỗi hoặc chèn collective bù không mong muốn. Vì vậy ta cần điều phối placement cẩn thận ở các "biên giới" zone, ví dụ trước attention (đầu vào sequence shard, đầu ra cần replicate cho softmax) và sau attention (output cần shard lại sequence cho residual).

Đây là vai trò của `PrepareModuleInput`, chương tiếp sẽ làm rõ.

## Norm khác có dùng được SP không

LayerNorm: cũng per-position trên chiều cuối, tương tự RMSNorm. Áp dụng `SequenceParallel()` được, không vấn đề.

GroupNorm: per-group trên chiều cuối, không trộn position. Cũng SP được.

BatchNorm: reduce trên chiều batch (và spatial cho image), không trộn sequence. SP cũng được, nhưng nếu batch cũng được shard (DP), thì BatchNorm cần SyncBatchNorm với AllReduce trên DP mesh, không phải TP mesh. Phức tạp hơn.

Trong LLM, ta gần như chỉ gặp RMSNorm và LayerNorm, cả hai đều SP-friendly.

## Một lưu ý về dropout

Dropout là phép element-wise per-position khi áp với mask ngẫu nhiên. Bản thân dropout commute với shard sequence. Nhưng RNG (random number generator) cần được đồng bộ giữa các rank trong TP group để các position trong cùng một tensor không bị drop với mask khác nhau theo cách không nhất quán.

PyTorch có cơ chế `RNGTracker` để đồng bộ. Trong toy code Llama-3, dropout không được dùng. Trong các Transformer khác, bạn cần lưu ý.

## Tóm tắt

| Module | Style | Collective | Lý do |
|--------|-------|-----------|--------|
| `attention_norm` (RMSNorm trước attention) | `SequenceParallel()` | Không | Per-position, không trộn S |
| `ffn_norm` (RMSNorm trước FFN) | `SequenceParallel()` | Không | Tương tự |
| `norm` (RMSNorm cuối model) | `SequenceParallel()` | Không | Tương tự |

Norm giữ vai trò "đệm" giữa các zone, là điểm tự nhiên để SP zone tồn tại. Chương tiếp ta sang transition (AllGather, ReduceScatter) khi đi từ SP zone vào TP zone và ngược lại.
