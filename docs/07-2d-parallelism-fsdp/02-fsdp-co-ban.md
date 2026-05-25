---
title: FSDP cơ bản
---

# FSDP cơ bản

FSDP (Fully Sharded Data Parallel) là kỹ thuật shard parameter, gradient, và optimizer state theo các rank trong một "DP mesh". Đây là sự phát triển từ DeepSpeed ZeRO, được PyTorch port lại thành API native. Phần này giới thiệu FSDP từ đầu, giả sử bạn quen DDP nhưng chưa quen FSDP.

## DDP nhắc lại

Trong Distributed Data Parallel (DDP):

- Mỗi rank giữ **bản sao đầy đủ** của model parameter.
- Mỗi rank xử lý một subset khác nhau của batch.
- Cuối mỗi backward, AllReduce gradient để các bản sao đồng bộ.
- Optimizer chạy local trên mỗi rank, update parameter.

Vấn đề: parameter + gradient + optimizer state đều full trên mỗi rank. Với Llama-3 8B + AdamW fp32: 8 B params $\times$ (4 + 4 + 8) bytes = **160 GB** mỗi rank. Không vừa GPU 80 GB. DDP không train được model lớn.

## ZeRO/FSDP ý tưởng

Shard tất cả tham số theo $P_{DP}$ rank:

- Parameter: mỗi rank giữ $1/P_{DP}$ params.
- Gradient: mỗi rank giữ $1/P_{DP}$ gradient (cho phần param mình giữ).
- Optimizer state: mỗi rank giữ $1/P_{DP}$ optimizer state.

Tổng: 160 GB / $P_{DP}$. Với $P_{DP} = 8$: 20 GB. Vừa GPU.

Đổi lại: khi cần forward, ta phải tạm thời gom parameter đầy đủ về để chạy compute. Đây là cost của FSDP.

## Forward và Backward với FSDP

Forward, từng layer:

1. **AllGather** parameter của layer này từ mọi rank trong DP mesh để có parameter đầy đủ.
2. Tính forward bình thường.
3. **Free** parameter đầy đủ, chỉ giữ lại shard của rank này.

Backward, từng layer:

1. **AllGather** parameter (tương tự forward).
2. Tính gradient với parameter và activation.
3. **ReduceScatter** gradient: cộng gradient từ mọi rank, mỗi rank chỉ giữ shard tương ứng với phần param của mình.

Optimizer step: chạy local trên shard, vì mỗi rank đã có gradient và optimizer state tương ứng với shard của mình.

## So sánh chi phí với DDP

DDP: 1 AllReduce gradient mỗi backward = $\frac{2(P-1)}{P} \cdot |\text{params}|$ bytes.

FSDP: 1 AllGather param forward + 1 AllGather param backward + 1 ReduceScatter gradient = $3 \cdot \frac{P-1}{P} \cdot |\text{params}|$ bytes.

FSDP tốn **1.5 lần** giao tiếp so với DDP, nhưng đổi lại train được model lớn hơn $\sim P$ lần. Đây là một trade-off đáng giá.

## Đơn vị shard: theo layer

FSDP không shard cả model thành một blob, mà chia theo "unit". Mỗi unit là một module được wrap với `fully_shard`. Bên trong unit, parameter được shard. Giữa các unit, parameter được giữ shard (không gom).

Thông thường, unit = một Transformer block. Lý do: gom toàn bộ block một lần, làm forward, rồi free. Chi phí AllGather một block tương đối nhỏ, có thể overlap với compute của block trước. Nếu unit quá lớn (cả model), AllGather một lần quá lớn, không overlap được. Nếu unit quá nhỏ (mỗi linear một unit), overhead AllGather quá nhiều.

## `fully_shard` API

Cú pháp PyTorch FSDP2:

```python
from torch.distributed._composable.fsdp.fully_shard import fully_shard

for layer_id, transformer_block in model.layers.items():
    fully_shard(transformer_block, mesh=dp_mesh)
fully_shard(model, mesh=dp_mesh)
```

`fully_shard` modify in-place module, làm parameter của module trở thành DTensor shard trên DP mesh. Sau khi áp:

- `block.attention.wq.weight` trở thành DTensor shard dim 0 trên DP mesh.
- Forward của block tự động AllGather + compute + free.

Khác biệt với FSDP1 (`FullyShardedDataParallel`): FSDP2 dùng DTensor làm storage, composable với TP, gọn hơn.

## `reshard_after_forward`

Option này điều khiển hành vi sau forward của một unit:

- `True`: free parameter ngay sau forward, sẽ AllGather lại khi backward đến unit này.
- `False`: giữ parameter sau forward, không cần AllGather lại trong backward.

Trade-off: `True` tiết kiệm bộ nhớ (mất parameter sau forward, chỉ giữ shard), nhưng tốn thêm 1 AllGather mỗi backward. `False` tốn bộ nhớ (giữ parameter cho cả forward và backward), nhưng tiết kiệm 1 AllGather.

Thực tế: với LLM lớn, gần như luôn dùng `True` vì bộ nhớ là bottleneck. Một số layer cuối có thể dùng `False` để tránh AllGather ngay trước backward (vì PyTorch không có cách prefetch tốt cho block cuối). Đây là tối ưu nhỏ.

## Tổng kết FSDP

| Aspect | DDP | FSDP |
|--------|------|------|
| Parameter mỗi rank | full | $1/P$ |
| Gradient mỗi rank | full (sau AllReduce) | $1/P$ |
| Optimizer state mỗi rank | full | $1/P$ |
| Forward collective | 0 | 1 AllGather mỗi unit |
| Backward collective | 1 AllReduce param | 1 AllGather + 1 ReduceScatter mỗi unit |
| Tổng bytes chuyển | $2(P-1)/P \cdot \|params\|$ | $3(P-1)/P \cdot \|params\|$ |

FSDP rộng rãi hơn DDP về khả năng scale, vẫn giữ semantics data parallel. Phần tiếp ta xem cách kết hợp FSDP với TP trên mesh 2D.
