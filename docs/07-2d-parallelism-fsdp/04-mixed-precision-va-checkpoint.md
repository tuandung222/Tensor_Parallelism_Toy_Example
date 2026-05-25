---
title: Mixed precision và activation checkpoint
---

# Mixed precision và activation checkpoint

Hai kỹ thuật bổ trợ thường đi cùng FSDP: mixed precision (bf16 forward + fp32 master weight) và activation checkpoint (đánh đổi compute lấy memory). Chương này trình bày cả hai trong context 2D parallelism.

## Mixed precision motivation

Trên A100, H100, B100, hiệu suất bf16 cao hơn fp32 khoảng $2-4$ lần. Bộ nhớ cũng giảm $2$ lần. Train LLM với bf16 là chuẩn de facto.

Nhưng: nếu cộng dồn nhỏ trong update (gradient $\times$ learning rate nhỏ) trong bf16, sai số tích lũy đáng kể. Adam moment cũng có vấn đề tương tự. Giải pháp: giữ một **master weight** ở fp32, chỉ dùng bf16 cho forward compute và một số khâu khác.

## Mixed precision policy với FSDP

PyTorch FSDP có `MixedPrecisionPolicy`:

```python
from torch.distributed._composable.fsdp import MixedPrecisionPolicy

mp_policy = MixedPrecisionPolicy(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,
)
fully_shard(block, mesh=dp_mesh, mp_policy=mp_policy)
```

Hai option:

- `param_dtype`: dtype của parameter khi forward/backward. Nếu bf16, FSDP cast parameter sang bf16 sau khi AllGather, dùng cho compute.
- `reduce_dtype`: dtype khi ReduceScatter gradient. fp32 cho phép cộng gradient với độ chính xác cao trước khi cast lại bf16 để lưu.

Master weight (shard) thực ra vẫn giữ ở dtype gốc của parameter khi khai báo (thường là fp32 nếu khởi tạo bằng `nn.Parameter(...)` mặc định, hoặc bf16 nếu khởi tạo bf16). Có hai paradigm:

**Paradigm A**: master weight fp32, compute bf16.

- Init parameter bf16 hoặc fp32, sau đó để FSDP lưu ở fp32.
- AllGather bf16, compute bf16, gradient bf16 hoặc fp32.
- ReduceScatter fp32 (an toàn).
- Optimizer update master fp32.

**Paradigm B**: master weight bf16 (toàn phần bf16).

- Khởi tạo bf16, lưu bf16.
- Compute bf16, gradient bf16.
- Đơn giản hơn, ít memory hơn, nhưng kém ổn định.

Trong toy code:

```python
mp_policy = MixedPrecisionPolicy(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,
)
```

Đây là Paradigm A. Đúng cho LLM training scale lớn.

## Activation checkpoint

Activation memory là một trong những bottleneck chính (Phần 6). Activation checkpoint là kỹ thuật: thay vì giữ mọi tensor trung gian cho backward, ta chỉ giữ một số "checkpoint", các tensor khác sẽ được **tính lại** trong backward.

Cụ thể, với một segment có L layer:

- Forward: tính qua L layer, **chỉ giữ activation đầu vào segment**.
- Backward đến segment này: re-compute forward để có activation, rồi backward.

Cost: 1 lần forward thêm cho mỗi segment = compute tăng $\sim 33\%$ (1 forward + 1 backward + 1 recompute, so với 1 forward + 1 backward bình thường, tăng 50% so với 2 work, thực tế ~33% so với toàn bộ).

Benefit: activation memory giảm. Nếu chia model thành $k$ segment đều nhau, activation memory giảm $L/k$ lần.

## `checkpoint_wrapper` cho FSDP

```python
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

for layer_id, transformer_block in model.layers.items():
    transformer_block = checkpoint_wrapper(transformer_block)
    fully_shard(transformer_block, mesh=dp_mesh)
    model.layers[layer_id] = transformer_block
```

`checkpoint_wrapper` wrap module, trong forward nó "ghi nhớ" input và output, không lưu intermediate. Trong backward, nó re-compute forward.

Trong toy code, mỗi Transformer block được wrap. Vậy mỗi block là một segment. Activation memory cho intermediate (Q, K, V, scores, hidden FFN) bên trong block không còn lưu giữ giữa các block. Chỉ lưu activation đầu vào mỗi block (residual).

## Tính toán bộ nhớ với checkpoint

Không checkpoint, activation memory $\sim L \cdot (\text{activation per block})$.

Với checkpoint mỗi block, activation memory $\sim L \cdot (\text{activation đầu vào block}) + 1 \cdot (\text{activation per block trong recompute})$.

Term thứ nhất là $L \cdot B S K$ (residual). Term thứ hai là $\text{full block}$, chỉ một lần (vì recompute sequential).

Tổng giảm từ $L \cdot \text{full block}$ xuống $L \cdot B S K + \text{full block}$. Với LLM, residual nhỏ hơn full block, giảm đáng kể.

Kết hợp với SP (giảm residual thêm $P$ lần):

$$
\text{Activation} \approx \frac{L \cdot B S K}{P_{TP}} + \text{full block local}
$$

Cấu hình production: TP+SP+FSDP+Checkpoint. Đây là setup chuẩn cho Llama-3 8B/70B training.

## Trade-off: checkpoint mọi block hay chỉ một số

Checkpoint mọi block: tăng compute $\sim 33\%$, giảm activation tối đa.

Selective checkpoint: chỉ checkpoint một số block (ví dụ mỗi 2 block một lần). Tăng compute $\sim 16\%$, giảm activation $\sim 50\%$. Tốt nếu activation chưa quá lớn.

Trong toy code, mọi block đều được checkpoint. Hợp lý cho LLM 8B với context dài.

## Selective activation checkpointing nâng cao

Một kỹ thuật mới: chỉ checkpoint một số activation đắt (như scores), giữ activation rẻ (như Q, K, V). Giảm compute tăng so với full checkpoint mà vẫn tiết kiệm bộ nhớ. PyTorch hỗ trợ via `SAC` (Selective Activation Checkpoint) API mới. Toy code chưa dùng.

## Tóm tắt phối hợp

| Kỹ thuật | Mục đích | Cost |
|----------|-----------|------|
| TP | Shard param + một số activation trên TP mesh | AllReduce/AllGather/ReduceScatter trên TP |
| SP | Shard residual activation theo sequence | AllGather + ReduceScatter trên TP |
| FSDP | Shard param + grad + opt state trên DP mesh | AllGather + ReduceScatter trên DP |
| Mixed precision (bf16) | Giảm bộ nhớ + tăng compute | Cast cost, độ chính xác giảm |
| Activation checkpoint | Re-compute để giảm activation | Compute tăng $\sim 33\%$ |

Bốn kỹ thuật này compose tốt nhau và đều có trong toy code `parallelism.py`. Chương cuối Phần 7 sẽ walkthrough đoạn code FSDP cụ thể.
