---
title: Walkthrough FSDP trong parallelism.py
---

# Walkthrough FSDP trong `parallelism.py`

Đoạn FSDP cuối hàm `parallelize` trong `02_large_language_model/parallelism.py`:

```python
if dp_mesh.size() > 1:
    assert dp_mesh.ndim == 1

    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
    )

    fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}
    for layer_id, transformer_block in model.layers.items():
        transformer_block = checkpoint_wrapper(transformer_block)
        reshard_after_forward = int(layer_id) < len(model.layers) - 1
        fully_shard(
            transformer_block,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )
        model.layers[layer_id] = transformer_block
    model = fully_shard(model, **fsdp_config)
```

Đi từng dòng.

## Điều kiện áp FSDP

```python
if dp_mesh.size() > 1:
    assert dp_mesh.ndim == 1
```

FSDP chỉ áp khi có DP dimension thực tế (`dp_size > 1`). Nếu world chỉ có TP, ta không cần FSDP. Assertion `ndim == 1` đảm bảo `dp_mesh` là 1D (1 chiều), vì `fully_shard` hiện chỉ hỗ trợ 1D DP mesh.

Nếu bạn muốn 3D parallelism (DP $\times$ TP $\times$ PP chẳng hạn), bạn sẽ cần một chiều thứ ba cho pipeline. Toy code không lo việc đó.

## Mixed precision policy

```python
mp_policy = MixedPrecisionPolicy(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,
)
```

Như đã giải thích: parameter cast bf16 cho compute, gradient reduce ở fp32 để giữ độ chính xác. Master weight tự lưu ở dtype gốc (thường fp32).

## fsdp_config

```python
fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}
```

Đối tượng cấu hình dùng cho mọi `fully_shard` sau. `mesh=dp_mesh` quan trọng: nó nói FSDP shard trên DP mesh, không phải world mesh. Nếu quên dòng này, FSDP sẽ shard cả TP rank, xung đột.

## Wrap mỗi block

```python
for layer_id, transformer_block in model.layers.items():
    transformer_block = checkpoint_wrapper(transformer_block)
    reshard_after_forward = int(layer_id) < len(model.layers) - 1
    fully_shard(
        transformer_block,
        **fsdp_config,
        reshard_after_forward=reshard_after_forward,
    )
    model.layers[layer_id] = transformer_block
```

Bốn việc cho mỗi block:

1. `checkpoint_wrapper(transformer_block)`: wrap để activation checkpoint, lưu phép forward để re-compute trong backward.
2. `reshard_after_forward = int(layer_id) < len(model.layers) - 1`: True cho mọi block trừ block cuối. Block cuối được giữ parameter sau forward, tránh AllGather lại ngay khi backward bắt đầu (PyTorch không có thời gian prefetch).
3. `fully_shard(transformer_block, ...)`: áp FSDP với mesh và mp_policy. Parameter shard trên DP, AllGather khi cần.
4. `model.layers[layer_id] = transformer_block`: cập nhật reference (vì `checkpoint_wrapper` tạo module mới).

Lưu ý: `checkpoint_wrapper` được áp **trước** `fully_shard`. Lý do: checkpoint cần wrap forward gốc, sau đó FSDP wrap thêm forward đó. Thứ tự ngược lại sẽ làm FSDP unaware của checkpoint.

## Wrap root model

```python
model = fully_shard(model, **fsdp_config)
```

Wrap toàn bộ model (kể cả `tok_embeddings`, `norm`, `output`) với FSDP. Đây là wrap "root", cho các parameter không thuộc transformer block riêng. Quan trọng cho việc FSDP biết đến mọi parameter của model.

Lý do tách wrap block và wrap root: FSDP hoạt động per-unit. Mỗi unit = một module được wrap với `fully_shard`. Block là một unit (AllGather/free khi cần). Root cũng là một unit (cho các param ngoài block). Nếu chỉ wrap root, mọi parameter cùng một unit, AllGather một lần cho cả model rất tốn. Nếu chỉ wrap block, parameter của embedding/norm/output không được FSDP quản lý.

Compose này cho phép:

- Embedding/norm/output là một unit (root unit).
- Mỗi block là một unit riêng.

Total $L + 1$ unit. Mỗi unit AllGather riêng, free riêng. Tốt cho overlap.

## Luồng forward đầy đủ

Forward một step (giả sử SP + TP + FSDP + Checkpoint đều bật):

1. **Root unit AllGather**: gom embedding + norm + output params từ DP mesh.
2. **Embedding lookup** với DTensor: AllReduce trên TP cho embedding shard vocab.
3. **PrepareModuleInput cho `layers.0`**: scatter sequence dim, vào SP zone.
4. **Layer 0 AllGather** (FSDP): gom layer 0 params từ DP mesh. Sau bước này có TP-only shard.
5. **Layer 0 forward** (qua checkpoint wrapper): chỉ giữ input, không lưu intermediate. Bên trong tính TP collective bình thường.
6. **Layer 0 free** (FSDP reshard_after_forward=True): ném param, giữ shard nhỏ.
7. Lặp lại bước 4-6 cho layer 1, ..., L-1.
8. **Layer L-1 reshard_after_forward=False**: giữ param đến backward.
9. **Norm cuối** + **Output**: với root unit param đã gathered ở bước 1.
10. **Loss** với `loss_parallel`.

Backward đối ngẫu, mỗi unit AllGather param (cho recompute và backward), ReduceScatter gradient.

## Khả năng overlap

PyTorch FSDP cố gắng overlap AllGather của unit $i+1$ với compute của unit $i$. Nếu băng thông NVLink/InfiniBand đủ, AllGather "ẩn mình" sau compute, không thêm latency thực tế.

Để overlap tốt, mỗi unit cần đủ lớn để compute mất nhiều thời gian hơn AllGather. Với Transformer block của Llama-3 8B trên A100, compute mỗi block $\sim 10 ms$, AllGather param block $\sim 5 ms$ trên NVLink: overlap được.

Block quá nhỏ (ví dụ chỉ wrap mỗi linear), compute ngắn, AllGather chiếm tỉ lệ lớn. Overlap kém. Đó là lý do wrap unit ở mức "block", không nhỏ hơn.

## Tổng kết Phần 7

Phần 7 ghép TP + SP + FSDP + Checkpoint + Mixed Precision lại thành công thức training LLM 8B trên GPU 24 GB. Mỗi kỹ thuật giải một bottleneck:

- TP: parameter intra-layer.
- SP: residual activation.
- FSDP: parameter + grad + opt state cross-data-rank.
- Checkpoint: intermediate activation.
- Mixed precision: parameter + compute memory + compute speed.

Phần 8 sẽ walkthrough toàn bộ `02_large_language_model/` đầy đủ, kết hợp tất cả các phần đã học. Phần 9 là performance và debugging thực tế.
