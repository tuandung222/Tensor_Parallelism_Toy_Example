---
title: parallelism.py
---

# `parallelism.py`, hàm `parallelize`

`parallelism.py` chứa duy nhất một hàm `parallelize(model, device_mesh)`. Đây là "bộ não" áp dụng TP và FSDP cho model Llama-3 thuần. Chương này đi qua từng phần.

## Import block

```python
import torch
from model import Transformer
from torch.distributed._composable.fsdp import MixedPrecisionPolicy
from torch.distributed._composable.fsdp.fully_shard import fully_shard
from torch.distributed._tensor import Replicate, Shard
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
    parallelize_module,
)
```

Sáu thành phần chính:

- `MixedPrecisionPolicy`, `fully_shard`: FSDP API.
- `Replicate, Shard`: Placement type của DTensor.
- `checkpoint_wrapper`: activation checkpoint.
- `DeviceMesh`: type hint.
- `ColwiseParallel, RowwiseParallel, SequenceParallel, PrepareModuleInput, parallelize_module`: TP API.

Đây là toàn bộ API cần để cài cả TP + SP + FSDP + Checkpoint + Mixed Precision.

## Signature và mesh

```python
def parallelize(model: Transformer, device_mesh: DeviceMesh) -> Transformer:
    dp_mesh = device_mesh["data_parallel"]
    tp_mesh = device_mesh["tensor_parallel"]
```

Lấy hai sub-mesh từ mesh 2D. Tên `"data_parallel"` và `"tensor_parallel"` phải khớp với khi tạo mesh trong `train.py` (Lightning's `ModelParallelStrategy` đặt tên này mặc định).

## Plan TP root

```python
if tp_mesh.size() > 1:
    plan = {
        "tok_embeddings": RowwiseParallel(input_layouts=Replicate()),
        "output": ColwiseParallel(
            input_layouts=Shard(1),
            output_layouts=Shard(-1),
            use_local_output=False,
        ),
        "norm": SequenceParallel(),
        "layers.0": PrepareModuleInput(
            input_layouts=(Replicate(), None),
            desired_input_layouts=(Shard(1), None),
            use_local_output=True,
        ),
    }
    model = parallelize_module(model, tp_mesh, plan)
```

Plan này áp dụng cho cấp root của model. Bốn module được wrap:

- `tok_embeddings`: RowwiseParallel với input Replicate (tokens giống nhau trên mọi rank).
- `output`: LM Head, ColwiseParallel với input Shard(1) (sequence shard từ norm), output Shard(-1) (vocab shard), giữ DTensor cho `loss_parallel`.
- `norm`: SequenceParallel (RMSNorm cuối model, áp trên sequence shard).
- `layers.0`: PrepareModuleInput để chuyển từ Replicate (sau embedding) sang Shard(1) (vào SP zone). `use_local_output=True` unwrap về tensor thường trước khi vào block đầu.

Lưu ý: chỉ `layers.0` được wrap PrepareModuleInput ở root level. Các block khác sẽ nhận input đã ở Shard(1) từ block trước (do residual SP), không cần convert.

## Plan TP cho mỗi block

```python
for transformer_block in model.layers.values():
    plan = {
        "attention": PrepareModuleInput(
            input_layouts=(Shard(1), None),
            desired_input_layouts=(Replicate(), None),
        ),
        "attention.wq": ColwiseParallel(),
        "attention.wk": ColwiseParallel(),
        "attention.wv": ColwiseParallel(),
        "attention.wo": RowwiseParallel(output_layouts=Shard(1)),
        "attention_norm": SequenceParallel(),
        "feed_forward": PrepareModuleInput(
            input_layouts=(Shard(1),),
            desired_input_layouts=(Replicate(),),
        ),
        "feed_forward.w1": ColwiseParallel(),
        "feed_forward.w2": RowwiseParallel(output_layouts=Shard(1)),
        "feed_forward.w3": ColwiseParallel(),
        "ffn_norm": SequenceParallel(),
    }

    attn_layer = transformer_block.attention
    attn_layer.n_heads = attn_layer.n_heads // tp_mesh.size()
    attn_layer.n_kv_heads = attn_layer.n_kv_heads // tp_mesh.size()

    parallelize_module(transformer_block, tp_mesh, plan)
```

Plan đầy đủ cho một Transformer block, áp dụng cho mọi block:

**Phần Attention**:

- `attention` (parent): PrepareModuleInput từ Shard(1) (sau attention_norm SP) sang Replicate (cho QKV linear).
- `attention.wq, wk, wv`: ColwiseParallel chuẩn.
- `attention.wo`: RowwiseParallel với output Shard(1) (về SP zone qua ReduceScatter).
- `attention_norm`: SequenceParallel.

**Phần FFN**:

- `feed_forward` (parent): PrepareModuleInput từ Shard(1) (sau ffn_norm SP) sang Replicate.
- `feed_forward.w1, w3`: ColwiseParallel.
- `feed_forward.w2`: RowwiseParallel với output Shard(1).
- `ffn_norm`: SequenceParallel.

**Điều chỉnh attribute**:

- `n_heads //= tp_size` và `n_kv_heads //= tp_size`: cập nhật để `view` trong forward khớp với shape local sau TP shard.

Đây là plan Megatron + Sequence Parallel hoàn chỉnh. Mọi pattern ta đã derive ở Phần 3, 4, 5, 6 đều xuất hiện ở đây.

## Phần FSDP

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

Đã walkthrough ở Phần 7. Tóm lại: mỗi block được wrap checkpoint rồi fully_shard, model root cũng được fully_shard. Mixed precision policy bf16 cho compute, fp32 cho reduce.

## Trật tự TP rồi FSDP

Quan trọng nhất: TP được áp **trước** FSDP. Nếu FSDP trước, parameter đã shard trên DP, TP không thấy được structure nguyên gốc để biết shard cột hay hàng. Lỗi xảy ra ngay tại `parallelize_module`.

Trật tự TP-FSDP cũng là khuyến nghị chính thức của PyTorch trong tài liệu composable.

## Return

```python
return model
```

Model trả về có:

- Parameter là DTensor 2D-shard (TP $\times$ DP).
- Mỗi block được wrap checkpoint + FSDP.
- Mỗi linear có forward TP-aware (AllReduce/AllGather/ReduceScatter tự động).
- Mixed precision policy được tích hợp.

Một dòng `parallelize(model, mesh)` đã làm tất cả việc này.

## Customize cho kiến trúc khác

Nếu bạn muốn port plan này sang Mistral hoặc Qwen:

- Mistral: cấu trúc giống Llama, plan TP áp dụng nguyên vẹn. Có thể có sliding window attention, nhưng không ảnh hưởng TP.
- Qwen: thường có bias trên một số linear (đặc biệt $W_Q, W_K, W_V$ trong một số version Qwen). PyTorch ColwiseParallel xử lý bias tự động (shard cùng chiều với weight).
- Mixtral (MoE): expert routing thêm chiều "expert", cần expert parallelism (EP). Đây là 3D parallelism, ngoài phạm vi Phần 8.

Đối với kiến trúc decoder-encoder (T5), cần plan cho cả encoder và decoder, plus cross attention. Pattern Col-Col-Col-Row vẫn áp dụng.

Chương tiếp ta sang `train.py`.
