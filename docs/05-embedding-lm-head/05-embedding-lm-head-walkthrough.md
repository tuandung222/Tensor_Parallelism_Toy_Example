---
title: Walkthrough Embedding và LM Head
---

# Walkthrough Embedding và LM Head trong toy code

Đoạn plan TP cho Embedding và LM Head trong `02_large_language_model/parallelism.py`:

```python
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

Đi từng dòng.

## `tok_embeddings`

Module này là `nn.Embedding(vocab_size, dim)`. `RowwiseParallel` shard `weight` theo dim 0 (chiều vocab). Mỗi rank giữ $V/P$ row.

`input_layouts=Replicate()`: tokens đầu vào của model giống nhau trên mọi rank trong TP group (do sampler). Khẳng định này cho PyTorch biết không cần convert input.

Output mặc định: tensor shape $(B, S, K)$. Sau lookup, kết quả là partial trên TP mesh (mỗi rank chỉ có giá trị thực ở các vị trí token thuộc range của mình). PyTorch sẽ AllReduce để biến thành replicate.

## `layers.0` và transition vào Sequence Parallel

```python
"layers.0": PrepareModuleInput(
    input_layouts=(Replicate(), None),
    desired_input_layouts=(Shard(1), None),
    use_local_output=True,
),
```

`PrepareModuleInput` là một transformer placement: nó nhận tensor ở placement này, chèn collective cần thiết để đưa về placement khác, rồi mới gọi forward của module wrapped.

Trên `layers.0` (block transformer đầu tiên):

- Input đến từ embedding, ở dạng Replicate (sau AllReduce ngầm của embedding).
- Desired input: `Shard(1)`, tức shard theo chiều sequence. Đây là yêu cầu của Sequence Parallel: chiều sequence được chia $P$ phần.

Để chuyển từ Replicate sang Shard(1), PyTorch chèn một ScatterScatter (lấy slice của chiều 1 cho rank tương ứng). Không phải collective AllReduce; chỉ là phép cắt local.

`use_local_output=True`: sau scatter, unwrap về tensor thường cho forward của block.

Đây là điểm bắt đầu Sequence Parallel zone, chi tiết ở Phần 6.

## `norm` (RMSNorm)

```python
"norm": SequenceParallel(),
```

`norm` ở đây là RMSNorm cuối cùng (sau toàn bộ layers, trước `output`). `SequenceParallel` style nói rằng tensor đi vào norm này đang shard chiều sequence (vì layers trước cũng SP), và norm tính trên chiều hidden độc lập theo position.

RMSNorm là phép theo position: normalize trên chiều last (hidden). Mỗi position xử lý độc lập, nên shard sequence không phá norm. SequenceParallel hoạt động bằng cách áp norm trên tensor đang shard sequence, kết quả vẫn shard sequence.

## `output` (LM Head)

```python
"output": ColwiseParallel(
    input_layouts=Shard(1),
    output_layouts=Shard(-1),
    use_local_output=False,
),
```

Input đến từ `norm`, ở placement `Shard(1)` (sequence shard). LM Head cần input replicate cho matmul. ColwiseParallel với `input_layouts=Shard(1)` ngầm thực hiện AllGather trên chiều sequence trước khi matmul.

Output: $\mathrm{Shard}(-1)$, tức shard cuối (chiều vocab). Đây là ColwiseParallel chuẩn.

`use_local_output=False`: giữ DTensor để `loss_parallel` nhận diện được.

## Tổng kết flow một forward đầy đủ

| Stage | Placement input | Placement output | Collective ngầm |
|-------|-----------------|-------------------|-----------------|
| `tok_embeddings` | Replicate (tokens) | Partial -> Replicate sau AllReduce | AllReduce |
| `layers.0` PrepareModuleInput | Replicate | Shard(1) | Scatter (slice local) |
| `layers.*` (transformer blocks) | Shard(1) | Shard(1) | tùy block, xem Phần 6 |
| `norm` | Shard(1) | Shard(1) | (không, SP local) |
| `output` LM Head | Shard(1) | Shard(-1) | AllGather (sequence) |
| `cross_entropy` với `loss_parallel` | Shard(-1) logits | scalar loss | AllReduce (LSE) |

Toàn bộ một forward, không kể nội bộ layer, có 1 AllReduce (embedding), 1 AllGather (LM Head input), 1 AllReduce (loss). Nội bộ mỗi layer thêm vài collective nữa cho TP và SP.

Phần 5 kết thúc. Bạn đã có toàn bộ pattern TP cho một Transformer LLM hoàn chỉnh. Phần 6 sẽ giải thích Sequence Parallel: vì sao chiều sequence cũng được shard, và lợi ích về activation memory.
