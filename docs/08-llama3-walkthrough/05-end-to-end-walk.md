---
title: End-to-end một step
---

# End-to-end một step training

Chương cuối Phần 8 mô tả một full step training của Llama-3 toy chạy trên mesh 2D, theo dõi placement tensor từ input đến gradient update. Giả sử TP=2, DP=2, world_size=4.

## Setup

Mesh 2D shape $(2, 2)$:

```
rank 0: (dp=0, tp=0)
rank 1: (dp=0, tp=1)
rank 2: (dp=1, tp=0)
rank 3: (dp=1, tp=1)
```

`tp_mesh` gồm `[rank 0, rank 1]` cho `dp_rank=0`, và `[rank 2, rank 3]` cho `dp_rank=1`. `dp_mesh` gồm `[rank 0, rank 2]` cho `tp_rank=0`, và `[rank 1, rank 3]` cho `tp_rank=1`.

Batch size 8, sequence length 128. `RandomTokenDataset` sample chia bởi `DistributedSampler`: rank 0 và 1 cùng nhận batch A (8 sample), rank 2 và 3 cùng nhận batch B (8 sample khác).

## Step 1: load batch và shift

Mỗi rank có `batch` shape $(8, 129)$ là tensor int64 (token IDs). Tensor này replicate trên TP group (rank 0 và 1 có cùng batch A), khác giữa DP group.

```python
inputs = batch[:, :-1]  # shape (8, 128)
labels = batch[:, 1:]   # shape (8, 128)
```

## Step 2: embedding lookup

```python
h = self.tok_embeddings(inputs)
```

`tok_embeddings` áp RowwiseParallel trên `tp_mesh`. Vocab 32000 chia 2 = 16000 token/rank. Rank tp=0 giữ token 0-15999, rank tp=1 giữ 16000-31999.

Forward: mỗi rank lookup. Cho token thuộc range của rank, lấy embedding. Cho token ngoài range, trả 0. Sau đó AllReduce trên TP group cộng kết quả, được embedding đầy đủ.

State sau dòng này:

- `h` là DTensor, placement `(Replicate trên DP, Replicate trên TP)`, shape $(8, 128, 4096)$.

(FSDP đã AllGather embedding params từ DP trước đó.)

## Step 3: vào layer 0

```python
h = layers[0](h, freqs_cis)
```

Trước khi forward layer 0:

- `PrepareModuleInput` cho `layers.0` (root plan): convert input từ Replicate sang Shard(1) trên TP. Scatter slice sequence dim. Sau bước này, `h` là DTensor `(Replicate trên DP, Shard(1) trên TP)`, shape local $(8, 64, 4096)$.
- `use_local_output=True` unwrap DTensor về tensor thường shape $(8, 64, 4096)$.

Vào `TransformerBlock.forward`:

```python
h_attn = self.attention(self.attention_norm(x), freqs_cis)
```

- `attention_norm(x)`: SequenceParallel, áp norm local trên $(8, 64, 4096)$. Output cùng shape.
- `attention(h_norm)`: PrepareModuleInput chuyển `(Shard(1),)` sang `(Replicate,)`, gồm AllGather chiều sequence trên TP. Sau bước này input của attention là $(8, 128, 4096)$ replicate trên TP.

Bên trong attention:

- `wq, wk, wv` ColumnParallel: output local Shard(-1). Shape $(8, 128, 8 \cdot 128 = 1024)$ cho wq (TP=2, n_heads=16 local), $(8, 128, 4 \cdot 128 = 512)$ cho wk, wv (n_kv_heads=4 local).
- View thành dạng head: $(8, 128, 16, 128)$ cho xq, $(8, 128, 4, 128)$ cho xk, xv. (Chú ý `n_heads` đã được giảm bởi `parallelize`.)
- Rotary, repeat_kv, transpose, scaled_dot_product_attention: tất cả local trên rank.
- Output sau attention: $(8, 128, 16 \cdot 128 = 2048)$ local, Shard(-1).
- `wo` RowwiseParallel với `output_layouts=Shard(1)`: ReduceScatter chiều sequence. Output local $(8, 64, 4096)$, placement Shard(1).

Residual:

```python
h = x + h_attn  # x: Shard(1), h_attn: Shard(1), output: Shard(1)
```

Trên DTensor, addition giữa hai tensor cùng placement không sinh collective.

Tương tự FFN:

```python
return h + self.feed_forward(self.ffn_norm(h))
```

- `ffn_norm`: SP local.
- `feed_forward`: PrepareModuleInput AllGather sequence. Bên trong `w1, w3` ColumnParallel, SiLU, multiply local, `w2` RowwiseParallel ReduceScatter. Output Shard(1).
- Residual.

Output của layer 0 là DTensor `(Replicate trên DP, Shard(1) trên TP)`, shape local $(8, 64, 4096)$.

## Step 4: qua tất cả layer

Lặp lại Step 3 cho 16 layer. FSDP AllGather param trước mỗi block, ReduceScatter param sau backward.

Mỗi layer:

- 2 AllGather sequence (vào attention và FFN).
- 2 ReduceScatter sequence (ra attention và FFN qua wo, w2).
- 1 AllGather param (FSDP, trước forward).
- (Backward sau này: 1 AllGather param + 1 ReduceScatter gradient.)

## Step 5: norm cuối và LM head

```python
h = self.norm(h)  # SP, local
return self.output(h).float()
```

- `norm`: SequenceParallel local.
- `output` ColwiseParallel với `input_layouts=Shard(1), output_layouts=Shard(-1)`. Trước matmul, AllGather sequence để có input replicate. Matmul, output local Shard(-1) chiều vocab. Shape local $(8, 128, 16000)$.
- `.float()` cast sang fp32.

Output cuối: DTensor `(Replicate trên DP, Shard(-1) trên TP)`, shape local $(8, 128, 16000)$. `use_local_output=False` giữ dưới dạng DTensor.

## Step 6: loss

```python
with loss_parallel():
    loss = F.cross_entropy(output.reshape(-1, output.size(-1)), labels.reshape(-1))
```

- `output.reshape(-1, output.size(-1))`: reshape DTensor Shard(-1) sang $(B \cdot S, V)$ local $(1024, 16000)$.
- `cross_entropy` với `loss_parallel`: tính LSE local trên shard cuối, AllReduce LSE trên TP, AllReduce term $z_{t^*}$, cộng ra loss.
- Output: scalar (DTensor replicate trên cả TP và DP).

Total collective trong loss: 1 AllReduce trên TP cho LSE.

## Step 7: backward

```python
self.backward(loss)
```

Inside `loss_parallel`:

- Gradient của loss với output: tensor Shard(-1), tính được local.
- Backward qua `output` ColwiseParallel: gradient với norm output là replicate (sau AllReduce trên TP). Output Shard(1) sau implicit scatter.
- Backward qua `norm`: SP local.
- Backward qua từng layer (ngược): với FSDP `reshard_after_forward=True`, AllGather param trước recompute (vì có checkpoint). Tính forward lại để lấy intermediate, rồi backward bình thường.
  - Bên trong block: backward qua ReduceScatter sequence (đối ngẫu: AllGather), backward qua attention/FFN, backward qua AllGather (đối ngẫu: ReduceScatter).
  - Gradient param: ReduceScatter trên DP để aggregate.
- Backward qua embedding: gradient của embedding param là shard hàng, AllReduce-shape trên TP để gom đóng góp.

Tổng collective backward per block: 4 (AllGather + ReduceScatter sequence forward đối ngẫu) + 1 (AllGather param) + 1 (ReduceScatter gradient param) = 6.

## Step 8: optimizer step

```python
optimizer.step()
```

AdamW chạy local trên shard. Mỗi rank update parameter shard của mình, dùng optimizer state shard tương ứng. Mixed precision: load param fp32 master, gradient fp32, update fp32, cast bf16 lưu (hoặc giữ fp32 tùy paradigm).

Không collective ở optimizer step.

## Tổng collective một step

Per Transformer block:

- Forward TP: 2 AllGather sequence + 2 ReduceScatter sequence = 4.
- Backward TP: 4 (đối ngẫu).
- FSDP: 1 AllGather forward + 1 AllGather backward + 1 ReduceScatter gradient = 3.

Tổng một block: 11 collective.

Per model (16 layer + embedding + LM head):

- 16 * 11 = 176 collective cho block.
- Vài AllReduce/AllGather thêm cho embedding, norm, LM head.

Tổng $\sim 180$ collective. Mỗi collective nhỏ (vài KB đến vài MB). NCCL ring trên NVLink xử lý $\sim 100\mu s$ mỗi cái, tổng overhead $\sim 18 ms$ một step. Compute một step Llama-3 8B trên 4 GPU $\sim 1 s$. Overhead giao tiếp $\sim 2\%$, nhỏ.

## Bài học cuối

Một step end-to-end của Llama-3 với 2D parallelism là một bản giao hưởng các collective trên hai mesh khác nhau, xen kẽ với compute, được PyTorch DTensor + FSDP + Lightning tự động hoá. Bạn không cần code dòng nào về collective, nhưng để debug được, bạn phải biết placement của mọi tensor ở mọi điểm. Đó là bài học chính của Phần 8.

Phần 9 tiếp theo sẽ bàn về performance, profiling, và các pitfall debug thường gặp.
