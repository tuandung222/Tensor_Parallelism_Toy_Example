---
title: Activation checkpoint chi tiết
---

# Activation checkpoint, các loại và trade-off

Phần 7 đã giới thiệu `checkpoint_wrapper`. Chương này đi sâu vào các biến thể: full checkpoint, selective checkpoint, CPU offload, và cách chọn cấu hình tối ưu.

## Full checkpoint

Wrap toàn bộ Transformer block. Forward chỉ giữ input của block, tất cả intermediate (Q, K, V, scores, hidden FFN) bị "quên". Backward đến block: re-compute forward để khôi phục intermediate, rồi backward.

- **Activation tiết kiệm**: $L$ block, mỗi block giữ chỉ input $\Theta(B S K)$, không phải full block $\Theta(B S K + B S H + B H S^2)$. Tiết kiệm $\sim 5-10$ lần activation.
- **Compute tăng**: +1 forward mỗi block, tổng compute increase $\sim 33\%$.

Đây là cấu hình mặc định trong toy code. Phù hợp khi activation là bottleneck cứng (LLM lớn, context dài).

## Selective Activation Checkpoint (SAC)

Ý tưởng: không phải mọi activation đều "đắt". Một số rẻ về compute nhưng tốn bộ nhớ, ta nên re-compute (giữ chỉ input). Một số khác đắt về compute (như scores, matmul), ta nên giữ. SAC cho phép chỉ định.

Ví dụ Llama-3 block:

- Scores $(B, H, S, S)$: tốn bộ nhớ và compute. Nếu re-compute, tăng compute đáng kể. Có thể chọn **giữ**.
- Q, K, V projection: cheap về compute, recompute không tốn nhiều. **Drop**.
- Hidden FFN: matmul đắt, recompute tốn. **Giữ**.
- SiLU, multiply: cheap, **drop**.

SAC mới có trong PyTorch experimental API (`torch.utils.checkpoint.checkpoint` với selective wrapper). Yêu cầu chỉ định `ops_to_save_in_recompute`.

Lợi ích so với full checkpoint: tiết kiệm $\sim 60\%$ activation memory (so với $\sim 80\%$ của full checkpoint), nhưng compute increase chỉ $\sim 10\%$ (so với $\sim 33\%$). Tradeoff hấp dẫn.

Toy code chưa dùng SAC, nhưng torchtitan trong các bản mới có dùng.

## CPU Offload checkpoint

Thay vì re-compute, ta **lưu activation lên CPU**, kéo về GPU khi cần backward.

- Activation memory GPU: gần 0 (chỉ giữ activation đang dùng).
- CPU memory: tốn nhưng CPU thường có TB RAM, không vấn đề.
- Cost: PCIe transfer GPU-CPU và ngược lại. PCIe 4.0 $\sim 32$ GB/s, một activation 8 GB mất $250 ms$. Quá chậm cho backward.

Trên server với HBM3 GPU và DDR5 CPU, PCIe 5.0 ($64$ GB/s), tỉ lệ tốt hơn. Nhưng vẫn chậm so với re-compute GPU-only.

CPU offload phù hợp khi:

- GPU memory cực kỳ hạn chế.
- Hoặc training context cực dài ($S \ge 128K$).
- Hoặc inference (không cần backward, chỉ giữ K/V cache).

Production LLM training thường không CPU offload, vì re-compute đủ tiết kiệm và rẻ hơn.

## NVMe Offload (DeepSpeed Zero-Infinity)

DeepSpeed có ZeRO-Infinity offload sang NVMe (SSD), không chỉ CPU. Cho phép train model "khổng lồ" trên cluster nhỏ. Đổi lại: bandwidth NVMe ~10 GB/s, chậm hơn nhiều. Chỉ dùng khi không có lựa chọn khác.

## Chọn cấu hình

Quy tắc thực dụng:

1. **Bắt đầu full checkpoint mọi block + SP + TP**. Đo activation memory peak.
2. Nếu memory thoải mái ($\le 60\%$ GPU), thử **bỏ checkpoint** vài layer cuối. Compute giảm.
3. Nếu memory chật ($\ge 90\%$), giữ full checkpoint. Cân nhắc thêm SAC.
4. Nếu vẫn OOM với full checkpoint, **giảm batch size**. Đây là cách rẻ nhất trước khi thử CPU offload.

## Đo lường

```python
import torch

torch.cuda.reset_peak_memory_stats()
for step in range(5):
    train_step()
peak_gb = torch.cuda.max_memory_allocated() / 1e9
print(f"Peak memory: {peak_gb:.2f} GB")
```

Nhớ chạy ít nhất 5 step. Step 1 có warmup, không đại diện.

Tách phần parameter, optimizer state, activation, gradient bằng cách đo ở các điểm khác nhau:

- Trước forward: parameter + optimizer state.
- Sau forward: + activation.
- Sau backward: + gradient.
- Sau optimizer step: trở về parameter + optimizer state.

Sự khác biệt cho biết mỗi thành phần chiếm bao nhiêu.

## Cân bằng compute và memory

LLM training là tối ưu hóa nhiều chiều:

$$
\text{Time per step} \propto \text{Compute} + \text{Communication} + \text{Synchronization}
$$

$$
\text{Memory} = \text{Param} + \text{Opt state} + \text{Gradient} + \text{Activation}
$$

Mọi kỹ thuật (TP, FSDP, checkpoint, mixed precision) là cách biến đổi giữa các đại lượng này. Mục tiêu: làm cả Time per step và Memory đều ≤ ngân sách hardware. Đây là bài toán tối ưu vector, không có công thức đóng. Phải thử và đo.

## Một cấu hình kinh nghiệm cho Llama-3 8B

Trên 8 GPU A100 80GB:

- TP=8, DP=1 (toàn TP).
- SP bật.
- FSDP với TP-only mesh (đơn giản hóa, DP=1 nên không FSDP).
- Mixed precision bf16/fp32.
- Full checkpoint mọi block.
- Batch size 8, sequence 8K.

MFU $\sim 50\%$, peak memory $\sim 60$ GB/GPU. Headroom cho context dài hơn hoặc batch lớn hơn.

Trên 32 GPU A100 80GB (4 node):

- TP=8 intra-node, FSDP=4 cross-node.
- 2D mesh.
- Toàn bộ kỹ thuật bật.

MFU $\sim 45\%$ (giảm chút do collective cross-node), batch effective $32$, throughput tăng 4x. Đây là setup chuẩn cho Llama-3 8B SFT.

Chương tiếp: profiling và MFU.
