---
title: Tổng quan Phần 6
---

# Phần 6: Sequence Parallelism

Sau Phần 3, 4, 5, bạn đã shard trọng số của mọi linear trong Transformer LLM. Vẫn còn một thứ chưa shard: **activation** dọc theo chiều sequence. Sequence Parallelism (SP) là kỹ thuật mở rộng TP để cũng shard chiều này, giúp giảm activation memory thêm $P$ lần.

## Vấn đề mà SP giải

Trong TP cổ điển (Megatron), giữa hai block transformer ta luôn quay về tensor replicate $(B, S, K)$. Tensor này có chứa toàn bộ thông tin sequence, không bị shard. Với context dài (S = 8K, 32K, 100K), kích thước này dominat bộ nhớ activation:

$$
\text{Activation memory residual} \approx L \cdot B \cdot S \cdot K \cdot 4\text{ bytes}
$$

với $L$ là số layer. Với $L = 32, B = 1, S = 32000, K = 4096$: $\approx 16$ GB chỉ riêng residual activation. Không vừa GPU 24 GB.

Quan sát then chốt: ở các điểm như RMSNorm, dropout, residual addition, các phép tính áp dụng **trên từng position độc lập** (per-token), không phụ thuộc các position khác. Vậy ta có thể shard chiều sequence ở những zone này mà không làm sai semantics.

## Sequence Parallel zone vs Tensor Parallel zone

Một Transformer block với SP được chia thành hai zone:

- **TP zone**: bên trong Attention và FFN, tensor có placement $\mathrm{Replicate}$ trên chiều sequence (vì cần đầy đủ sequence để tính softmax đối với attention; matmul đối với FFN tự nhiên tách head/hidden mà không tách sequence).
- **SP zone**: ở RMSNorm, residual, dropout, tensor có placement $\mathrm{Shard}(1)$ trên chiều sequence.

Ranh giới giữa hai zone cần một transition: 

- Vào TP zone (từ SP): AllGather chiều sequence để có sequence đầy đủ.
- Ra TP zone (về SP): ReduceScatter chiều sequence để cộng các shard (từ output partial của Row Parallel) đồng thời scatter ra sequence shard.

Sự khôn ngoan của SP nằm ở chỗ: ReduceScatter rẻ hơn AllReduce + Scatter riêng lẻ. Tổng chi phí giao tiếp của SP **bằng** chi phí TP cổ điển, nhưng kích thước tensor activation giảm $P$ lần ở SP zone.

## Mục tiêu sau Phần 6

Sau bốn chương:

1. Hiểu chính xác zone nào shard sequence, zone nào replicate sequence.
2. Derive chi phí giao tiếp của SP, chứng minh rằng nó **không tăng** so với TP cổ điển.
3. Đọc được `SequenceParallel()` style và `PrepareModuleInput` cho transition.
4. Tính được activation memory tiết kiệm.

## Cấu trúc bốn chương

Chương đầu (`02-motivation-activation-memory`) phân tích kỹ vấn đề activation memory, tính cụ thể bộ nhớ cho LLM phổ biến và chỉ ra vì sao chỉ TP không đủ.

Chương hai (`03-norm-sequence-parallel`) phát biểu rõ SequenceParallel style cho RMSNorm/LayerNorm. Derive lý do tại sao norm tự nhiên SP, và vì sao SP zone phải có cấu trúc nhất quán.

Chương ba (`04-prepare-module-input-transitions`) phân tích ranh giới TP-SP. AllGather và ReduceScatter, vì sao chúng đối ngẫu, và cách `PrepareModuleInput` đóng gói.

Chương cuối (`05-sp-cost-analysis`) tổng kết chi phí: giao tiếp, bộ nhớ, compute. Đếm thực tế trên Llama-3 8B.

## Quan hệ với các phần trước

SP **không thay thế** TP, nó **mở rộng** TP. Plan TP của Phần 3, 4, 5 vẫn áp dụng nguyên vẹn. SP chỉ thêm style `SequenceParallel()` cho norm và thêm `PrepareModuleInput` ở các điểm chuyển zone. Code thực tế khác biệt vài dòng, nhưng activation memory giảm $P$ lần.

Phần 6 này quan trọng đặc biệt khi bạn train LLM với context dài (Llama-3 8K, Qwen 32K, hoặc dài hơn). Nếu chỉ TP, bạn sẽ OOM. Có SP, bạn fit.
