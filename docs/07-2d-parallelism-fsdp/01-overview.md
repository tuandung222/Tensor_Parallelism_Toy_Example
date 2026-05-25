---
title: Tổng quan Phần 7
---

# Phần 7: 2D Parallelism với FSDP

Đến Phần 6, ta đã thấy TP + SP chia parameter và activation theo chiều "TP mesh". Nhưng còn một chiều quan trọng chưa khai thác: chiều "DP mesh", nơi mỗi rank xử lý một subset khác nhau của dữ liệu. Khi kết hợp TP với FSDP (Fully Sharded Data Parallel), ta có 2D parallelism, đây là công thức training chuẩn cho LLM lớn ở scale lớn.

## Vì sao 2D parallelism

Riêng TP có giới hạn $P_{TP} \le n_{kv\_heads}$ (Phần 4). Với Llama-3 8B $n_{kv\_heads} = 8$, TP tối đa 8. Nếu ta có cluster 64 GPU, riêng TP không đủ tận dụng. Cần thêm DP.

Riêng FSDP thì shard parameter và optimizer state theo dim 0 nhưng không shard intra-layer như TP. Compute mỗi rank vẫn là compute cho toàn bộ layer trên minibatch của rank đó. Nếu một layer không vừa trong GPU (params + activation), FSDP đơn lẻ không cứu được.

2D = TP cho intra-layer + FSDP cho dữ liệu. Hai chiều bù trừ nhau: TP cứu compute và activation, FSDP cứu parameter + optimizer state. Cả hai cùng dùng, không xung đột nếu khai báo mesh đúng.

## Cấu trúc bốn chương

Chương đầu (`02-fsdp-co-ban`) giới thiệu FSDP: parameter sharding theo data parallel rank, all-gather để forward, reduce-scatter cho gradient. So sánh với DDP cổ điển.

Chương hai (`03-mesh-2d-va-luong-gradient`) giải thích mesh 2D, cách init, và phân tích luồng gradient khi TP + FSDP chung.

Chương ba (`04-mixed-precision-va-checkpoint`) bàn về mixed precision policy (bf16 cho compute, fp32 cho master weight, fp32 cho gradient reduce) và activation checkpoint trong context FSDP.

Chương cuối (`05-fsdp-walkthrough`) walkthrough đoạn FSDP trong `parallelism.py`: `fully_shard`, `checkpoint_wrapper`, `reshard_after_forward`.

## Mục tiêu sau Phần 7

Bạn:

1. Hiểu FSDP shard cái gì, khi nào, và phân biệt với DDP.
2. Vẽ được luồng forward + backward cho một layer khi cả TP và FSDP cùng áp.
3. Đọc được `fully_shard(transformer_block, mesh=dp_mesh, ...)` và biết tham số `reshard_after_forward` ảnh hưởng gì.
4. Cài đặt được mixed precision policy phù hợp với LLM training.

Phần 7 này quan trọng vì 95% production LLM training dùng 2D parallelism. Nếu bạn chỉ biết TP một mình, bạn đang nhìn một nửa bức tranh.
