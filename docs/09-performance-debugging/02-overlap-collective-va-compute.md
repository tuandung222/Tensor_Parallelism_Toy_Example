---
title: Overlap collective với compute
---

# Overlap collective với compute

Một step training LLM gồm hàng trăm collective xen kẽ với compute. Nếu mỗi collective phải đợi compute trước, rồi compute đợi collective sau, tổng thời gian là tổng tất cả. Mục tiêu của overlap: cho collective chạy **đồng thời** với compute, tổng thời gian gần với phần lớn hơn.

## CUDA stream cơ bản

CUDA stream là một hàng đợi lệnh trên GPU. Lệnh trong cùng stream chạy tuần tự, lệnh trong các stream khác nhau có thể chạy đồng thời (nếu tài nguyên cho phép).

PyTorch mặc định dùng một stream chính cho compute. NCCL tạo các stream phụ cho collective.

Để overlap, ta xếp:

- Compute trên stream chính.
- Collective trên stream NCCL.
- Đồng bộ giữa hai stream chỉ khi có dependency thực sự.

## FSDP AllGather overlap

FSDP cố gắng prefetch parameter của unit kế tiếp trong khi unit hiện tại đang compute. Cụ thể:

1. Bắt đầu forward unit $i$ trên compute stream.
2. Đồng thời, issue AllGather param unit $i+1$ trên NCCL stream.
3. Khi compute unit $i$ xong, AllGather unit $i+1$ thường đã xong (hoặc gần xong). Đồng bộ.
4. Bắt đầu forward unit $i+1$.

Nếu compute unit $i$ mất $T_c$ và AllGather unit $i+1$ mất $T_g$, tổng thời gian = $\max(T_c, T_g)$ thay vì $T_c + T_g$. Nếu $T_c \ge T_g$, overlap "ẩn" hoàn toàn AllGather, không thêm latency.

PyTorch FSDP2 tự động prefetch một số unit về phía trước. Số unit prefetch điều khiển bởi `prefetch_limit`. Mặc định 1, nghĩa là chỉ prefetch unit kế tiếp.

## TP collective overlap

TP collective (AllReduce, AllGather, ReduceScatter) thường nằm trên đường tới hạn (critical path) của một block, vì kết quả collective là input của bước compute tiếp theo. Khó overlap hơn FSDP.

Một số kỹ thuật:

**Sequence Parallel + AllGather đầu zone**: cho phép AllGather sequence chạy **trước** khi cần. Ví dụ AllGather chiều sequence của Q/K/V có thể bắt đầu trong khi norm vẫn đang chạy trên tensor cuối cùng. PyTorch cố gắng schedule sao cho được.

**Async TP**: experimental, dùng async collective với async tensor parallel API. PyTorch đang phát triển; chưa stable trong toy code.

**Compute-communication overlap với compile**: `torch.compile` kết hợp với DTensor có thể fuse và reorder để tăng overlap. Vẫn experimental.

Trong thực tế, TP collective overlap kém hơn FSDP overlap. Vì vậy TP nên trên NVLink (latency thấp), FSDP có thể trên InfiniBand (overlap che giấu).

## Prefetch giới hạn

Prefetch quá nhiều unit về phía trước có cái giá: bộ nhớ peak tăng vì nhiều unit cùng giữ parameter đầy đủ.

Trade-off:

- Prefetch 1 unit: overlap đủ tốt cho compute lớn, bộ nhớ thấp.
- Prefetch 3-5 unit: overlap tối đa nhưng bộ nhớ tăng. Phù hợp khi compute mỗi unit nhỏ.

Trên LLM 8B với A100, prefetch 1 thường đủ. Trên LLM nhỏ với compute mỗi unit ngắn, có thể tăng lên 2-3.

## Bottleneck phân tích

Nếu profile cho thấy compute chiếm $\ge 90\%$ thời gian (collective ẩn hết), bạn đang ở chế độ "compute-bound". Tối ưu compute (FlashAttention, mixed precision, fused kernel) sẽ giảm thời gian step.

Nếu collective chiếm $\ge 30\%$ thời gian (vượt khả năng overlap), bạn đang "communication-bound". Tối ưu: tăng batch để compute lớn hơn, giảm TP size để collective nhỏ hơn, hoặc nâng cấp hardware (NVLink faster).

Trên Llama-3 8B trên A100 NVLink, MFU thường $\sim 40-50\%$, compute chiếm $\sim 75\%$, communication chiếm $\sim 25\%$. Có thể đẩy MFU lên $55-60\%$ với tuning, nhưng overhead luôn còn.

## Stream tuning trong PyTorch

`torch.cuda.set_stream(stream)`: chỉ định stream cho operation tiếp theo. Hữu ích khi viết custom kernel.

`with torch.cuda.stream(s)`: context manager cho stream.

Trong code FSDP/DTensor PyTorch, bạn ít khi cần can thiệp stream manual. Framework tự lo. Chỉ khi viết custom communication (ví dụ all-to-all cho expert parallelism), bạn mới phải tự manage stream.

## Visualize overlap

Cách tốt nhất để biết overlap có hiệu quả là profile với Nsight Systems hoặc PyTorch Profiler có support trace:

```python
import torch.profiler

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    profile_memory=True,
) as prof:
    for step in range(10):
        train_step()
        prof.step()

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

Xem trace trên tensorboard, sẽ thấy stream timeline. Compute stream và NCCL stream chạy song song hay tuần tự đều hiện ra. Đây là cách trực quan nhất để hiểu overlap.

## Tóm tắt

| Vấn đề | Giải pháp |
|--------|-----------|
| FSDP AllGather chậm | Tăng prefetch, tăng unit size |
| TP AllReduce chậm | Đặt TP intra-node NVLink |
| Compute không full GPU | FlashAttention, fused kernel |
| Stream không overlap | Profile + check sync |

Chương tiếp: activation checkpointing thông minh.
