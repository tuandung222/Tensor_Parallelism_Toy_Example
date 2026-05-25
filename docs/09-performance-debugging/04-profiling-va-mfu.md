---
title: Profiling và MFU
---

# Profiling và Model FLOPs Utilization

Tối ưu mà không profile là tối ưu mù. Chương này hướng dẫn dùng PyTorch Profiler và Nsight Systems để đo training step, và cách tính MFU (Model FLOPs Utilization) để biết bạn đang khai thác bao nhiêu phần trăm khả năng GPU.

## PyTorch Profiler nhanh

```python
import torch.profiler as profiler

with profiler.profile(
    activities=[
        profiler.ProfilerActivity.CPU,
        profiler.ProfilerActivity.CUDA,
    ],
    schedule=profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
    on_trace_ready=profiler.tensorboard_trace_handler("./profile"),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    for step in range(10):
        train_step()
        prof.step()
```

Schedule: bỏ 1 step (wait), warmup 1, profile 3 step, lặp 1 lần. Tổng 5 step được profile.

Trace được lưu ở `./profile`, mở bằng TensorBoard:

```bash
tensorboard --logdir ./profile
```

Trong TensorBoard, tab "PyTorch Profiler" cho:

- Overview: tổng thời gian theo CPU, GPU, communication.
- Operator: thời gian top operator.
- Memory: bộ nhớ theo thời gian.
- Trace: timeline trên stream.

Trace là quan trọng nhất. Bạn sẽ thấy:

- Compute stream: kernel chạy nối tiếp.
- NCCL stream: collective xen kẽ.
- Khoảng trống (gap): GPU đang chờ.

Gap nhiều = bottleneck synchronization. Compute stream nối tiếp không gap = compute-bound (tốt). NCCL stream chiếm phần lớn = communication-bound (cần tối ưu).

## Nsight Systems

PyTorch Profiler tốt cho high-level view. Nsight Systems sâu hơn, cho thấy kernel CUDA, NCCL primitive, memory transfer.

```bash
nsys profile --trace=cuda,nvtx,osrt --output report python train.py
```

Mở report bằng Nsight Systems GUI. Trace có nhiều lane (stream, host thread, CPU). Có thể zoom xuống nano-second.

Đặc biệt hữu ích để xem overlap CUDA compute và NCCL collective. Nếu hai stream chạy đồng thời, bạn thấy "song song" trên timeline. Nếu không, có sync barrier rõ ràng.

## Tính MFU

MFU đo phần trăm peak FLOPs của GPU mà training đạt được.

Công thức:

$$
\text{MFU} = \frac{\text{Achieved TFLOPs/s}}{\text{Peak TFLOPs/s of GPU}}
$$

Achieved TFLOPs/s = Total FLOPs per step / Time per step.

Total FLOPs cho Transformer LLM (forward + backward):

$$
\text{FLOPs} \approx 6 \cdot N_{params} \cdot B \cdot S
$$

(Hệ số 6: 2 cho forward, 4 cho backward. Đây là approximation Chinchilla.)

Với $N_{params} = 8 \cdot 10^9$ (Llama-3 8B), $B = 8, S = 2048$:

$$
\text{FLOPs} \approx 6 \cdot 8 \cdot 10^9 \cdot 8 \cdot 2048 = 7.86 \cdot 10^{14}
$$

Nếu step mất $1$ giây: achieved $= 786$ TFLOPs/s.

A100 SXM4 80GB bf16 peak: $312$ TFLOPs/s. Nhưng chú ý: ta đo trên cả cluster.

Tổng peak cluster = $P_{GPUs} \cdot 312$ TFLOPs/s. Với 4 GPU: $1248$ TFLOPs/s.

MFU = $786 / 1248 = 63\%$. Khá cao.

H100 SXM5 bf16 peak: $989$ TFLOPs/s. Cùng setup, MFU = $786 / (4 \cdot 989) = 20\%$. Thấp. Có thể vì compute H100 quá nhanh, collective trở thành bottleneck. Cần điều chỉnh.

## Đọc trace và diagnose

Pattern phổ biến và cách diagnose:

**Pattern 1**: Compute stream gap lớn giữa các kernel.
- Nguyên nhân: synchronization. Có lẽ tensor được chuyển CPU-GPU không cần, hoặc `loss.item()` được gọi inside loop.
- Fix: tránh `.item()`, `.cpu()` trong inner loop.

**Pattern 2**: NCCL stream chiếm phần lớn timeline.
- Nguyên nhân: collective quá lớn so với compute. TP size quá cao, hoặc batch quá nhỏ.
- Fix: giảm TP, tăng batch.

**Pattern 3**: Compute kernel ngắn, gap nhỏ.
- Nguyên nhân: kernel launch overhead. Compile chưa được dùng, hoặc kernel không fused.
- Fix: `torch.compile`, FlashAttention, fused AdamW.

**Pattern 4**: Spike memory ở một step.
- Nguyên nhân: optimizer state init lần đầu, hoặc activation peak trong một block phức tạp.
- Fix: warmup steps để stabilize memory, bật checkpoint sâu hơn.

## Theo dõi MFU theo thời gian

Trong production training, log MFU mỗi $N$ steps. Nếu MFU giảm giữa chừng:

- Có thể model hit một regime mới (sequence dài hơn, hoặc bucket changing).
- Có thể network bị slowdown.
- Có thể GPU thermal throttle.

MFU stable là dấu hiệu training ổn định. Spike hoặc giảm bất thường đáng để điều tra.

## Lưu ý về benchmark

Không so MFU giữa các paper trực tiếp. Mỗi paper tính FLOPs khác nhau:

- Một số dùng 6N theo Chinchilla.
- Một số dùng 6N + attention term (cho sequence dài).
- Một số dùng đo trực tiếp.

Khi report MFU của bạn, ghi rõ công thức và setup (model, hardware, batch, sequence). Như vậy người khác có thể so sánh fair.

## Tổng kết

Profile thường xuyên, tối ưu có chứng cớ. Đừng đoán bottleneck. MFU $\ge 50\%$ là tốt, $\ge 60\%$ là rất tốt, $\ge 70\%$ chỉ có ở tối ưu sâu (FlashAttention, custom kernel).

Chương cuối Phần 9: các bug thường gặp với TP + FSDP.
