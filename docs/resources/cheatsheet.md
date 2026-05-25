---
title: Cheatsheet
---

# Cheatsheet Tensor Parallelism

Bảng tổng hợp một trang cho người đã đọc các phần lý thuyết. Dùng làm tài liệu tra cứu khi viết code.

## Quy tắc đặt placement cho một linear đơn

| Tình huống đầu vào | Tình huống đầu ra mong muốn | Style nên dùng |
|--------------------|------------------------------|----------------|
| Replicate          | Shard chiều cuối             | `ColwiseParallel()` |
| Shard chiều cuối   | Replicate (sau AllReduce)    | `RowwiseParallel()` |
| Replicate          | Replicate (giữ nguyên)       | Không dùng TP cho linear này |
| Shard chiều cuối   | Shard chiều cuối (giữ nguyên) | Không phổ biến, cân nhắc kiến trúc |

## Quy tắc cho chuỗi hai linear liên tiếp

Nếu chuỗi có dạng `Replicate -> Linear -> Activation element-wise -> Linear -> Replicate`, đặt linear đầu Column và linear sau Row. Đây là Megatron pattern.

Nếu có nhiều linear "mở" song song trước khi hội tụ vào element-wise multiply (như SwiGLU), tất cả linear mở đều Column, linear đóng cuối Row.

## Forward và backward collective

| Style | Forward collective | Backward collective trên gradient input |
|-------|---------------------|------------------------------------------|
| Column | (không) | AllReduce |
| Row    | AllReduce ở cuối | (không, identity) |

Quy tắc đối ngẫu: forward AllReduce kéo theo backward identity, và ngược lại.

## Bảng chi phí cho Megatron MLP

Một MLP block với $P$ rank TP, batch $B$, hidden $K$, FFN hidden $H$, dùng SwiGLU:

| Đại lượng | Trị số |
|-----------|--------|
| Parameter mỗi rank | $3 K H / P$ (không tính norm) |
| Forward FLOPs mỗi rank | $\approx 6 B K H / P$ |
| Forward collective | 1 AllReduce kích thước $B \times K$ |
| Backward collective | 1 AllReduce kích thước $B \times K$ |
| Activation peak mỗi rank (không checkpoint) | $\approx 2 B H / P$ (cho hai nhánh shard cột) |

## Mesh khai báo thường gặp

```python
from torch.distributed.device_mesh import init_device_mesh

# 1D mesh chỉ có TP
mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("tensor_parallel",))

# 2D mesh: TP cho layer, FSDP cho dữ liệu
mesh_2d = init_device_mesh(
    "cuda",
    (dp_size, tp_size),
    mesh_dim_names=("data_parallel", "tensor_parallel"),
)

tp_mesh = mesh_2d["tensor_parallel"]
dp_mesh = mesh_2d["data_parallel"]
```

## Plan TP cho Llama-style Transformer block

```python
plan = {
    "attention": PrepareModuleInput(...),
    "attention.wq": ColwiseParallel(),
    "attention.wk": ColwiseParallel(),
    "attention.wv": ColwiseParallel(),
    "attention.wo": RowwiseParallel(),
    "attention_norm": SequenceParallel(),

    "feed_forward": PrepareModuleInput(...),
    "feed_forward.w1": ColwiseParallel(),
    "feed_forward.w2": RowwiseParallel(),
    "feed_forward.w3": ColwiseParallel(),
    "ffn_norm": SequenceParallel(),
}
```

Nhớ chỉnh `n_heads = n_heads // tp_size` và `n_kv_heads = n_kv_heads // tp_size` cho mỗi Attention block (TP chia head). Embedding và LM head có pattern riêng (Phần 5).

## Lệnh chạy `01_simple_model` trên 4 GPU

```bash
cd 01_simple_model
python train.py
```

Lightning sẽ tự dựng mesh và chạy TP=4. Yêu cầu CUDA, NCCL, và 4 GPU.

## Lệnh chạy `02_large_language_model`

```bash
cd 02_large_language_model
torchrun --nproc_per_node=4 train.py
```

Cấu hình 2D parallelism với TP=2 và DP=2 (xem `train.py` để chỉnh). Yêu cầu GPU đủ bộ nhớ cho mỗi shard.

## Debug placement mismatch

Triệu chứng: PyTorch trả lỗi như `expected Replicate but got Partial`, hoặc kết quả loss bị NaN ngay step đầu.

Checklist:

1. Mỗi `PrepareModuleInput` có đúng `input_layouts` không. Khi mesh có nhiều chiều, input có nhiều placement (một cho mỗi chiều).
2. Cấu hình `output_layouts` của Row Parallel có khớp với input expected của module kế tiếp không. Đặc biệt khi bật Sequence Parallel.
3. Số head có chia hết cho TP size không. Nếu `n_heads = 32` và `tp_size = 6`, sẽ lỗi.
4. Sampler có replicate batch trong TP group không. Trong 2D, dùng `DistributedSampler` với `num_replicas = dp_size`.

## Cảnh báo bộ nhớ

Peak memory thường không ở forward, mà ở backward khi optimizer step. Đo `torch.cuda.max_memory_allocated()` sau ít nhất một full step. Trước step đầu tiên, FSDP/CPU offload có thể che giấu chi phí thực.

## Bộ nhớ vs giao tiếp, kim chỉ nam

Tăng TP size giảm tham số mỗi rank tuyến tính, nhưng kích thước all-reduce mỗi forward gần như cố định trên tensor $(B, K)$. Sau ngưỡng TP=8 trên NVLink, lợi ích thường lệch sang FSDP và Activation Checkpointing thay vì tăng TP tiếp.
