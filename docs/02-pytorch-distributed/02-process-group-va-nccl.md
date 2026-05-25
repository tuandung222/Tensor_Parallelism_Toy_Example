---
title: ProcessGroup và NCCL Backend
---

# ProcessGroup và NCCL Backend

Trước khi nói tensor parallelism, hãy hiểu cái nền móng nhất: các process trên các GPU tìm thấy nhau và nói chuyện như thế nào. Đây là cơ chế của `torch.distributed.init_process_group`.

## Mô hình thực thi đa process

PyTorch distributed không dùng một process duy nhất với nhiều thread. Nó dùng **nhiều process, mỗi process gắn với một GPU**. Đây là mô hình SPMD (Single Program Multiple Data): cùng một file Python được chạy trên $P$ process, mỗi process có rank riêng và xử lý dữ liệu khác nhau.

Trên 4 GPU một node, bạn launch 4 process. Trên 16 GPU 2 node (8 GPU mỗi node), bạn launch 16 process. Công cụ launch phổ biến:

- `torchrun --nproc_per_node=4 train.py`: launch 4 process cùng node.
- `torchrun --nproc_per_node=8 --nnodes=2 --rdzv_endpoint=master:29500 train.py`: launch trên nhiều node.
- Lightning ẩn việc này: bạn chỉ cấu hình `devices=4` trong `Trainer`, Lightning tự launch process.

Mỗi process biết rank của mình qua biến môi trường `RANK`, `LOCAL_RANK`, và `WORLD_SIZE`. PyTorch đọc các biến này khi `init_process_group` được gọi.

## init_process_group

Đây là dòng code đầu tiên cho distributed training:

```python
import torch
import torch.distributed as dist

dist.init_process_group(backend="nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()
torch.cuda.set_device(rank % torch.cuda.device_count())
```

Backend `nccl` là lựa chọn chuẩn cho GPU. NCCL (NVIDIA Collective Communications Library) xử lý các collective primitive cho GPU một cách tối ưu, dùng NVLink khi có.

Sau khi init, mọi process đã biết nhau. `dist.all_reduce`, `dist.broadcast`, vân vân, sẵn sàng.

Trong Lightning, `init_process_group` được gọi tự động khi `ModelParallelStrategy` được khởi tạo. Bạn không phải viết tay.

## NCCL là gì

NCCL là thư viện C++ của NVIDIA cho collective primitive trên GPU. Nó tận dụng NVLink khi GPU cùng node, fallback về PCIe khi không có NVLink, dùng RDMA Infiniband cho inter-node.

Đặc điểm NCCL.

Topology-aware: tự phát hiện kết nối giữa GPU và chọn algorithm tối ưu (Ring, Tree, NVLS).

Async: collective trả về handle, không block CPU. Bạn có thể overlap với compute.

GPU-direct: dữ liệu đi thẳng giữa GPU không qua CPU, tận dụng băng thông HBM và NVLink.

NCCL không phải lựa chọn duy nhất. Backend `gloo` chạy trên CPU và mạng thông thường, dùng khi debug hoặc không có GPU. Backend `mpi` dùng MPI lib hệ thống. Cho TP, gần như luôn dùng `nccl`.

## ProcessGroup

`ProcessGroup` là một subset của các process tham gia chung một collective. Mặc định khi init, có một ProcessGroup global gồm mọi process.

Tại sao cần subset. Vì trong 2D parallelism, ta có hai chiều: TP và DP. Mỗi all-reduce TP chỉ nên xảy ra giữa các rank trong cùng TP group, không liên quan tới rank thuộc DP group khác.

Ví dụ với 4 GPU, TP=2 DP=2:

- Rank 0 và 1 thuộc TP group 0. Rank 2 và 3 thuộc TP group 1.
- Rank 0 và 2 thuộc DP group 0. Rank 1 và 3 thuộc DP group 1.

Khi all-reduce TP, rank 0 chỉ giao tiếp với rank 1. Khi all-reduce DP (cho FSDP), rank 0 chỉ giao tiếp với rank 2.

Cách thủ công tạo group:

```python
tp_group = dist.new_group(ranks=[0, 1])
dp_group = dist.new_group(ranks=[0, 2])
```

Cách hiện đại: dùng `DeviceMesh`, sẽ thấy ở chương sau.

## Biến môi trường quan trọng

Khi debug, biết các biến môi trường này giúp ích nhiều.

`RANK`: rank toàn cục của process (0 tới WORLD_SIZE-1).

`LOCAL_RANK`: rank trong node (0 tới GPU-per-node minus 1). Dùng cho `cuda.set_device`.

`WORLD_SIZE`: tổng số process.

`MASTER_ADDR`, `MASTER_PORT`: địa chỉ rank 0 để các rank khác tìm tới khi init. Cho multi-node.

`NCCL_DEBUG=INFO`: bật log NCCL chi tiết, rất hữu ích để xem topology và chọn algorithm.

## Một sai lầm phổ biến: `cuda.set_device`

Một sai lầm hay gặp là quên `torch.cuda.set_device(local_rank)`. Khi đó mọi tensor tạo trên `cuda` mặc định đi về GPU 0, tất cả process tranh nhau GPU 0, các GPU khác idle. Lỗi này im lặng, chỉ thấy chậm bất thường.

Dòng `torch.cuda.set_device(rank % torch.cuda.device_count())` (hoặc dùng `LOCAL_RANK`) phải đứng ngay sau `init_process_group`.

Trong Lightning, việc này được lo. Khi viết code custom, đừng quên.

## Một sai lầm phổ biến khác: deadlock collective

Collective phải được gọi **theo cùng thứ tự** trên mọi rank. Nếu rank 0 gọi `all_reduce(A)` rồi `all_reduce(B)`, mà rank 1 gọi `all_reduce(B)` rồi `all_reduce(A)`, hai bên sẽ chờ nhau vĩnh viễn. Đây là deadlock.

Tương tự, nếu có một rank skip một collective vì điều kiện rẽ nhánh, mọi rank khác sẽ treo. Để tránh: mọi rank phải đi qua cùng path collective.

Đây là lý do code TP thường tránh `if rank == 0`. Mọi rank chạy cùng logic, chỉ khác dữ liệu.

## Sanity check trước khi train

Trước khi chạy train script thật, một sanity check nhỏ:

```python
import torch.distributed as dist
dist.init_process_group(backend="nccl")
rank = dist.get_rank()
torch.cuda.set_device(rank)
t = torch.tensor([rank], device="cuda")
dist.all_reduce(t)
print(f"rank {rank} got {t.item()}")
dist.destroy_process_group()
```

Chạy với `torchrun --nproc_per_node=4 sanity.py`. Kỳ vọng mọi rank in `got 6` (vì 0+1+2+3=6). Nếu khác, có vấn đề về setup.

Đến đây bạn đã hiểu nền móng. Chương sau ta lên `DeviceMesh`, tầng trừu tượng tự nhiên hơn cho TP.
