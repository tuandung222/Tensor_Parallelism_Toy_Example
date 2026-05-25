---
title: Mesh 2D và luồng gradient
---

# Mesh 2D và luồng gradient

Khi áp cả TP (intra-layer) và FSDP (data parallel) đồng thời, ta cần một DeviceMesh 2 chiều. Chương này phân tích cấu trúc mesh đó, cách parameter và gradient di chuyển trên hai chiều, và các điểm tinh tế khi hai cơ chế gặp nhau.

## Khai báo mesh 2D

```python
from torch.distributed.device_mesh import init_device_mesh

device_mesh = init_device_mesh(
    "cuda",
    (dp_size, tp_size),
    mesh_dim_names=("data_parallel", "tensor_parallel"),
)
```

Với `world_size = dp_size * tp_size = 4 * 2 = 8` rank, mesh là 2D shape $(4, 2)$. Rank $r$ có tọa độ $(r \div 2, r \mod 2)$.

Lấy sub-mesh:

```python
tp_mesh = device_mesh["tensor_parallel"]  # 1D mesh, kích thước 2
dp_mesh = device_mesh["data_parallel"]    # 1D mesh, kích thước 4
```

`tp_mesh` gồm các rank cùng hàng (cùng `dp_rank`), tức $\{0,1\}, \{2,3\}, \{4,5\}, \{6,7\}$ là bốn TP group. `dp_mesh` gồm các rank cùng cột (cùng `tp_rank`), tức $\{0,2,4,6\}, \{1,3,5,7\}$ là hai DP group.

## Parameter trên mesh 2D

Một parameter (ví dụ `attention.wq.weight`) sau khi áp cả TP và FSDP có placement:

```
(Shard(0) on DP, Shard(0) on TP)
```

Tức shard cả hai chiều mesh. Mỗi rank giữ một block $\frac{1}{P_{DP}} \cdot \frac{1}{P_{TP}}$ của parameter đầy đủ.

Vế FSDP shard "chiều outer features" (chiều dim 0 của `nn.Linear.weight`, ánh xạ về chiều output features theo cách PyTorch lưu).

Vế TP cũng shard chiều 0 (ColwiseParallel). Hai cấp shard chồng lên cùng một chiều, mỗi cấp chia tiếp.

Cụ thể: nếu $W$ shape $(K, H)$, sau TP ColumnParallel: mỗi TP rank giữ $(K/P_{TP}, H)$. Sau FSDP: mỗi rank (cả TP và DP) giữ $(K / (P_{TP} \cdot P_{DP}), H)$.

## Forward: ai gom với ai

Forward trên một rank $(r_{DP}, r_{TP})$:

1. **FSDP AllGather** trên DP mesh: gom shard parameter của tất cả `dp_size` rank cùng cột (cùng `tp_rank`). Sau bước này, rank có parameter shape $(K/P_{TP}, H)$, tức "TP-only shard".
2. **TP compute**: matmul local với parameter $(K/P_{TP}, H)$, sinh output partial.
3. **TP collective** (AllReduce hoặc ReduceScatter) trên TP mesh: gom kết quả TP-shard.
4. **FSDP free**: ném parameter đầy đủ, giữ lại shard nhỏ.

Hai chiều collective tách biệt: TP collective hoạt động **chỉ** trên TP group, FSDP collective **chỉ** trên DP group. Không trộn.

## Backward đối ngẫu

1. AllGather param (FSDP) trên DP, để có TP-only shard.
2. Compute gradient: partial trên cả hai chiều.
3. AllReduce/ReduceScatter trên TP cho gradient input.
4. ReduceScatter trên DP cho gradient param: cộng và shard lại theo DP.
5. Free parameter đầy đủ.

Toàn bộ flow này được PyTorch quản lý tự động khi bạn khai báo plan TP + FSDP đúng.

## Compose: FSDP đầu, TP sau

Câu hỏi quan trọng: thứ tự áp TP và FSDP có quan trọng không.

Có. Thứ tự đúng:

```python
# 1. Áp TP trước
parallelize_module(model, tp_mesh, tp_plan)

# 2. Sau đó áp FSDP
for block in model.layers.values():
    fully_shard(block, mesh=dp_mesh)
fully_shard(model, mesh=dp_mesh)
```

Lý do: TP biến parameter thành DTensor shard trên TP mesh. FSDP nhận DTensor đó, thêm shard trên DP mesh, biến thành DTensor 2D-shard. Nếu làm ngược, FSDP shard trước rồi TP, TP sẽ không thấy được parameter ở dạng nguyên gốc để biết shard cột hay hàng.

Trong toy code `parallelism.py`, thứ tự đúng được tuân thủ.

## Số collective tổng cộng một block

Per Transformer block per step:

**TP collective** (giả sử SP bật):

- 2 AllGather chiều sequence (PrepareModuleInput).
- 2 ReduceScatter chiều sequence (Row Parallel với output Shard(1)).
- Forward + backward: gấp đôi.

**FSDP collective** (FSDP áp trên block):

- 1 AllGather param forward.
- 1 AllGather param backward.
- 1 ReduceScatter gradient backward.

Tổng collective cho một block một step: $2 \cdot 4 + 3 = 11$. Mỗi collective trên một mesh riêng (TP hoặc DP), nên không xung đột.

Trên NCCL ring, các collective trên các mesh khác nhau có thể chạy đồng thời nếu băng thông cho phép. PyTorch tự lo dispatch.

## Cấu hình thực tế

Cluster phổ biến: 8 GPU mỗi node, nhiều node. NVLink intra-node, InfiniBand inter-node.

NVLink (intra-node) băng thông cao, độ trễ thấp: lý tưởng cho TP (giao tiếp tensor sequence dài).

InfiniBand (inter-node) chậm hơn vài lần: lý tưởng cho FSDP (giao tiếp parameter nhưng có thể overlap với compute).

Nên: TP **intra-node**, FSDP **inter-node**. Với 4 node $\times$ 8 GPU = 32 GPU: TP=8, FSDP=4. Mesh 2D shape $(4, 8)$, `dp_mesh` chứa 4 rank cross-node, `tp_mesh` chứa 8 rank intra-node.

Trong toy code (`02_large_language_model/train.py`), Lightning's `ModelParallelStrategy` với `data_parallel_size="auto", tensor_parallel_size="auto"` áp dụng heuristic này.

## Sai lầm thường gặp

Sai lầm 1: áp FSDP trước TP. Trật tự sai làm TP không thấy được DTensor structure.

Sai lầm 2: dùng mesh 1D cho cả TP và FSDP. Không có sub-mesh để tách collective, hai cơ chế trộn nhau.

Sai lầm 3: TP cross-node, FSDP intra-node. Hiệu năng tệ vì TP có collective lớn trên InfiniBand chậm.

Sai lầm 4: quên `mesh=dp_mesh` khi gọi `fully_shard`. Mặc định FSDP dùng world mesh, sẽ shard cả TP rank lẫn DP rank, trùng với TP shard, trở nên hỗn loạn.

Chương tiếp ta sang mixed precision và activation checkpoint.
