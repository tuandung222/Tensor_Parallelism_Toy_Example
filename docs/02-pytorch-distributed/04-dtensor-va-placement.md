---
title: DTensor và Placement
---

# DTensor và Placement

`DTensor` (Distributed Tensor) là cầu nối giữa toán Phần 1 và code PyTorch. Một DTensor là một tensor có metadata "tôi đang phân bố như thế nào trên mesh". Có ba loại metadata cốt lõi, gọi là `Placement`: `Replicate`, `Shard`, `Partial`.

## DTensor là gì

Hình dung: bạn có tensor logic shape $(B, K)$. Nó đáng lý nằm trên một GPU, nhưng bạn muốn phân bố trên 4 GPU. DTensor là **một đối tượng Python duy nhất** trên mỗi rank, đại diện cho tensor logic, mà bên trong nó tự biết mình đang giữ phần nào của tensor đầy đủ.

Khi bạn gọi `dt.to_local()`, bạn lấy tensor cục bộ thực tế trên rank này. Khi bạn gọi phép tính trên DTensor, PyTorch tự sinh collective nếu cần.

DTensor có ba thuộc tính chính:

- `device_mesh`: mesh mà nó sống trên.
- `placements`: một tuple, mỗi phần tử cho biết shard/replicate trên chiều mesh nào.
- `shape`: shape **logic đầy đủ**, không phải shape local.

## Ba loại Placement

### Replicate

`Replicate()` nghĩa là mọi rank trong chiều mesh tương ứng đều giữ **bản sao đầy đủ** của tensor. Không có shard.

Ví dụ: mesh 1D `(4,)`, tensor shape `(B, K)` với placement `(Replicate(),)`. Mỗi rank giữ tensor `(B, K)` đầy đủ. Tổng bộ nhớ 4 lần kích thước tensor.

Trong TP, input của Column Parallel có placement `Replicate()`. Output của Row Parallel (sau all-reduce) cũng `Replicate()`.

### Shard(dim)

`Shard(dim)` nghĩa là tensor được cắt theo chiều `dim` thành $P$ phần, mỗi rank giữ một phần.

Ví dụ: mesh `(4,)`, tensor logic shape `(B, K)` với placement `(Shard(1),)`. Tensor được cắt theo dim 1 ($K$) thành 4 phần. Mỗi rank giữ tensor local shape `(B, K/4)`.

Chú ý: dim được index từ 0 trên shape của tensor. `Shard(0)` là shard chiều đầu (thường là batch), `Shard(-1)` là shard chiều cuối (thường là output dim).

Trong TP:

- Param của Column Parallel có placement `Shard(0)` (shard chiều output, vì PyTorch shape `(d_out, d_in)`).
- Param của Row Parallel có placement `Shard(1)` (shard chiều input).
- Output Column có placement `Shard(-1)` (shard chiều cuối của output).
- Input Row có placement `Shard(-1)` (shard chiều cuối của input).

### Partial

`Partial()` là placement đặc biệt và quan trọng nhất để hiểu. Nó nghĩa là mỗi rank giữ một **partial sum** của tensor đầy đủ.

Cụ thể: tensor logic là $T = \sum_r T_r$, mỗi rank $r$ giữ $T_r$. Cả $T$ và $T_r$ cùng shape. Nhưng giá trị trên mỗi rank là một mảnh của tổng.

`Partial` xuất hiện trong forward Row Parallel: sau matmul, mỗi rank có một partial của output. Để chuyển từ Partial sang Replicate, PyTorch tự sinh **all-reduce** với phép cộng.

Đây là vẻ đẹp của abstraction: bạn không phải nghĩ collective, chỉ nghĩ placement. PyTorch lo chuyển.

## Bảng chuyển đổi giữa các Placement

| Từ | Sang | Collective |
|---|---|---|
| Replicate | Shard(d) | Split (không cần network, chỉ lấy local slice) |
| Shard(d) | Replicate | All-gather |
| Partial | Replicate | All-reduce |
| Partial | Shard(d) | Reduce-scatter |
| Shard(d) | Shard(d') khác | All-gather + Split (đắt) |

Bảng này nội hóa được sẽ giúp bạn đọc mọi placement trong code mà không cần ghi nhớ collective riêng.

## Code ví dụ với DTensor

```python
import torch
from torch.distributed._tensor import DTensor, Replicate, Shard
from torch.distributed.device_mesh import init_device_mesh

mesh = init_device_mesh("cuda", (4,), mesh_dim_names=("tp",))

# Tạo một tensor local trên mỗi rank, tổng hợp thành DTensor logic
local = torch.randn(8, 1024, device="cuda")  # mỗi rank có shape (8, 1024)
dt = DTensor.from_local(local, mesh, placements=[Shard(1)])
print(dt.shape)  # (8, 4096), shape logic đầy đủ
print(dt.to_local().shape)  # (8, 1024), shape local

# Chuyển sang Replicate, sinh all-gather
dt_full = dt.redistribute(placements=[Replicate()])
print(dt_full.to_local().shape)  # (8, 4096) trên mọi rank
```

`from_local` ngược lại: bạn nói rank này giữ phần này, mesh và placement là gì, PyTorch tạo DTensor logic.

`redistribute` chuyển placement, sinh collective tự động.

`to_local` lấy tensor local thực tế của rank này.

## Placement trong toy code

Quay lại `02_large_language_model/parallelism.py`:

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
```

Đọc từng dòng.

`tok_embeddings` là `RowwiseParallel` với `input_layouts=Replicate()`. Token ID input là replicate (mọi rank có cùng input). Param của embedding shard theo hàng (vocab dim). Output sẽ là Partial sau matmul, tự động chuyển sang Replicate qua all-reduce. Phần 5 sẽ phân tích kỹ.

`output` là `ColwiseParallel` với `input_layouts=Shard(1)` và `output_layouts=Shard(-1)`. Input là shard dim 1 (sequence dim, vì norm trước đó là Sequence Parallel). Output shard dim cuối (vocab). `use_local_output=False` để giữ DTensor (không gọi `to_local`), vì loss tiếp theo dùng `loss_parallel` cần biết shard.

`norm` là `SequenceParallel`. Norm thực hiện trên activation đã shard theo sequence dim, dùng `LayerNorm` trên local part và collective khi cần.

`layers.0` dùng `PrepareModuleInput` để chuyển input từ `Replicate()` (đầu vào của layer đầu tiên) sang `Shard(1)` (sequence shard cho Sequence Parallel). Đây là điểm "khâu" giữa hai pattern khác nhau.

Mọi dòng này đều có thể đọc theo placement, không cần đoán.

## Một quan sát thực tế: `use_local_output`

Một số API có flag `use_local_output=True/False`. Khi True, output của parallelize trả về tensor local (gọi `to_local` tự động). Khi False, giữ là DTensor.

Khi tiếp tục pattern TP, để DTensor (False) là hợp lý. Khi cần đưa cho code không-TP-aware (ví dụ user-defined loss thuần PyTorch), gọi True để có tensor thường.

## Kiểm tra trí nhớ

Một, sau forward Row Parallel với input Shard(-1), output có placement gì trước khi all-reduce.

Hai, nếu một DTensor có placement Shard(0) và bạn redistribute sang Replicate, collective gì xảy ra.

Ba, trong PyTorch shape `(d_out, d_in)` của nn.Linear, Column Parallel shard param theo dim nào, Row Parallel shard dim nào.

Đáp số: Partial. All-gather. Column shard dim 0 (output), Row shard dim 1 (input).
