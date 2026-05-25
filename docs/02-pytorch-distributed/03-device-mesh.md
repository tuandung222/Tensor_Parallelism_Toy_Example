---
title: DeviceMesh, grid logic của rank
---

# DeviceMesh, grid logic của rank

`DeviceMesh` là khái niệm trung tâm trong PyTorch Distributed hiện đại. Nó cho phép bạn tổ chức rank thành grid có chiều và tên, từ đó mọi pattern parallelism phức tạp (TP, DP, FSDP, kết hợp) trở nên gọn gàng.

## Một mesh 1D đơn giản

Cho 4 GPU. Tạo một mesh 1D với chiều `tensor_parallel`:

```python
from torch.distributed.device_mesh import init_device_mesh

mesh = init_device_mesh("cuda", (4,), mesh_dim_names=("tensor_parallel",))
```

Mesh này là một mảng 1D 4 phần tử: rank 0, 1, 2, 3 thuộc chiều `tensor_parallel`. Mọi collective trên mesh này đều xảy ra giữa cả 4 rank.

Truy cập subgroup:

```python
tp_group = mesh["tensor_parallel"]
```

`tp_group` là một ProcessGroup chỉ chứa các rank trong chiều `tensor_parallel`. Với mesh 1D 4 chiều, đó là cả 4 rank. Khi mesh 2D, mỗi chiều cho một group khác nhau.

## Một mesh 2D cho TP + DP

Cho 4 GPU, ta muốn TP=2 và DP=2. Mesh 2D:

```python
mesh = init_device_mesh(
    "cuda", (2, 2), mesh_dim_names=("data_parallel", "tensor_parallel"),
)
```

Sơ đồ:

```
                  tensor_parallel
                   col 0   col 1
data_parallel ┌──────────────────┐
        row 0 │  rank 0   rank 1 │
        row 1 │  rank 2   rank 3 │
              └──────────────────┘
```

Đọc theo chiều:

- Chiều `tensor_parallel` (cột): nhóm 0 gồm `{0, 2}`, nhóm 1 gồm `{1, 3}`. Mỗi cột là một TP group.
- Chiều `data_parallel` (hàng): nhóm 0 gồm `{0, 1}`, nhóm 1 gồm `{2, 3}`. Mỗi hàng là một DP group.

Khi all-reduce TP trên `mesh["tensor_parallel"]`, rank 0 chỉ giao tiếp với rank 2 (cùng cột 0). Rank 1 chỉ với rank 3. Đúng đối tác.

Khi all-reduce DP (cho FSDP) trên `mesh["data_parallel"]`, rank 0 chỉ với rank 1 (cùng hàng 0). Đúng đối tác cho DP.

Đây là trái tim của 2D parallelism. Hai chiều độc lập, mesh phân loại rõ rank nào thuộc nhóm nào trong mỗi chiều.

## Cách Lightning tạo mesh tự động

Trong toy `01_simple_model/train.py`:

```python
class LitModel(L.LightningModule):
    def configure_model(self):
        if self.device_mesh is None:
            return
        tp_mesh = self.device_mesh["tensor_parallel"]
        plan = {...}
        parallelize_module(self.model, tp_mesh, plan)
```

`self.device_mesh` được Lightning tự tạo theo `ModelParallelStrategy`. Bạn chỉ cần truy cập `["tensor_parallel"]` để lấy subgroup TP.

Trong `02_large_language_model/train.py`:

```python
strategy = ModelParallelStrategy(
    data_parallel_size=data_parallel_size,
    tensor_parallel_size=tensor_parallel_size,
)
trainer = L.Trainer(accelerator="cuda", devices=4, strategy=strategy, ...)
```

Lightning tạo mesh 2D với hai chiều `data_parallel` và `tensor_parallel`. Trong `parallelism.py`:

```python
def parallelize(model: Transformer, device_mesh: DeviceMesh) -> Transformer:
    dp_mesh = device_mesh["data_parallel"]
    tp_mesh = device_mesh["tensor_parallel"]
    if tp_mesh.size() > 1:
        # áp dụng TP plan
        ...
    if dp_mesh.size() > 1:
        # áp dụng FSDP
        ...
```

Pattern này rất sạch: tách hai chiều ra, mỗi chiều áp dụng độc lập.

## Vì sao đặt tên thay vì index

Một câu hỏi: tại sao không dùng `mesh[0]`, `mesh[1]` mà phải `mesh["data_parallel"]`, `mesh["tensor_parallel"]`.

Câu trả lời: code đọc-được. Khi mesh 3D (TP + PP + DP) hoặc 4D (TP + PP + DP + SP), index số trở nên khó hiểu. Tên có ngữ nghĩa rõ ràng. Một code review viên có thể đọc `dp_mesh = mesh["data_parallel"]` và biết ngay đây là chiều nào, không phải đoán.

Convention đặt tên trong thực tế:

- `tp` hoặc `tensor_parallel` cho TP.
- `dp` hoặc `data_parallel` cho DP/FSDP.
- `pp` hoặc `pipeline_parallel` cho PP.
- `sp` hoặc `sequence_parallel` cho SP (nếu tách riêng, hiếm vì SP thường gộp với TP).

Khi viết code TP riêng, hãy theo convention này.

## Mesh dim size

`mesh.size("tensor_parallel")` trả về số rank trong chiều `tensor_parallel`. Đây là `P` mà Phần 1 dùng. Khi bạn cần biết “TP size hiện tại là bao nhiêu” trong code, hỏi mesh.

Ví dụ trong `parallelism.py`:

```python
attn_layer.n_heads = attn_layer.n_heads // tp_mesh.size()
attn_layer.n_kv_heads = attn_layer.n_kv_heads // tp_mesh.size()
```

Khi shard attention, mỗi rank giữ một subset của head. Tổng head chia cho TP size. Đây là lý do `n_heads` phải chia hết TP size, đã nói ở chương 4 Phần 0.

## Mesh có thể có chiều kích thước 1

Trường hợp đặc biệt: TP=1 hoặc DP=1. Mesh vẫn có chiều đó, nhưng kích thước 1. Code thường có:

```python
if tp_mesh.size() > 1:
    # apply TP plan
```

Khi `tp_mesh.size() == 1`, không cần apply, vì TP=1 nghĩa là không TP. Lightning vẫn tạo mesh để code đường đi không phải `if device_mesh is None`. Pattern giúp code đồng nhất giữa các config.

## DeviceMesh và DTensor

DeviceMesh là **không gian** mà DTensor sống trên đó. Một DTensor luôn được gắn với một mesh và có placement chỉ định nó shard trên chiều nào, replicate trên chiều nào.

Hiểu thế này: DeviceMesh là "lưới điện", DTensor là "thiết bị cắm vào lưới". Chương sau đi vào DTensor.

## Kiểm tra trí nhớ

Một, mesh 2D `(2, 2)` với tên `("dp", "tp")`, rank 0 thuộc TP group nào và DP group nào.

Hai, khi gọi all-reduce trên `mesh["tp"]`, communication có scale với DP size không. Vì sao.

Ba, nếu bạn có 8 GPU và muốn TP=4, FSDP=2, mesh shape gì.

Đáp số: rank 0 ở TP group cột 0 `{0, 2}` và DP group hàng 0 `{0, 1}`. All-reduce trên TP không liên quan DP, communication chỉ giữa rank cùng TP. Mesh shape `(2, 4)` với tên `("data_parallel", "tensor_parallel")`.
