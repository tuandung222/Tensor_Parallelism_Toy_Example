---
title: Walkthrough 01_simple_model/train.py
---

# Walkthrough `01_simple_model/train.py`

Đây là điểm bạn chuyển từ lý thuyết sang code thật. File chỉ dưới 80 dòng, nhưng mỗi nhóm dòng đều ánh xạ vào một khái niệm ta đã derive ở Phần 1, Phần 2, và hai chương đầu Phần 3. Ta đi từ trên xuống dưới, dừng lại tại mỗi đoạn có ý nghĩa.

## Import block

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel
from torch.distributed.tensor.parallel import parallelize_module

import lightning as L
from lightning.pytorch.demos.boring_classes import RandomDataset
from lightning.pytorch.strategies import ModelParallelStrategy
```

Hai dòng quan trọng nhất là dòng 5 và dòng 6.

`ColwiseParallel` và `RowwiseParallel` là hai lớp "style". Mỗi instance không phải module, mà là một **bản kế hoạch nhỏ** mô tả cách wrap một `nn.Linear` thường thành phiên bản TP. Khi `parallelize_module` được gọi, nó nhận từ điển `{tên submodule : style}` và áp dụng từng style lên submodule tương ứng. Style sẽ làm những việc sau với linear đó: shard tensor `weight` thành DTensor với placement phù hợp, wrap forward để đảm bảo input/output có placement đúng (nếu cần thì chèn collective), và đăng ký hook cho backward.

`parallelize_module` là entry point chuẩn của PyTorch cho khai báo TP. Nó nhận `(module, device_mesh, plan)` và trả về module mới (in-place trên parameter).

Lightning's `ModelParallelStrategy` tự dựng `device_mesh` cho bạn, bao gồm chiều `"tensor_parallel"` và `"data_parallel"`. Trong ví dụ này ta chỉ dùng chiều `"tensor_parallel"`.

## Định nghĩa `FeedForward`

```python
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
```

Đây là SwiGLU dạng tối giản. Lưu ý ba điểm.

Thứ nhất, ba linear `w1, w2, w3` đặt tên trùng đúng với convention Llama. Nếu sau này bạn áp dụng cùng plan cho FFN của Llama-3 (như trong `02_large_language_model/`), bạn copy-paste plan được luôn.

Thứ hai, `bias=False` không phải tình cờ. Bias làm pattern TP phức tạp thêm: bias của Column Parallel phải shard, còn bias của Row Parallel chỉ áp ở rank 0 (hoặc replicate trước all-reduce). Llama dùng RMSNorm để chuẩn hóa, nên bias trong linear bị bỏ. Đây là một trong những đơn giản hóa giúp TP gọn hơn.

Thứ ba, `dim = hidden_dim = 8192` trong ví dụ. Đây là kích thước thật, không phải toy. Một linear $8192 \times 8192$ float32 chiếm 256 MB. Có 3 linear, tổng tham số 768 MB cho riêng FFN này. Trên một GPU 24 GB, đó là một con số đáng kể. TP chia 4 sẽ giảm xuống 192 MB mỗi rank, cho riêng tham số.

## `LitModel.__init__` và `configure_model`

```python
class LitModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = FeedForward(8192, 8192)

    def configure_model(self):
        if self.device_mesh is None:
            return

        tp_mesh = self.device_mesh["tensor_parallel"]
        plan = {
            "w1": ColwiseParallel(),
            "w2": RowwiseParallel(),
            "w3": ColwiseParallel(),
        }
        parallelize_module(self.model, tp_mesh, plan)
```

`__init__` tạo `FeedForward` trên CPU (hoặc meta device tùy strategy). Tham số chưa shard ở đây, đó vẫn là `nn.Linear` thường.

`configure_model` là hook do Lightning gọi sau khi mesh đã sẵn sàng. Đây là nơi ta khai báo plan TP. Trình tự:

- Lấy sub-mesh chiều `"tensor_parallel"`. Nếu world có 4 GPU và ta chỉ dùng TP, mesh sẽ là 1D có 4 phần tử.
- Định nghĩa plan: ba key là tên submodule trong `self.model`, ba value là style. Đây chính là plan đã derive ở chương SwiGLU: `w1, w3` Col, `w2` Row.
- Gọi `parallelize_module`. Sau khi gọi, mỗi linear có `weight` là DTensor với placement tương ứng, và mỗi `forward` được wrap để xử lý collective.

Một chi tiết tinh tế: thứ tự key trong plan không quan trọng. PyTorch áp dụng style cho mỗi key độc lập, dựa vào tên `dotted` của submodule. Quan trọng là tên phải khớp với cấu trúc module.

## Trạng thái sau `parallelize_module`

Sau dòng `parallelize_module`, hãy hình dung trạng thái như sau (giả sử 4 GPU, mỗi rank tên $r \in \{0, 1, 2, 3\}$):

| Module | Tham số trên rank $r$ | Shape | Placement |
|--------|------------------------|-------|-----------|
| `w1.weight` | $W_1^{(r)}$ | $(2048, 8192)$ (transpose so với toán) | Shard(0) trên DTensor; nghĩa toán: cột $r$ |
| `w2.weight` | $W_2^{(r)}$ | $(8192, 2048)$ | Shard(1); nghĩa toán: hàng $r$ |
| `w3.weight` | $W_3^{(r)}$ | $(2048, 8192)$ | Shard(0); nghĩa toán: cột $r$ |

Lưu ý quan trọng: `nn.Linear` lưu weight theo convention `(out_features, in_features)`, ngược với phong cách toán $(K, H)$. Khi ColwiseParallel "shard cột" theo nghĩa toán, trên tensor PyTorch nó shard chiều 0 (chiều `out_features`). Tương tự, RowwiseParallel "shard hàng" theo nghĩa toán nhưng trên tensor PyTorch là shard chiều 1 (chiều `in_features`). Đừng nhầm hai convention này khi đọc log debug.

## Forward trên DTensor

Khi `self.model(x)` chạy:

- `x` là tensor thường (Lightning đã đảm bảo `x` giống nhau trên mọi rank trong TP group, xem phần `train_dataloader`). Khi đi vào `self.w1` (Column Parallel), input được tự động đối xử như Replicate trên TP mesh.
- `self.w1(x)` tạo DTensor shard(-1), local shape $(B, H/P)$.
- `F.silu(...)` chạy element-wise trên DTensor, kết quả vẫn shard(-1).
- `self.w3(x)` tương tự, ra shard(-1).
- Phép `*` (element-wise multiply) giữa hai DTensor shard cùng chiều: PyTorch nhận ra hai placement match, thực hiện multiply local, không sinh collective.
- `self.w2(...)` (Row Parallel) nhận input shard(-1), tính local matmul, ra partial. PyTorch tự chèn `AllReduce` để biến partial thành replicate. Output cuối là DTensor replicate, local shape $(B, K)$.

Toàn bộ chuỗi trên là **một** collective duy nhất (all-reduce ở cuối), đúng như lý thuyết.

## `training_step` và backward

```python
def training_step(self, batch):
    output = self.model(batch)
    loss = output.sum()
    return loss
```

Hàm loss tổng quát hóa nhất có thể: `output.sum()`. Vì output đã replicate sau forward, `sum()` trên DTensor replicate cho cùng kết quả trên mọi rank, không cần collective thêm.

Backward được Lightning gọi tự động trên `loss`. Lúc này:

- Gradient của `output` là tensor toàn 1 (vì $\partial \mathrm{sum}/\partial y_{ij} = 1$), replicate.
- Backward qua $W_2$ Row Parallel: identity, gradient với hidden vẫn shard(-1) như forward.
- Backward qua element-wise multiply và SiLU: element-wise, tự nhiên trên shard.
- Backward qua $W_1$ Column Parallel: gradient với input của $w_1$ là partial trên TP mesh. PyTorch chèn `AllReduce` để biến thành replicate (hoặc giữ partial, tùy `output_layouts` cấu hình). Trong ví dụ mặc định, gradient input cũng replicate.

Tổng cộng forward + backward: hai all-reduce trên tensor $(B, K) = (8, 8192)$ trong một step. Chi phí truyền $\approx 8 \times 8192 \times 4\,\text{bytes} = 256\,\text{KB}$ mỗi all-reduce mỗi forward. Với NCCL nội node trên NVLink, đây là tiếng động nhỏ, không phải bottleneck.

## `train_dataloader` và sampler

```python
def train_dataloader(self):
    dataset = RandomDataset(8192, 64)
    return torch.utils.data.DataLoader(dataset, batch_size=8, num_workers=2)
```

Một điểm thường gây bối rối lần đầu: trong TP, **mọi rank phải nhận cùng input batch**. Lý do: $X$ phải replicate trên TP mesh. Nếu rank 0 và rank 1 nhận hai batch khác nhau, kết quả $X W_1^{(0)}$ và $X W_1^{(1)}$ không phải là hai shard của cùng một $Y$, all-reduce ở cuối sẽ trộn lung tung và loss không có ý nghĩa.

Lightning xử lý chuyện này: với `ModelParallelStrategy`, sampler được tự cấu hình sao cho mọi rank trong TP group nhận đúng cùng minibatch. Nếu bạn dùng 2D parallelism (TP + DP), thì các rank cùng TP group nhận batch giống nhau, các rank khác DP group nhận batch khác nhau. Lightning lo việc này tự động khi bạn dùng `ModelParallelStrategy`.

Nếu bạn viết training loop bằng tay không dùng Lightning, bạn phải tự cài sampler đảm bảo điều kiện trên. Đây là một trong những nguồn bug rất khó tìm. Cách đơn giản: dùng `DistributedSampler` với `num_replicas = dp_size` (không phải `world_size`) và `rank = dp_rank` (không phải `global_rank`).

## `strategy` và `Trainer`

```python
strategy = ModelParallelStrategy()
trainer = L.Trainer(
    accelerator="cuda",
    devices=4,
    strategy=strategy,
    max_epochs=1,
    logger=False,
    enable_checkpointing=False,
)
```

`ModelParallelStrategy` mặc định dùng `tensor_parallel_size = world_size`, nghĩa là 4 GPU đều thuộc một TP group duy nhất, không có DP. Nếu bạn muốn 2D (ví dụ 2 DP $\times$ 2 TP), truyền `tensor_parallel_size=2`.

`logger=False` và `enable_checkpointing=False` chỉ để demo gọn. Trong thực tế bạn cần checkpoint, nhưng checkpoint một model TP cần lưu cả device mesh state, ta sẽ bàn ở Phần 9.

## Đo bộ nhớ peak

```python
trainer.print(f"Peak memory usage: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")
```

`trainer.print` chỉ in trên rank 0. `torch.cuda.max_memory_allocated()` cho biết peak GPU memory đã cấp phát trong process. Đây là cách thực dụng để verify TP đang hoạt động: với và không có TP, peak memory phải khác nhau theo đúng tỉ số $P$ trên phần tham số.

Bài tập cho bạn: chạy script này hai lần, một lần với `devices=4` (TP=4), một lần với `devices=1` (không TP). So sánh peak memory. Bạn sẽ thấy tỉ số bộ nhớ tham số gần 4 lần, còn bộ nhớ activation cũng giảm rõ rệt.

## Tổng kết walkthrough

Bạn vừa đọc một file 80 dòng, nhưng dưới mặt nước có toàn bộ Phần 1, Phần 2, và hai chương đầu Phần 3.

Mỗi dòng `parallelize_module(... plan ...)` thay thế cho hàng trăm dòng code thấp hơn về scatter, gather, all-reduce, hook backward. PyTorch DTensor là abstraction tốt, nhưng nếu bạn không hiểu lớp dưới, bạn sẽ bí khi gặp lỗi như "placement mismatch" hay "expected Replicate but got Partial".

Sau Phần 3 này, bạn đã đủ vũ khí để bước vào Self-Attention (Phần 4), nơi pattern tương tự nhưng có thêm chiều "head" làm phong phú thêm cách shard.
