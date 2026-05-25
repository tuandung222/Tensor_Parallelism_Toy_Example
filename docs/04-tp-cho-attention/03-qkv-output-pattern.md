---
title: Pattern QKV-Column và Output-Row
---

# Pattern QKV-Column và Output-Row

Chương trước đã phát biểu pattern. Chương này lý giải vì sao pattern là **duy nhất** hợp lệ và phân tích các lựa chọn thay thế xem dở chỗ nào, để bạn đặt câu hỏi đúng khi gặp kiến trúc lạ.

## Bốn vai trò trong Attention

Một block Self-Attention chuẩn có bốn linear quan trọng, và ta cần đặt placement cho cả bốn:

- $W_Q, W_K, W_V$: ba "projection" tạo Query, Key, Value từ $X$.
- $W_O$: "output projection" gom kết quả các head và đưa trở về model dim.

Ba projection đầu giống nhau về vai trò shape: input $(B, K)$, output $(B, K)$, đứng đồng song song. Bốn lựa chọn shard cho mỗi cái: Column, Row, Replicate, hoặc Shard chiều khác.

## Loại bỏ Replicate

Nếu ta để $W_Q, W_K, W_V$ replicate (không shard) thì không tiết kiệm được bộ nhớ. Mục tiêu của TP là giảm parameter mỗi rank, replicate đi ngược mục tiêu. Loại.

Tương tự nếu để $W_O$ replicate: tiết kiệm chỉ ở Q/K/V mà $W_O$ vẫn full, sẽ chiếm $1/4$ tham số attention không giảm. Cũng loại.

## Loại bỏ Row cho QKV

Nếu $W_Q$ shard hàng, ta cần input shard cột để Row Parallel hoạt động. Nhưng $X$ đến từ residual của block trước, đang là replicate. Buộc replicate thành shard cột tốn một scatter. Tệ hơn, sau $W_Q$ Row, ta được tensor partial trên chiều output. Tensor partial này không có ý nghĩa "head shard" vì các head bị trộn trên các rank. Để dùng nó trong attention, lại phải all-reduce thành tensor đầy đủ. Hai collective không cần.

Hơn nữa, Row trên $W_Q$ phá vỡ cấu trúc head tự nhiên. Mỗi head là một block cột liên tiếp của $W_Q$. Shard hàng của $W_Q$ chia mỗi head theo head_dim, tức "chia head ra làm $P$ phần dim". Như đã chứng minh ở chương trước, shard head_dim không tương thích với softmax. Loại.

## Loại bỏ Column cho $W_O$

Nếu $W_O$ shard cột, input phải replicate. Nhưng output của attention block các head là tensor đang shard theo head (mỗi rank giữ kết quả của $H/P$ head riêng). Để cho $W_O$ Col Parallel hoạt động, ta phải all-gather các head thành tensor đầy đủ. Đây là một collective không cần thiết vì tổng kết quả tự nhiên muốn là Row Parallel.

Sau $W_O$ Col, output shard theo chiều output. Để biến thành replicate cho block kế tiếp, lại all-gather. Hai all-gather thay vì một all-reduce. Tệ hơn nữa, all-gather đắt hơn all-reduce trên cùng kích thước tensor trong nhiều cấu hình NCCL.

Loại trừ bằng phép đếm: Row là lựa chọn tự nhiên cho $W_O$ vì input của nó đang shard cuối theo head, và output cần replicate.

## Vậy pattern duy nhất là Col-Col-Col-Row

Tóm lại sau khi loại các phương án dở:

- $W_Q$: Column (Row làm hỏng cấu trúc head)
- $W_K$: Column (cùng lý do)
- $W_V$: Column (cùng lý do)
- $W_O$: Row (Col bắt phải all-gather không cần)

Đây là pattern Megatron, và nó là **duy nhất** tối ưu cho cấu trúc Multi-Head Attention chuẩn.

## So sánh số collective

| Pattern | Forward collective | Backward collective |
|---------|---------------------|----------------------|
| **Col-Col-Col-Row** (Megatron) | 1 AllReduce | 1 AllReduce |
| Row-Row-Row-Col | 1 Scatter + 1 AllReduce + 1 AllGather + ... | nhiều hơn |
| Col-Col-Col-Col | 1 AllGather (giả định) | nhiều hơn |
| Mixed bất hợp lý | softmax sai | sai |

Megatron pattern không chỉ tối thiểu collective mà còn là duy nhất giữ đúng ngữ nghĩa softmax.

## Hiệu quả bộ nhớ và compute

Một MHA block với $K = 4096$, $H = 32$, $d_k = 128$ trên $P = 8$ rank:

| Đại lượng | Baseline | Với TP=8 |
|-----------|----------|-----------|
| Parameter mỗi rank | $4 \cdot 4096^2 = 67$ M | $67 / 8 \approx 8.4$ M |
| Q/K/V tensor per rank | $B \cdot S \cdot K = B \cdot S \cdot 4096$ | $B \cdot S \cdot 512$ |
| Score matrix per rank | $B \cdot H \cdot S^2 = B \cdot 32 \cdot S^2$ | $B \cdot 4 \cdot S^2$ |
| FLOPs per rank (compute) | $\Theta(B S^2 K)$ | $\Theta(B S^2 K / 8)$ |

Tổng giao tiếp một block một step (forward + backward): $2 \cdot B \cdot S \cdot K = 2 \cdot B \cdot S \cdot 4096$ phần tử AllReduce. Đây là chi phí trao đổi giữa compute và bộ nhớ, một sự đổi chác mà ở $P \le 8$ trên NVLink thường lãi to.

## Liên hệ tới `02_large_language_model/parallelism.py`

Trong file `parallelism.py` của repo, plan attention cho mỗi block là:

```python
plan = {
    "attention": PrepareModuleInput(
        input_layouts=(Shard(1), None),
        desired_input_layouts=(Replicate(), None),
    ),
    "attention.wq": ColwiseParallel(),
    "attention.wk": ColwiseParallel(),
    "attention.wv": ColwiseParallel(),
    "attention.wo": RowwiseParallel(output_layouts=Shard(1)),
    ...
}
```

Đây là pattern Col-Col-Col-Row đúng như ta vừa chứng minh. Hai chỗ đặc biệt:

`PrepareModuleInput` trên `"attention"`: nó nhận tensor đang shard theo chiều sequence (do Sequence Parallel norm phía trước, Phần 6 sẽ giải thích) và biến thành tensor replicate để feed vào QKV linear. Đây là một all-gather ngầm.

`RowwiseParallel(output_layouts=Shard(1))`: thay vì output replicate, output được shard lại theo chiều sequence. Đây là một reduce-scatter (kết hợp của all-reduce và scatter), tiết kiệm so với all-reduce rồi scatter riêng. Lý do dùng Shard(1) liên quan tới Sequence Parallel, ta sẽ derive ở Phần 6.

Tạm thời, hãy nhận diện: bốn linear vẫn theo đúng pattern Col-Col-Col-Row. Các option bên ngoài chỉ điều chỉnh placement của input/output để khớp với Sequence Parallel zones.

## Dòng `n_heads //= tp_size`

Trong cùng file:

```python
attn_layer.n_heads = attn_layer.n_heads // tp_mesh.size()
attn_layer.n_kv_heads = attn_layer.n_kv_heads // tp_mesh.size()
```

Dòng này điều chỉnh attribute trong `Attention` module sau khi TP đã được áp. Lý do: trong `Attention.forward`, code làm:

```python
xq = xq.view(bs, seqlen, self.n_heads, self.head_dim)
```

Sau TP, tensor `xq` đã được shard theo chiều head, local shape có $H/P$ head thay vì $H$. Vì vậy `self.n_heads` phải được giảm tương ứng để `view` ra đúng shape local. Không chỉnh dòng này, code sẽ crash với lỗi shape mismatch.

Đây là một trong những "sửa nhỏ thủ công" cần làm khi áp TP lên kiến trúc Llama. `parallelize_module` lo việc shard weight, nhưng không tự biết các attribute liên quan đến shape đang được dùng đâu trong forward. Bạn phải chỉnh tay.

Chương tiếp theo ta sang trường hợp GQA/MQA, nơi $n_{kv\_heads} < n_{heads}$, và phép `repeat_kv` xen vào giữa.
