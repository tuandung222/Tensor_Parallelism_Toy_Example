---
title: Column Parallel, derivation đầy đủ
---

# Column Parallel, derivation đầy đủ

Column Parallel là cách shard tự nhiên nhất khi input của bạn đang replicate trên các rank, và bạn chấp nhận output bị shard. Hãy đi qua derivation đầy đủ và phân tích chi phí.

## Setup

Cho linear layer $Y = XW$ với:

- $X \in \mathbb{R}^{B \times K}$, **replicate** trên mọi rank TP. Tức là rank 0, 1, ..., $P-1$ đều có cùng $X$.
- $W \in \mathbb{R}^{K \times N}$, **shard theo cột** thành $P$ phần: $W = [W_0, W_1, \dots, W_{P-1}]$ với $W_r \in \mathbb{R}^{K \times N/P}$.

Mỗi rank $r$ giữ $W_r$.

## Forward

Mỗi rank tính độc lập:

$$
Y_r = X W_r \in \mathbb{R}^{B \times N/P}
$$

Đây là phép nhân ma trận thuần, không có collective. Cuối forward, rank $r$ giữ $Y_r$, là cột thứ $r N/P$ tới $(r+1) N/P$ của $Y$ đầy đủ.

Trạng thái cuối: $Y$ bị shard theo chiều $N$ trên các rank. Mỗi rank giữ một slice.

```mermaid
flowchart LR
  subgraph Rank0[Rank 0]
    X0[X] --> mul0[matmul] --> Y0[Y_0]
    W0[W_0] --> mul0
  end
  subgraph RankN[Rank P-1]
    Xn[X] --> muln[matmul] --> Yn[Y_{P-1}]
    Wn[W_{P-1}] --> muln
  end
```

## Backward

Trong backward, ta nhận gradient của output từ layer phía trên. Vì $Y$ bị shard, gradient của $Y$ cũng đến dưới dạng shard: $\frac{\partial L}{\partial Y_r} \in \mathbb{R}^{B \times N/P}$ trên rank $r$.

Có hai gradient cần tính.

**Gradient của $W_r$**. Theo công thức:

$$
\frac{\partial L}{\partial W_r} = X^T \frac{\partial L}{\partial Y_r} \in \mathbb{R}^{K \times N/P}
$$

Mỗi rank tính độc lập, dùng $X$ replicate sẵn có. Không collective. Mỗi rank update $W_r$ của mình.

**Gradient của $X$**. Đây là phần thú vị. Vì $X$ replicate, mọi rank cần cùng giá trị $\frac{\partial L}{\partial X}$. Theo công thức trên một rank:

$$
\left(\frac{\partial L}{\partial X}\right)_r = \frac{\partial L}{\partial Y_r} \, W_r^T \in \mathbb{R}^{B \times K}
$$

Đây chỉ là **một phần** đóng góp cho $\partial L / \partial X$. Tổng đầy đủ là:

$$
\frac{\partial L}{\partial X} = \sum_{r=0}^{P-1} \frac{\partial L}{\partial Y_r} \, W_r^T
$$

Để mọi rank có cùng $\partial L / \partial X$ đầy đủ, ta cần **all-reduce** với phép cộng. Đây là collective duy nhất trong backward Column Parallel khi input replicate.

## Tóm tắt sơ đồ Column Parallel

| Bước | Trạng thái $X$ | Trạng thái $Y$ | Collective |
|---|---|---|---|
| Vào forward | Replicate | (chưa có) | Không |
| Sau forward | Replicate | Shard theo $N$ | Không |
| Vào backward | Replicate, $\partial L / \partial Y_r$ shard | | Không |
| Sau backward | Có $\partial L / \partial X$ đầy đủ trên mọi rank | $\partial L / \partial W_r$ shard | All-reduce trên $\partial L / \partial X$ |

Quan sát quan trọng: **forward không có collective**. Mọi communication dồn vào backward.

## Tương ứng với PyTorch `ColwiseParallel`

Trong `torch.distributed.tensor.parallel`, lớp `ColwiseParallel()` đặt placement của param và input/output đúng như derivation trên. Mặc định:

- Input layout: `Replicate()`.
- Param `weight`: shard theo dim output (PyTorch dim 0 vì shape là `(d_out, d_in)`).
- Output layout: `Shard(-1)` (shard theo dim cuối, chính là $N$).

Khi gradient flow ngược, PyTorch tự sinh ra all-reduce để re-construct gradient của input. Bạn không phải tự viết collective.

Code minh họa trực tiếp trong `01_simple_model/train.py` ở repo:

```python
plan = {
    "w1": ColwiseParallel(),
    "w2": RowwiseParallel(),
    "w3": ColwiseParallel(),
}
parallelize_module(self.model, tp_mesh, plan)
```

`w1` và `w3` là Column Parallel vì input là $X$ chung (replicate) và output sẽ feed vào một phép Row tiếp theo. Phần 3 sẽ đi sâu.

## Chi phí communication

Trong forward: **0**.

Trong backward: một all-reduce trên tensor có shape $B \times K$. Chi phí một all-reduce trên $P$ rank xấp xỉ:

$$
C_{\text{all-reduce}} \approx 2 \cdot \frac{P-1}{P} \cdot |\text{tensor}|
$$

(theo Ring all-reduce). Với tensor $B \times K$ và bytes-per-element $b$, chi phí khoảng $2 b B K (P-1)/P$ byte mỗi backward.

Với batch $B$, sequence $S$ trong transformer, $X$ shape $(B \cdot S, K)$, chi phí một backward là $O(B \cdot S \cdot K)$. Lưu ý: **không** scale với $N$, vì all-reduce chỉ trên gradient input, không trên gradient weight.

## Khi nào dùng Column Parallel đơn lẻ

Trong transformer, hiếm khi dùng Column Parallel đơn lẻ. Thường nó là tầng đầu của một cặp Column-Row. Lý do: nếu chỉ Column, output bị shard, layer sau muốn dùng phải all-gather hoặc tự xử lý shard, tạo thêm collective vô ích.

Pattern khôn ngoan: **Column tầng $\ell$ + Row tầng $\ell+1$**. Output Column ăn khớp tự nhiên với input Row, không cần collective ở giữa. Phần 4 (Row) sẽ làm rõ cơ chế khớp này.

## Một biến thể: Column với output_layouts khác

Đôi khi ta cần output Column ở dạng `Replicate()` thay vì `Shard()`. Đó là khi layer sau là một phép không TP-friendly và cần input đầy đủ. Trong trường hợp đó, ta khai báo `output_layouts=Replicate()` và PyTorch sẽ thêm all-gather để gộp $Y$ về đầy đủ.

Trong toy `02_large_language_model/parallelism.py`, layer `output` (LM head) dùng `output_layouts=Shard(-1)`, vì layer kế là loss và ta dùng `loss_parallel` để xử lý shard tiếp tục. Đây là tinh tế cao cấp, Phần 5 sẽ giải thích.

## Kiểm tra trí nhớ

Trước khi qua chương Row, hãy tự trả lời.

Một, trong Column Parallel, sau forward, có cần collective không.

Hai, gradient nào cần all-reduce trong backward Column: gradient của $W$, gradient của $X$, hay cả hai.

Ba, nếu input $X$ không phải replicate mà là shard (đến từ một Row layer trước đó), Column có chạy đúng không. Lý do.

Câu trả lời: một, không. Hai, gradient của $X$. Ba, không chạy đúng trực tiếp, vì Column cần input replicate. Phải có collective (all-gather) để chuyển input từ shard sang replicate trước khi áp Column.
