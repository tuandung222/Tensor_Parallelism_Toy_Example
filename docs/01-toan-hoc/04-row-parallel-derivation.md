---
title: Row Parallel, derivation đầy đủ
---

# Row Parallel, derivation đầy đủ

Row Parallel là cách shard ma trận trọng số theo hàng, đối ngẫu với Column Parallel. Nó tự nhiên khi input của bạn đã được shard theo chiều inner, và bạn cần output replicate. Trong transformer, Row Parallel là tầng đối tác của Column và tạo nên pattern đặc trưng của Megatron-LM.

## Setup

Cho linear layer $Y = XW$ với:

- $X \in \mathbb{R}^{B \times K}$, **shard theo cột** (chia chiều $K$) thành $P$ phần: $X = [X_0, X_1, \dots, X_{P-1}]$ với $X_r \in \mathbb{R}^{B \times K/P}$ trên rank $r$.
- $W \in \mathbb{R}^{K \times N}$, **shard theo hàng** thành $P$ phần: $W = \begin{pmatrix} W_0 \\ W_1 \\ \vdots \\ W_{P-1} \end{pmatrix}$ với $W_r \in \mathbb{R}^{K/P \times N}$ trên rank $r$.

Note quan trọng: shard của $X$ và $W$ phải **khớp chiều** $K$. Đây là điều kiện cho phép phân tách.

## Forward

Mỗi rank tính:

$$
Y_r^{\text{partial}} = X_r W_r \in \mathbb{R}^{B \times N}
$$

Lưu ý $Y_r^{\text{partial}}$ có **cùng shape** với $Y$ đầy đủ, nhưng nó là partial sum: chỉ đóng góp của chiều $K$ thuộc rank $r$. Tổng các partial mới ra $Y$ thật:

$$
Y = \sum_{r=0}^{P-1} Y_r^{\text{partial}} = \sum_{r=0}^{P-1} X_r W_r
$$

Để mọi rank có $Y$ đầy đủ, ta cần **all-reduce** với phép cộng:

$$
Y = \text{AllReduce}_+(Y_0^{\text{partial}}, Y_1^{\text{partial}}, \dots, Y_{P-1}^{\text{partial}})
$$

Đây là collective duy nhất trong forward Row Parallel với input shard.

```mermaid
flowchart LR
  subgraph Rank0[Rank 0]
    X0[X_0] --> mul0[matmul] --> P0[Y_0_partial]
    W0[W_0] --> mul0
  end
  subgraph RankN[Rank P-1]
    Xn[X_{P-1}] --> muln[matmul] --> Pn[Y_{P-1}_partial]
    Wn[W_{P-1}] --> muln
  end
  P0 --> AR[All-Reduce Sum]
  Pn --> AR
  AR --> Y[Y full on all ranks]
```

## Backward

Gradient của $Y$, ký hiệu $\partial L / \partial Y$, đến dưới dạng **replicate** trên mọi rank (vì $Y$ là replicate sau all-reduce).

**Gradient của $W_r$**. Áp dụng công thức:

$$
\frac{\partial L}{\partial W_r} = X_r^T \frac{\partial L}{\partial Y} \in \mathbb{R}^{K/P \times N}
$$

Mỗi rank tính độc lập với $X_r$ của mình và $\partial L / \partial Y$ replicate. Không collective.

**Gradient của $X_r$**. Áp dụng công thức:

$$
\frac{\partial L}{\partial X_r} = \frac{\partial L}{\partial Y} \, W_r^T \in \mathbb{R}^{B \times K/P}
$$

Mỗi rank tính độc lập. Kết quả là một slice của $\partial L / \partial X$ tương ứng phần $K/P$ mà rank đang giữ. Không collective.

Tóm lại: backward của Row Parallel **không cần collective**. Toàn bộ communication dồn vào forward. Đây là đối ngẫu với Column Parallel.

## Tóm tắt sơ đồ Row Parallel

| Bước | Trạng thái $X$ | Trạng thái $Y$ | Collective |
|---|---|---|---|
| Vào forward | Shard theo $K$ | (chưa có) | Không |
| Trong forward | | Partial trên mỗi rank | (chưa) |
| Sau forward | Shard | Replicate (đầy đủ) | All-reduce |
| Vào backward | $\partial L / \partial Y$ replicate | | Không |
| Sau backward | $\partial L / \partial X$ shard | $\partial L / \partial W$ shard | Không |

So với Column:

| | Column | Row |
|---|---|---|
| Input layout | Replicate | Shard |
| Output layout | Shard | Replicate (sau all-reduce) |
| Collective forward | Không | All-reduce |
| Collective backward | All-reduce | Không |

Đây là cặp đối ngẫu hoàn hảo. Cộng cả hai vào, **mỗi cặp Column-Row dùng đúng một all-reduce** (trong forward của Row) và một all-reduce nữa (trong backward của Column). Tổng hai all-reduce cho cả forward+backward của một cặp.

## Tương ứng với PyTorch `RowwiseParallel`

```python
plan = {"w2": RowwiseParallel()}
parallelize_module(self.model, tp_mesh, plan)
```

Mặc định `RowwiseParallel()`:

- Input layout: `Shard(-1)` (shard theo dim cuối của input, là chiều $K$).
- Param `weight`: shard theo dim input (PyTorch dim 1 vì shape `(d_out, d_in)`).
- Output layout: `Replicate()`.

PyTorch tự sinh all-reduce sau matmul để hợp partial về đầy đủ.

## Vì sao Column trước, Row sau

Bây giờ ta có thể hiểu pattern Megatron MLP một cách triệt để.

Cho MLP: $Z = \sigma(X W_1) W_2$.

Lựa chọn 1: Column cho $W_1$, Row cho $W_2$.

- $X$ vào replicate.
- $W_1$ Column: output $XW_1$ shard theo chiều output.
- $\sigma$ point-wise: giữ shard.
- $W_2$ Row: input cần shard theo chiều input của $W_2$, tức chiều output của $W_1$. Khớp tự nhiên.
- $W_2$ Row output replicate sau all-reduce.

Tổng forward: một all-reduce. Backward: một all-reduce nữa từ Column. **Tổng cộng 2 all-reduce cho cả layer MLP**, không phải 4.

Lựa chọn 2: Row cho $W_1$, Column cho $W_2$.

- $X$ vào: phải shard theo $K$, cần collective (vì $X$ đến từ layer trước thường là replicate).
- $W_1$ Row: cần all-reduce sau matmul.
- $\sigma$ point-wise, output replicate.
- $W_2$ Column: output shard.
- Cuối: $Z$ shard, layer sau muốn dùng phải gather, thêm collective.

Lựa chọn 2 cộng thêm 1-2 collective không cần thiết. **Megatron chọn lựa chọn 1 không phải tình cờ**, mà vì toán nó kinh tế hơn.

## Quan sát cuối: tại sao SiLU không cần collective

Trong derivation trên ta nói "$\sigma$ point-wise: giữ shard". Đó là vì SiLU, ReLU, GELU đều áp từng phần tử độc lập:

$$
\sigma(t)_{ij} = f(t_{ij})
$$

Không trộn các cột, không trộn các hàng. Nếu input shard theo cột, output cũng shard theo cột, không cần collective.

Đây là một bài học chung: **point-wise operation giữ nguyên layout**. Norm thì không hoàn toàn (nó tính mean/variance), nên norm cần xử lý đặc biệt (đây là lý do Sequence Parallel xuất hiện, Phần 6 sẽ làm rõ).

## Kiểm tra trí nhớ

Một, trong Row Parallel với input shard, collective nào xảy ra ở forward.

Hai, có cần collective trong backward Row không. Vì sao.

Ba, trong MLP Column-then-Row, một forward đầy đủ có bao nhiêu all-reduce.

Đáp số: một all-reduce ở forward. Không cần collective ở backward. Một all-reduce cho cả MLP trong một forward.
