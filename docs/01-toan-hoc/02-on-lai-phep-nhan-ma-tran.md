---
title: Ôn lại phép nhân ma trận và tính phân tách
---

# Ôn lại phép nhân ma trận và tính phân tách

Mọi chuyện trong Tensor Parallelism dựa trên một sự thật đơn giản: phép nhân ma trận có thể tính theo nhiều cách khác nhau cho ra cùng một kết quả. Khác biệt giữa các cách đó là số phép tính và lượng dữ liệu cần luân chuyển. Phần này ôn lại tính chất phân tách của $Y = XW$ để Phần 3 và 4 cảm thấy tự nhiên.

## Định nghĩa cơ bản

Cho $X \in \mathbb{R}^{B \times K}$ và $W \in \mathbb{R}^{K \times N}$. Tích $Y = XW \in \mathbb{R}^{B \times N}$ được tính bởi:

$$
Y_{ij} = \sum_{k=1}^{K} X_{ik} \, W_{kj}
$$

với $i \in [1, B]$ và $j \in [1, N]$. Đây là định nghĩa tích vô hướng giữa hàng thứ $i$ của $X$ và cột thứ $j$ của $W$.

Để TP làm việc, ta cần ba quan sát phân tách.

## Quan sát 1: phân tách theo chiều cột của $W$

Cột thứ $j$ của $Y$, ký hiệu $Y_{:,j}$, chỉ phụ thuộc cột thứ $j$ của $W$:

$$
Y_{:,j} = X \cdot W_{:,j}
$$

Tức là nếu ta có cột thứ $j$ của $W$, ta có thể tính cột thứ $j$ của $Y$ một cách độc lập. Đây là cơ sở của **Column Parallel**.

Chia $W$ thành $P$ khối theo cột, $W = [W_0, W_1, \dots, W_{P-1}]$ với mỗi $W_r \in \mathbb{R}^{K \times N/P}$. Khi đó:

$$
Y = X W = X [W_0, W_1, \dots, W_{P-1}] = [X W_0, X W_1, \dots, X W_{P-1}]
$$

Mỗi rank $r$ giữ $W_r$, nhận $X$ đầy đủ (replicate), tính $Y_r = X W_r \in \mathbb{R}^{B \times N/P}$. Khối $Y_r$ là một phần $Y$. Không cần collective để có $Y_r$. Đây là vẻ đẹp của Column Parallel.

## Quan sát 2: phân tách theo chiều hàng của $W$

Một cách khác là phân tách theo chiều **trong** của phép nhân, tức là chiều $K$. Nếu ta cắt $X$ theo cột (chiều $K$) và cắt $W$ theo hàng (chiều $K$), tích thành tổng của các tích con.

Cho $X = [X_0, X_1, \dots, X_{P-1}]$ với $X_r \in \mathbb{R}^{B \times K/P}$, và $W = \begin{pmatrix} W_0 \\ W_1 \\ \vdots \\ W_{P-1} \end{pmatrix}$ với $W_r \in \mathbb{R}^{K/P \times N}$. Khi đó:

$$
Y = X W = \sum_{r=0}^{P-1} X_r W_r
$$

Mỗi rank $r$ giữ $W_r$ và $X_r$, tính $Y_r^{\text{partial}} = X_r W_r \in \mathbb{R}^{B \times N}$. Note: $Y_r^{\text{partial}}$ có **cùng shape** với $Y$ đầy đủ, nhưng nó chỉ là một phần của tổng. $Y$ thật là tổng của tất cả $Y_r^{\text{partial}}$. Để có $Y$ đầy đủ trên mọi rank, ta cần **all-reduce** với phép cộng.

Đây là cơ sở của **Row Parallel**.

## Quan sát 3: cùng một $Y$, hai cách chia $W$

Hai quan sát trên cho ra hai cách shard hoàn toàn khác nhau cho cùng phép nhân. Bảng so sánh:

| Tiêu chí | Column Parallel | Row Parallel |
|---|---|---|
| Cách shard $W$ | Theo cột (chia $N$ thành $P$) | Theo hàng (chia $K$ thành $P$) |
| Input $X$ | Replicate đầy đủ trên mọi rank | Phải shard theo $K$ |
| Output $Y$ | Bị shard theo $N$ trên mỗi rank | Đầy đủ shape $B \times N$ nhưng là partial sum |
| Collective sau forward | Không cần | All-reduce |
| Communication cost | Có thể có ở **input** nếu $X$ chưa replicate | Ở **output** với kích thước $B \times N$ |

Đây là cặp đối ngẫu. Lựa chọn cái nào phụ thuộc vào **trạng thái của input và output** mong muốn.

## Tính tương đương về toán

Ta cần xác nhận: hai cách shard cho ra **cùng kết quả** với phép tính đơn lẻ. Phép tính ma trận có tính kết hợp và phân phối, nên:

$$
X W = X [W_0, \dots, W_{P-1}] = [X W_0, \dots, X W_{P-1}]
$$

và

$$
X W = \sum_{r=0}^{P-1} X_r W_r \quad \text{(khi $X$ được cắt theo chiều $K$ và $W$ tương ứng)}
$$

Cả hai cùng cho ra $Y \in \mathbb{R}^{B \times N}$. Khác biệt nằm ở **nơi đặt kết quả** và **chi phí communication**. Đây là điểm cốt yếu: TP đúng về toán, chỉ khác về cost.

## Một quan sát quan trọng cho transformer

Trong transformer, các phép tính dài hơn $Y = XW$ một phép. Ví dụ trong MLP:

$$
Z = \sigma(XW_1) W_2
$$

với $\sigma$ là activation (như SiLU). Nếu ta shard $W_1$ theo cột, đầu ra $XW_1$ shard theo chiều output. Activation $\sigma$ tác động phần tử (point-wise), không trộn các chiều, nên giữ shard không cần collective.

Ngược lại, nếu $W_2$ là Row Parallel với input là $\sigma(XW_1)$ đang shard theo chiều output (cũng là chiều input của $W_2$), thì input đã đúng format mà Row cần. Sau Row Parallel, ta được $Z$ partial, all-reduce ra kết quả.

Tức là **một Column tiếp theo một Row chỉ cần một all-reduce** cho hai linear gộp lại. Đây là Megatron MLP pattern. Tự nó xuất hiện từ ba quan sát trên. Phần 3 sẽ derive chi tiết.

## Một bài tập tay nhỏ

Cho $K=4$, $N=4$, $P=2$. Tính bằng tay:

$$
X = \begin{pmatrix} 1 & 2 & 3 & 4 \end{pmatrix}, \quad
W = \begin{pmatrix} 1 & 0 & 1 & 0 \\ 0 & 1 & 0 & 1 \\ 1 & 0 & 1 & 0 \\ 0 & 1 & 0 & 1 \end{pmatrix}
$$

Một, tính $Y = XW$ trực tiếp.

Hai, shard $W$ theo cột (Column): $W_0$ là hai cột đầu, $W_1$ là hai cột sau. Tính $XW_0, XW_1$. Ghép lại có khớp với $Y$ không.

Ba, shard $W$ theo hàng (Row): $W_0$ là hai hàng đầu, $W_1$ là hai hàng sau. Tương ứng shard $X$ thành $X_0 = (1, 2)$ và $X_1 = (3, 4)$. Tính $X_0 W_0$ và $X_1 W_1$. Cộng lại có khớp với $Y$ không.

Làm xong bài tập này bằng tay, bạn sẽ thấy tính phân tách rõ ràng và không bao giờ quên nữa.
