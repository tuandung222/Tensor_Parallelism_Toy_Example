---
title: Tổng quan Phần 1
---

# Tổng quan Phần 1

Phần 1 là phần quan trọng nhất của toàn chuỗi. Sau khi học xong phần này, bạn sẽ tự derive được mọi pattern Tensor Parallelism mà không cần nhớ. Megatron MLP, attention, embedding, mọi thứ sẽ trở nên hiển nhiên vì bạn nắm được nguyên lý.

## Câu hỏi trung tâm

Cho một phép tính $Y = XW$ trên một GPU. Nếu ma trận $W$ quá lớn, ta shard $W$ giữa nhiều GPU. Có hai cách tự nhiên để shard $W$: theo cột, hoặc theo hàng. Câu hỏi là:

- Mỗi cách shard biến đổi $X$ và $Y$ như thế nào.
- Collective operation nào cần thiết để kết quả đúng.
- Chi phí communication của mỗi cách là bao nhiêu.
- Khi nào chọn cách nào, và vì sao trong transformer ta thấy cả hai xen kẽ.

Trả lời được bốn câu hỏi đó là trả lời được toàn bộ Tensor Parallelism.

## Quy ước ký hiệu

Cho tới hết Phần 1, tôi sẽ dùng các ký hiệu sau, hãy nội hóa chúng.

$N$ là tổng số tham số. $P$ là số GPU trong TP group, hay nói khác là TP size. $r$ là rank của GPU (từ 0 đến $P-1$).

$W \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}$ là ma trận trọng số. Quy ước hàng-cột chuẩn của PyTorch là `nn.Linear(d_in, d_out)` cho `weight.shape = (d_out, d_in)`, và phép tính là $y = x W^T + b$. Để derivation gọn, ta sẽ dùng $Y = X W$ với $W$ đã transpose. Khi triển khai vào code, đừng quên PyTorch có thứ tự ngược.

$X \in \mathbb{R}^{B \times d_{\text{in}}}$ là input batch. $Y \in \mathbb{R}^{B \times d_{\text{out}}}$ là output.

Khi shard, tôi sẽ viết $W = [W_0, W_1, \dots, W_{P-1}]$ cho shard theo **cột** (column-wise), và $W = \begin{pmatrix} W_0 \\ W_1 \\ \vdots \\ W_{P-1} \end{pmatrix}$ cho shard theo **hàng** (row-wise).

## Cấu trúc Phần 1

Chương 2 ôn lại nhanh phép nhân ma trận và đặc tính phân tách của nó. Nếu bạn đã chắc, đọc lướt.

Chương 3 derive đầy đủ Column Parallel: cách shard $W$ theo cột, $X$ replicate, $Y$ được shard. Cuối forward, không cần collective. Chương 4 derive Row Parallel: cách shard $W$ theo hàng, $X$ phải shard, $Y$ partial trên mỗi rank, cần all-reduce để gộp. Chương 5 đi vào collective operation: all-reduce, all-gather, reduce-scatter, broadcast, với mô hình chi phí.

## Mục tiêu sau Phần 1

Sau khi đọc xong, bạn nên làm được ba việc.

Một, vẽ trên một tờ giấy: cho một MLP hai tầng `Linear(d, 4d) -> SiLU -> Linear(4d, d)`, chỉ ra cách shard sao cho tối thiểu collective. Bạn nên kết luận: tầng đầu Column, tầng hai Row, chỉ cần một all-reduce ở cuối.

Hai, viết được công thức chi phí communication cho TP với batch $B$, sequence $S$, hidden $h$, TP size $P$. Đáp số: mỗi tầng linear shard cần truyền $O(B \cdot S \cdot h)$ phần tử qua mạng. Đây là lý do TP đòi NVLink.

Ba, giải thích được vì sao trong attention `wq, wk, wv` là Column và `wo` là Row, không phải kết hợp khác.

Nếu bạn làm được ba việc đó, bạn có khả năng derive lại mọi paper TP. Đó là mục tiêu.
