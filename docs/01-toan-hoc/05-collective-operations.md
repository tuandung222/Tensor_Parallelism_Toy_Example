---
title: Collective Operations và mô hình chi phí
---

# Collective Operations và mô hình chi phí

Trong Tensor Parallelism, ta gặp bốn collective operation cốt lõi: **all-reduce**, **all-gather**, **reduce-scatter**, và **broadcast**. Hiểu cơ chế và chi phí của chúng cho phép bạn dự đoán hiệu năng TP trước khi chạy code.

## Bốn collective cốt lõi

### All-reduce

**Định nghĩa**. Cho $P$ rank, mỗi rank $r$ giữ tensor $t_r$ cùng shape. Sau all-reduce với phép cộng, mọi rank có cùng tensor kết quả $t = \sum_{r=0}^{P-1} t_r$.

**Khi dùng trong TP**. Sau forward Row Parallel để gộp partial sum. Sau backward Column Parallel để gộp gradient của input.

**Cài đặt phổ biến**. **Ring all-reduce**. Mỗi tensor được chia thành $P$ chunk. Trong $P-1$ bước, từng chunk được luân chuyển qua mỗi rank theo vòng, tích lũy tổng. Tiếp theo $P-1$ bước nữa để phát kết quả cho mọi rank.

**Chi phí**. Mỗi rank gửi và nhận tổng cộng $2 (P-1)/P \cdot |t|$ byte qua mạng. Với NCCL trên NVLink, throughput thực tế gần đạt bandwidth tối đa.

### All-gather

**Định nghĩa**. Mỗi rank $r$ giữ chunk $t_r$ shape $(s,)$. Sau all-gather, mọi rank có tensor đầy đủ $(t_0, t_1, \dots, t_{P-1})$ shape $(P s,)$.

**Khi dùng trong TP**. Khi cần chuyển output shard sang replicate. Trong FSDP, để gather param trước khi compute.

**Chi phí**. Mỗi rank gửi và nhận $(P-1)/P \cdot |t_{\text{full}}|$ byte. Rẻ hơn all-reduce vì không có cộng.

### Reduce-scatter

**Định nghĩa**. Ngược với all-gather. Mỗi rank giữ tensor đầy đủ shape $(Ps,)$. Sau reduce-scatter, mỗi rank chỉ có chunk thứ $r$ shape $(s,)$ với giá trị là tổng chunk thứ $r$ qua mọi rank.

**Khi dùng trong TP**. Trong sequence parallelism khi ta muốn shard output theo sequence dim sau khi norm replicate. Trong FSDP để scatter gradient.

**Chi phí**. Tương đương all-gather. Reduce-scatter + all-gather = all-reduce (về cost).

### Broadcast

**Định nghĩa**. Một rank gốc giữ tensor $t$. Sau broadcast, mọi rank đều có cùng $t$.

**Khi dùng trong TP**. Khởi tạo: rank 0 đọc checkpoint, broadcast tới rank khác. Sync seed cho dataloader. Hiếm khi dùng trong inner loop training.

**Chi phí**. $(P-1)/P \cdot |t|$ byte gửi từ rank gốc. Cài đặt tree-broadcast tốt giảm xuống $\log P$ bước.

## Tóm tắt bảng

| Collective | Input | Output | Khi dùng | Chi phí (đơn vị tensor full) |
|---|---|---|---|---|
| All-reduce | Mỗi rank: tensor full | Mỗi rank: cùng tensor full (tổng) | Row forward, Column backward | $\sim 2 (P-1)/P$ |
| All-gather | Mỗi rank: chunk | Mỗi rank: tensor full (ghép) | Chuyển shard sang replicate | $\sim (P-1)/P$ |
| Reduce-scatter | Mỗi rank: tensor full | Mỗi rank: chunk (tổng chunk) | Chuyển partial sang shard | $\sim (P-1)/P$ |
| Broadcast | 1 rank: tensor | Mọi rank: tensor | Init checkpoint, seed | $\sim (P-1)/P$ |

## Mô hình chi phí

Một collective qua một network link tốn:

$$
T = \alpha + \frac{|t|}{B}
$$

với $\alpha$ là latency cố định (microseconds, thường nhỏ), $|t|$ là kích thước tensor (byte), $B$ là bandwidth (byte/s).

Với NCCL trên NVLink ở A100 (600 GB/s), một all-reduce 100 MB mất xấp xỉ $100 / 600 = 167 \mu s$. Khá nhanh.

Với cùng all-reduce qua Ethernet 100 Gbit/s (12.5 GB/s), thời gian là $100 / 12.5 = 8 ms$. Chậm hơn gần 50 lần.

Với một mô hình transformer 7B, mỗi layer MLP có $XW_2$ với $W_2$ shape $(4d, d)$ và output $B \cdot S \times d$. Với $d=4096$, $B \cdot S = 8 \cdot 4096 = 32768$, tensor sau matmul shape $(32768, 4096) = 1.3 \cdot 10^8$ phần tử, ở `bfloat16` là 256 MB. Một all-reduce qua NVLink mất $\sim 0.4 ms$. Có 32 layer, mỗi forward 2 all-reduce (MLP và attention), tổng $32 \cdot 2 \cdot 0.4 = 25 ms$ chỉ cho communication.

Đây là một con số đáng nhớ: communication không miễn phí, nhưng chấp nhận được trên NVLink. Qua Ethernet, nó trở thành nút thắt chí mạng.

## Phân tích chi phí cho TP

Cho transformer $L$ layer, hidden $h$, batch $B$, sequence $S$, TP size $P$. Mỗi layer có hai linear "lớn" (attention output $W_O$ và MLP $W_2$), mỗi cái là Row Parallel với output replicate.

**Forward**. Mỗi layer cần 2 all-reduce. Tensor all-reduce có kích thước $B \cdot S \cdot h$ phần tử. Với $b$ byte/element, mỗi all-reduce tốn $2(P-1)/P \cdot b \cdot B \cdot S \cdot h$ byte.

Tổng communication forward một step:

$$
C_{\text{fwd}} = L \cdot 2 \cdot 2 \cdot \frac{P-1}{P} \cdot b \cdot B \cdot S \cdot h
$$

Hệ số 2 đầu vì có MLP và attention. Hệ số 2 sau là factor of all-reduce.

**Backward**. Mỗi Column Parallel cần một all-reduce trên gradient input. Có 4 Column trong một layer (Q, K, V, và $W_1$ của MLP). Mỗi all-reduce trên tensor $B \cdot S \cdot h$. Cộng thêm communication tổng của backward thường gấp đôi forward.

Quy luật xấp xỉ: **communication mỗi step xấp xỉ $O(L \cdot B \cdot S \cdot h)$**, không phụ thuộc $N$ tổng tham số một cách trực tiếp.

Khi tăng $P$ (TP size), communication tăng dần (hệ số $(P-1)/P$), nhưng tốc độ tăng chậm. Khi $P=2$, hệ số 0.5. Khi $P=8$, hệ số 0.875. Khi $P=128$, hệ số 0.992. Tức là phần lớn cost đã có từ $P$ nhỏ; tăng $P$ không làm cost bùng nổ.

## Quan sát cuối

Có ba quan sát đáng nhớ.

Một, communication trong TP scale với **hidden dim**, không phải param count. Mô hình rộng (hidden lớn) communication nhiều hơn, không phải mô hình sâu (nhiều layer).

Hai, communication scale tuyến tính với batch và sequence. Tăng batch lớn lên không miễn phí về communication.

Ba, NVLink là điều kiện cần thực tế cho TP có ý nghĩa. Đó là lý do data center deep learning có topology cẩn thận: trong một node 8 GPU đều nối qua NVSwitch, giữa các node qua Infiniband. Mọi TP nằm trong node.

## Một sự thật cuối: overlap

PyTorch và NCCL ngày nay có thể **overlap** một số collective với compute. Trong khi GPU đang nhân ma trận layer $\ell$, communication cho layer $\ell-1$ có thể đang chạy nền. Điều này giảm wall-clock cost của communication. Tuy nhiên, không bao giờ giảm về 0. Phần 9 sẽ phân tích overlap chi tiết.

Đến đây, bạn đã có toàn bộ vocabulary toán học cần thiết. Phần 2 đi vào primitives của PyTorch để biến những công thức này thành code thật.
