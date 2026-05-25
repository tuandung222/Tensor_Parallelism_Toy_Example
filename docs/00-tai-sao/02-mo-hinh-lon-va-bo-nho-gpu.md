---
title: Mô hình lớn và bộ nhớ GPU
---

# Mô hình lớn và bộ nhớ GPU

Trước khi nói parallelism, hãy ngồi một mình với câu hỏi: chính xác cái gì chiếm bộ nhớ trên GPU khi training. Nếu bạn không trả lời được câu này một cách định lượng, mọi quyết định parallelism phía sau chỉ là cargo-culting.

## Năm thành phần trong bộ nhớ

Khi train một transformer trên một GPU, bộ nhớ chia thành năm thành phần lớn.

**Tham số (parameters)** là tensor trọng số của model. Một linear layer `nn.Linear(8192, 8192, bias=False)` có $8192 \times 8192$ phần tử, mỗi phần tử thường là 2 byte ở `bfloat16`, tổng khoảng 128 MB. Một transformer 3.5B param ở `bfloat16` ngốn khoảng 7 GB chỉ để chứa weight.

**Gradient** có cùng shape với param. Mỗi backward sẽ ghi vào một tensor gradient. Với 3.5B param ở `bfloat16` reduce dtype `float32`, riêng gradient có thể chiếm 7 đến 14 GB tuỳ chiến lược lưu trữ.

**Optimizer state** là phần thường bị đánh giá thấp. Với AdamW, mỗi param có hai trạng thái: moment bậc nhất `m` và moment bậc hai `v`, cả hai thường ở `float32`. Tức là cứ một param 2 byte, AdamW thêm 8 byte (4+4). Với 3.5B param, optimizer state riêng đã chiếm 28 GB.

**Activation** là output trung gian của mỗi layer trong forward, được giữ lại để dùng trong backward. Activation phụ thuộc batch size, sequence length, hidden dim và số layer. Với transformer dài và batch lớn, activation có thể vượt cả param.

**KV cache** chỉ tồn tại trong inference autoregressive. Mỗi attention layer cần lưu key và value của mọi token đã sinh, tỉ lệ thuận với sequence length. Trong training thuần forward-backward không có KV cache, nên trong chuỗi này ta sẽ chủ yếu lo bốn thành phần đầu.

## Một phép tính trần trụi cho Llama-3 thu nhỏ

Lấy chính ví dụ trong toy repo `02_large_language_model/`. Llama-3 với `dim=4096`, `n_layers=16`, `vocab=32000`, đầu ra `nn.Linear(dim, vocab)`. Tham số tổng cỡ 3.5B (chính xác theo log của trainer).

Trên một GPU 24 GB, nếu train ở `bfloat16` cho param với AdamW `float32` optimizer state, công thức tham khảo cho memory chỉ cho param và optimizer:

$$
M_{\text{param+opt}} \approx 2 N + 8 N = 10 N \text{ byte}
$$

với $N$ là số tham số. Với $N = 3.5 \times 10^9$:

$$
M_{\text{param+opt}} \approx 10 \times 3.5 \times 10^9 = 35 \text{ GB}
$$

Đã vượt 24 GB ngay cả khi chưa tính gradient và activation. Tức là một GPU 24 GB không có cách nào chứa nổi.

Đây là “vấn đề lớn”. Nó không phải tốc độ. Nó là một ràng buộc cứng về dung lượng.

## Vì sao bfloat16 không cứu

Một câu hỏi tự nhiên: tại sao không dùng `bfloat16` cho tất cả. Câu trả lời ngắn là độ chính xác. AdamW cần moment bậc hai chính xác để chia, và `bfloat16` có precision quá thấp (chỉ 7 bit mantissa) khiến accumulation tích lũy sai số rất nhanh. Mixed precision đúng cách giữ param ở `bfloat16` cho compute, optimizer state ở `float32` cho stability, và đó là lý do hệ số 8 byte cho optimizer state là khó tránh.

Có các trick như `bfloat16` optimizer hoặc 8-bit optimizer (Adam-8bit), nhưng đó là tối ưu hạng hai. Lớp tối ưu hạng nhất, dành cho mô hình thực sự lớn, là phân chia mô hình giữa nhiều GPU.

## Phép tính activation: con voi ẩn

Activation thường bị quên. Với một transformer hidden $h$, sequence length $s$, batch $b$, một layer cơ bản giữ activation tỷ lệ $O(b \cdot s \cdot h)$. Với nhiều layer $L$, tổng activation $O(L \cdot b \cdot s \cdot h)$.

Trong toy của chúng ta `dim=4096`, `seq_length=128`, `batch=8`, $L=16$. Activation tổng theo bậc lớn:

$$
b \cdot s \cdot h \cdot L = 8 \times 128 \times 4096 \times 16 \approx 6.7 \times 10^7
$$

phần tử. Ở `bfloat16`, đó là 134 MB. Nghe nhỏ, nhưng activation thực tế bao gồm nhiều intermediate (Q, K, V, attention scores, FFN intermediate), nhân thêm 8 tới 12 lần. Tổng tới vài GB.

Khi tăng `seq_length` lên 4096 hoặc 8192, activation thành con voi ẩn. Đó là lý do **activation checkpointing** xuất hiện rất sớm trong file `parallelism.py` của toy.

## Bộ nhớ là ràng buộc cứng

Nhiều người nghĩ parallelism là để tăng tốc. Đúng một nửa. Nó cũng là điều kiện để bài toán có thể thực thi được. Một GPU không vừa thì không có chuyện chậm hay nhanh. Bộ nhớ là ràng buộc cứng.

Mọi loại parallelism mà chúng ta sẽ học đều có thể hiểu qua lăng kính: nó cắt thành phần nào trong năm thành phần trên thành các phần nhỏ hơn để vừa GPU.

Data Parallel sao chép toàn bộ model, không cắt gì cả về param/opt state. Nó chỉ chia batch. Đây là lý do DP không cứu được mô hình lớn.

Tensor Parallel cắt từng linear layer thành lát theo cột hoặc hàng. Param, gradient và optimizer state của layer đó đều bị cắt.

Pipeline Parallel cắt các layer khác nhau cho các GPU khác nhau. Layer 1-8 trên GPU 0, layer 9-16 trên GPU 1.

FSDP (Fully Sharded Data Parallel) cắt mọi thứ (param, grad, opt state) theo flat dimension và shard giữa các DP replica. Đây là phiên bản tinh tế nhất của DP truyền thống.

## Bài tập nhỏ kết chương

Trước khi đi sang chương sau, hãy cố trả lời.

Một, với Llama-3 3.5B, nếu bạn có 8 GPU 24 GB, lý thuyết bạn nên chia param thành mấy phần để vừa. Giả sử mọi thứ scale tuyến tính.

Hai, nếu bạn chỉ dùng Data Parallel thuần (replicate model trên 8 GPU), bộ nhớ một GPU có giảm không. Nếu không, sao mọi người vẫn dùng DP.

Ba, hãy nghĩ một chiến lược kết hợp: TP=2 và FSDP=4. Mỗi linear bị shard 2 cách, sau đó remainder bị shard tiếp 4 cách. Tức là param/opt state chia làm 8. Vừa được Llama-3 3.5B trên 8 GPU 24 GB không.

Câu trả lời chính xác sẽ rõ trong các phần sau. Nhưng nếu bạn biết tự đặt ra ba câu này, bạn đã có tư duy đúng.
