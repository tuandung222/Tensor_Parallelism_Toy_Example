---
title: Khi nào nên chọn Tensor Parallelism
---

# Khi nào nên chọn Tensor Parallelism

Tensor Parallelism rất mạnh nhưng không miễn phí. Nó tạo communication ở giữa từng linear layer trong cả forward lẫn backward. Nếu chọn sai bối cảnh, overhead có thể vượt lợi ích. Chương này đưa ra một mental checklist để quyết định khi nào TP đáng dùng.

## Ba câu hỏi trước khi chọn TP

### Một, có vấn đề bộ nhớ thật không

TP tồn tại để giải quyết bộ nhớ. Nếu mô hình của bạn vừa thoải mái trong một GPU, đừng dùng TP. Dùng DP thuần để tăng throughput. TP cộng thêm communication mà không trả lại gì có ích.

Cách kiểm tra nhanh: chạy một forward+backward với batch=1 trên một GPU. Nếu nó OOM, bạn có vấn đề bộ nhớ và cần TP hoặc FSDP. Nếu nó không OOM nhưng batch lớn lên OOM, bạn cần gradient accumulation hoặc activation checkpointing trước khi nghĩ đến TP.

### Hai, có băng thông intra-node tốt không

TP đòi hỏi nhiều collective. Mỗi linear bị shard sinh ra ít nhất một all-reduce mỗi forward và mỗi backward. Băng thông giữa các GPU phải đủ cao để collective không trở thành nút thắt.

Trong một node có NVLink (như A100, H100, V100 SXM), băng thông giữa GPU đạt 600 GB/s tới 900 GB/s. TP rất phù hợp.

Qua mạng inter-node (Infiniband 200 Gbit/s = 25 GB/s, hoặc Ethernet còn chậm hơn), TP gần như không khả thi. Cùng workload đó qua Ethernet 100 Gbit/s sẽ chậm hơn 50 lần.

Quy tắc bỏ túi: **TP intra-node, FSDP inter-node**.

### Ba, layer của model có đủ lớn để TP tiết kiệm xứng đáng không

TP chỉ tiết kiệm bộ nhớ cho layer có ma trận trọng số lớn. Trong transformer, đó là QKV projection, output projection, và hai (hoặc ba với SwiGLU) FFN linear. Norm và embedding nhỏ hơn nhiều.

Nếu hidden dim nhỏ (ví dụ 768 cho BERT-base), mỗi linear chỉ vài MB. Communication overhead khi TP có thể vượt tiết kiệm bộ nhớ. TP kinh tế thật sự khi hidden dim từ 4096 trở lên, đó là zone của Llama-2 7B và lớn hơn.

## TP=N: chọn N như thế nào

TP size phải chia hết hidden dim, vocab size, và quan trọng nhất là **n_heads** trong attention. Vì khi shard attention theo TP, mỗi rank cần một số nguyên các head. Nếu `n_heads=32` và TP=5, có rank được 6 head và rank được 7 head, mất đối xứng và sai.

Quy tắc: TP là một ước của n_heads. Phổ biến TP=2, 4, 8.

Lớn hơn TP=8 thường không hiệu quả: chi phí communication tăng theo TP còn lợi ích giảm dần. Nhiều paper khuyên TP không vượt số GPU trong một node (thường 8).

## Khi nào TP KHÔNG đáng

Một số trường hợp TP làm hại nhiều hơn lợi.

**Mô hình nhỏ vừa trong một GPU**. TP chỉ thêm overhead, không giảm gì có ích.

**Hidden dim nhỏ (dưới 1024)**. Communication overhead nuốt mất tiết kiệm.

**Mạng inter-node kém**. Nếu các GPU không cùng node và không có RDMA tốt, TP tệ.

**Inference với batch=1**. TP thêm latency trên đường critical path mà không tăng throughput. Khi serving, TP chỉ đáng nếu một GPU không chứa nổi model.

**Khi mô hình đã vừa với FSDP nhẹ nhàng**. Nếu FSDP một mình giải được bài toán bộ nhớ, chưa cần thêm TP.

## Khi nào TP đáng

Ngược lại, TP là lựa chọn tự nhiên trong các tình huống sau.

**Mô hình hàng tỷ tham số trên node có NVLink**. Llama-2 7B, Llama-3 8B, Mistral 7B trở lên trên A100/H100. Đây là ngữ cảnh “canonical” của TP.

**Khi muốn batch effective lớn với chi phí communication thấp**. TP shard param, để lại nhiều bộ nhớ cho activation. Có thể tăng batch sequence trên cùng GPU.

**Inference với batch lớn**. Khi serving model lớn, TP giảm latency mỗi token vì nhân ma trận lớn được chia giữa nhiều GPU. Trade-off với communication, nhưng tổng latency thường tốt hơn nếu intra-node.

**Khi cần 2D Parallelism**. TP + FSDP hoặc TP + PP gần như không thể thiếu cho mô hình từ vài chục tỷ trở lên.

## Toy của chúng ta nằm ở đâu

Toy `02_large_language_model/` có Llama-3 thu nhỏ 3.5B trên 4 GPU 24 GB. Tính nhanh: chỉ param+opt đã 35 GB tổng, không có TP thì OOM. Với TP=2, mỗi GPU giữ 17.5 GB cho param+opt. Cộng FSDP=2 thì shard tiếp một lần nữa, chỉ còn 8.75 GB. Đủ chỗ cho activation, gradient và overhead. Đó là lý do `data_parallel_size=auto` và `tensor_parallel_size=auto` trong Lightning chọn TP=2 và FSDP=2 cho 4 GPU 24 GB.

Đây là sweet spot toy thiết kế ra: đủ lớn để cảm nhận TP có ý nghĩa, đủ nhỏ để chạy được trên một Studio Lightning có 4 GPU.

## Bài tập kết chương

Trước khi sang Phần 1, hãy trả lời cho mình.

Một, nếu bạn có 8 H100 80 GB với NVSwitch full mesh, train Llama-2 70B, bạn chọn TP/PP/FSDP ra sao và vì sao.

Hai, nếu mạng giữa các GPU là Ethernet 25 Gbit/s, bạn có dùng TP không. Lý do là gì.

Ba, hãy đoán: với toy `01_simple_model/train.py` (FeedForward 8192x8192), nếu chuyển từ TP=4 sang TP=1 (chạy trên một GPU), khả năng cao là OOM hay không. Lập luận theo bộ nhớ.

Khi bạn có thể trả lời rành mạch, bạn đã có **trực giác** đúng. Phần 1 sẽ cho bạn **toán học** đúng.
