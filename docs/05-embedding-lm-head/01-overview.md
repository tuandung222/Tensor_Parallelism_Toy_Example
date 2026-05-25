---
title: Tổng quan Phần 5
---

# Phần 5: Embedding và LM Head

Sau Phần 3 (MLP) và Phần 4 (Attention), bạn đã shard được tất cả nội dung của một Transformer block. Còn lại hai khối "vành đai": Embedding ở đầu vào (biến token index thành vector $K$ chiều) và LM Head ở đầu ra (biến vector $K$ chiều trở lại logit trên vocab). Đối với LLM, đây là hai khối có vocab size $V$ rất lớn (32K cho Llama, 128K cho Llama-3, 256K cho Qwen-VL), nên không shard chúng cũng là một bottleneck đáng kể.

## Vì sao Embedding và LM Head cần TP riêng

Cả hai khối có một chiều "vocab" rộng. Embedding là một bảng tra $E \in \mathbb{R}^{V \times K}$, LM Head là một linear $W_{out} \in \mathbb{R}^{K \times V}$. Với $V = 128000, K = 4096$, mỗi khối chứa $5 \cdot 10^8$ tham số, hơn 500 MB ở fp32. Trên một GPU 24 GB, chỉ riêng Embedding + LM head đã chiếm gần $5\%$ bộ nhớ, không tính optimizer state.

TP cho hai khối này tự nhiên: shard theo chiều vocab $V$. Nhưng pattern khác MLP/Attention vì cách hoạt động khác. Embedding là phép tra bảng (gather theo index), không phải matmul. LM Head là matmul nhưng output là logit dùng cho cross entropy, có nghĩa ta phải tính loss đúng cách trên output shard.

## Cấu trúc bốn chương

Chương đầu (`02-embedding-rowwise`) phân tích Embedding TP: shard theo hàng (vì hàng = vocab). Mỗi rank giữ một phần vocab, lookup chỉ tìm thấy trên rank "đúng", các rank khác trả về 0. Output cuối được all-reduce.

Chương hai (`03-output-colwise-loss-parallel`) phân tích LM Head: shard theo cột (vì cột = vocab). Output là logit shard cuối. Để tính cross entropy đúng mà không all-gather full $V$, ta dùng `loss_parallel`, tính softmax và NLL trên shard cuối.

Chương ba (`04-weight-tying-considerations`) bàn về weight tying: Llama không tie, nhưng GPT-2 và nhiều mô hình nhỏ thì tie. Weight tying + TP có một số rào cản kỹ thuật ta cần biết.

Chương cuối (`05-embedding-lm-head-walkthrough`) đọc `tok_embeddings` và `output` trong `02_large_language_model/model.py` và `parallelism.py`, lý giải các option `input_layouts`, `output_layouts`, `use_local_output`.

## Mục tiêu sau Phần 5

Sau bốn chương, bạn:

1. Hiểu Embedding TP shard vocab, vì sao và làm cách nào.
2. Hiểu LM Head TP và `loss_parallel`, vì sao tránh all-gather full logit.
3. Phân biệt được Embedding tied và untied trong context TP.
4. Đọc được plan TP cho `tok_embeddings` và `output` trong toy code.

Sau Phần 5, bạn đã có toàn bộ pattern TP cho một LLM "intra-layer" + "đầu vào ra đầu ra". Các phần tiếp theo (Sequence Parallel, FSDP, 2D parallelism) là tối ưu thêm, không phải pattern mới về toán.
