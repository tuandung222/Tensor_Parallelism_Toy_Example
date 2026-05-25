---
title: Bảng thuật ngữ
---

# Bảng thuật ngữ

Bảng tra cứu nhanh các thuật ngữ chính xuất hiện trong chuỗi bài giảng. Định nghĩa cô đọng, dành cho người đã đọc qua các chương tương ứng.

## Parallelism families

**Data Parallel (DP).** Mỗi rank giữ một bản sao đầy đủ của model, mỗi rank xử lý một phần khác nhau của batch. Gradient được all-reduce ở cuối mỗi step để giữ các bản sao đồng bộ.

**Tensor Parallel (TP).** Mỗi rank giữ một phần (shard) của từng tensor trọng số trong một layer. Tính toán trên cùng một mini-batch được chia cho nhiều rank. Yêu cầu collective trong forward và backward.

**Pipeline Parallel (PP).** Mỗi rank giữ một subset các layer liên tiếp. Forward đi qua các stage tuần tự, có lúc bị bubble nếu không có micro-batching.

**Fully Sharded Data Parallel (FSDP).** Tương tự DP nhưng tham số, gradient, và optimizer state đều được shard. Forward và backward unshard tạm thời từng layer khi cần.

**2D Parallelism.** Kết hợp hai chiều parallelism, ví dụ TP cho layer trong và FSDP cho dữ liệu. Mỗi rank được gắn vào một tọa độ trên DeviceMesh 2D.

## TP-specific

**Column Parallel.** Shard ma trận trọng số theo cột. Input replicate, output shard cuối. Không collective trong forward, all-reduce trong backward với gradient của input.

**Row Parallel.** Shard ma trận trọng số theo hàng. Input shard cuối, output partial, cần all-reduce ở cuối forward để được replicate. Identity trong backward với gradient của input.

**Sequence Parallel (SP).** Mở rộng của TP, shard luôn chiều sequence của activation, không chỉ chiều hidden. Yêu cầu thêm reduce-scatter và all-gather ở biên giữa TP và SP zones.

**Megatron MLP pattern.** Đặt linear thứ nhất ở Column và linear thứ hai ở Row. Cho phép forward có đúng một all-reduce ở cuối, không có collective ở giữa. Pattern này áp dụng được cả cho MLP cổ điển và SwiGLU.

## PyTorch primitives

**ProcessGroup.** Đối tượng đại diện cho một tập hợp process tham gia chung trong các collective. Mặc định ánh xạ về thư viện NCCL trên GPU.

**Collective.** Phép giao tiếp đồng bộ giữa các rank trong ProcessGroup. Các collective phổ biến: AllReduce, AllGather, ReduceScatter, Broadcast, Scatter.

**DeviceMesh.** Sắp xếp world process thành lưới đa chiều, mỗi chiều có nhãn. Cho phép gom các rank theo trục để dùng cho TP, DP, PP, hoặc kết hợp.

**DTensor.** Tensor có metadata phân bố. Một đối tượng Python duy nhất trên mỗi rank đại diện cho tensor logic, biết phần local của mình ánh xạ vào tensor đầy đủ ra sao.

**Placement.** Loại metadata của DTensor mô tả cách phân bố trên một chiều mesh: `Replicate`, `Shard(dim)`, hoặc `Partial`.

**parallelize_module.** Hàm chính của `torch.distributed.tensor.parallel`. Nhận `(module, mesh, plan)` và áp dụng plan TP lên module, biến các `nn.Linear` thành phiên bản dùng DTensor.

**ColwiseParallel / RowwiseParallel.** Hai style chuẩn cho linear, đại diện cho Column và Row Parallel. Có thể cấu hình `input_layouts` và `output_layouts` để khớp với context xung quanh.

## Collective operations

**AllReduce.** Tổng (hoặc op khác) tensor từ tất cả các rank, kết quả replicate trên mọi rank. Chi phí $\approx 2 \cdot \text{size}$ byte chuyển.

**AllGather.** Mỗi rank đóng góp một shard, mọi rank nhận về tensor đầy đủ concat lại. Chi phí $\approx (P-1) \cdot \text{shard size}$ byte chuyển.

**ReduceScatter.** Ngược lại AllGather. Mỗi rank đóng góp tensor đầy đủ, nhận về một shard đã reduce.

**Broadcast.** Một rank gửi tensor đến mọi rank khác. Chi phí $\approx \text{size}$ byte.

## Memory-related

**Activation memory.** Bộ nhớ giữ kết quả trung gian trong forward để dùng cho backward. Thường lớn hơn parameter memory nhiều lần với sequence dài.

**Activation checkpointing.** Kỹ thuật chỉ giữ một số activation chọn lọc, tính lại các activation khác trong backward để đổi tính toán lấy bộ nhớ.

**Optimizer state.** Bộ nhớ giữ các moment của optimizer (Adam có hai moment). Với fp32, optimizer state chiếm $2\times$ parameter memory; tổng cộng parameter + optimizer = $3\times$ parameter size.

**Mixed precision (bf16).** Lưu parameter và compute ở bf16, giữ một bản fp32 cho weight update. Giảm parameter memory tổng thể nhưng tăng độ phức tạp.

## Llama-3 toy code

**`tok_embeddings`.** Embedding table $\mathbb{R}^{V \times K}$. Thường shard theo vocab (Row Parallel) khi vocab lớn.

**`output` / LM head.** Linear $\mathbb{R}^{K \times V}$ tính logits. Shard cột (Column Parallel) trên chiều $V$.

**`wq, wk, wv, wo`.** Bốn linear của Self-Attention. `wq, wk, wv` Column, `wo` Row. Số head được chia đều cho TP size.

**`feed_forward.w1, w2, w3`.** Ba linear của SwiGLU FFN. `w1, w3` Column, `w2` Row.

**`attention_norm, ffn_norm`.** RMSNorm trước attention và FFN. Trong Sequence Parallel, đánh dấu `SequenceParallel()`.
