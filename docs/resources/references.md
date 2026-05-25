---
title: Tài liệu tham khảo
---

# Tài liệu tham khảo

Danh sách rút gọn các tài liệu nền tảng và tài liệu sâu mà chuỗi bài giảng này dựa vào hoặc khuyến nghị đọc thêm.

## Bài báo nền tảng

**Megatron-LM (Shoeybi và cộng sự, 2019)**: bài đặt nền cho pattern Column-then-Row trong MLP và Self-Attention. Đây là gốc của hầu hết kỹ thuật TP hiện nay. Đọc Section 3 (Model Parallel Transformers) cho derivation cốt lõi.

**Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM (Narayanan và cộng sự, 2021)**: mở rộng Megatron với pipeline parallel và 3D parallelism. Khái niệm "interleaved 1F1B" và phân tích chi phí giao tiếp.

**GShard và GSPMD (Lepikhin và cộng sự, 2020; Xu và cộng sự, 2021)**: framework SPMD tổng quát hơn cho parallelism trên TPU, sau này ảnh hưởng đến PyTorch DTensor.

**ZeRO (Rajbhandari và cộng sự, 2019)**: nền tảng của FSDP. ZeRO-1, 2, 3 chia optimizer state, gradient, và parameter. Cùng dùng với TP thành 2D parallelism là một trong những công thức training LLM phổ biến nhất.

**Reducing Activation Recomputation in Large Transformer Models (Korthikanti và cộng sự, 2022)**: bài giới thiệu Sequence Parallel và Selective Activation Checkpointing. Đây là cốt lõi cho Phần 6 của chuỗi bài giảng này.

**SwiGLU (Shazeer, 2020)**: bài ngắn giới thiệu các biến thể GLU activation, trong đó SwiGLU là biến thể đang được Llama, Mistral, và phần lớn LLM sử dụng.

## Tài liệu PyTorch chính thức

**`torch.distributed.tensor.parallel`**: tài liệu API gốc cho `parallelize_module`, `ColwiseParallel`, `RowwiseParallel`, `SequenceParallel`, `PrepareModuleInput`. Đây là entry point chính khi viết plan TP.

**`torch.distributed.device_mesh`**: tài liệu cho `DeviceMesh` và `init_device_mesh`. Bao gồm cách khai báo mesh đa chiều và lấy sub-mesh.

**`torch.distributed.fsdp.fully_shard`** (FSDP2): API mới của FSDP, tích hợp tốt với DTensor và DeviceMesh. Đây là API được dùng trong `02_large_language_model/parallelism.py`.

**`torch.distributed._composable`**: namespace cho các API composable mới (FSDP2, activation checkpoint, TP). Cho phép kết hợp nhiều kỹ thuật mà không xung đột.

## Reference implementation

**torchtitan**: implementation chính thức của PyTorch cho training LLM ở scale lớn, dùng làm template cho `02_large_language_model/parallelism.py` trong toy code. Đây là code "production-near" tốt nhất để học pattern thật.

**FairScale, DeepSpeed**: hai framework lớn cùng thời với FSDP. DeepSpeed cài đặt ZeRO gốc và cung cấp nhiều tối ưu hóa. FairScale là tiền thân của FSDP trong PyTorch.

**Megatron-LM repo (NVIDIA)**: codebase gốc của Megatron. Phức tạp nhưng là nguồn ground truth cho pattern TP và pipeline. Đáng đọc khi bạn cần debug case khó.

## Bài giảng và bài viết kỹ thuật mở rộng

**HuggingFace Performance and Scalability docs**: tài liệu thực dụng cho training và inference LLM, có phần TP, FSDP, và mixed precision. Phù hợp khi tìm "công thức" cụ thể cho phần cứng quen thuộc.

**Lightning Fabric / PyTorch Lightning advanced docs**: phần `ModelParallelStrategy` và `Fabric` mô tả cách Lightning đóng gói TP và FSDP. Là context cho `01_simple_model/train.py`.

**NVIDIA Developer Blog**: nhiều bài kỹ thuật về NCCL tuning, FlashAttention, fused kernel. Hữu ích cho Phần 9 (Performance) khi tối ưu thực tế.

## Sách

**Programming Massively Parallel Processors (Kirk và Hwu)**: nếu bạn muốn hiểu sâu CUDA và collective dưới đáy, đây là sách kinh điển.

**Distributed and Parallel Computing với tài liệu MPI**: dù không phải về deep learning, hiểu mô hình MPI giúp bạn nắm bắt collective operation tổng quát, vì NCCL về mặt API rất gần MPI.

## Cách sử dụng danh sách này

Trong khi đọc chuỗi bài giảng, nếu một khái niệm gặp khó, hãy tra:

- Lý thuyết shard và pattern: Megatron-LM paper.
- API PyTorch cụ thể: tài liệu chính thức tương ứng.
- Implementation chi tiết: torchtitan repo.
- Tối ưu phần cứng: NVIDIA blog và Megatron paper 2021.

Không cần đọc hết, dùng làm tra cứu khi cần đào sâu một chương cụ thể.
