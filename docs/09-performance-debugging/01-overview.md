---
title: Tổng quan Phần 9
---

# Phần 9: Performance và Debugging

Bạn đã viết được plan TP đúng. Bây giờ đến lúc làm cho nó chạy nhanh và khi không nhanh thì biết tại sao. Phần 9 tập trung vào performance engineering và debugging thực tế.

## Hai mặt của tối ưu

**Mặt 1: tối ưu compute và memory mỗi rank.** FlashAttention, fused kernel, mixed precision tốt hơn, activation checkpointing thông minh hơn. Đây là tối ưu "trong" GPU.

**Mặt 2: tối ưu giao tiếp giữa các rank.** Overlap collective với compute, chọn topology mesh phù hợp, bucket gradient cho FSDP. Đây là tối ưu "giữa" các GPU.

Cả hai mặt cùng quan trọng. LLM training tốt cân bằng cả hai để đạt $\ge 50\%$ Model FLOPs Utilization (MFU) trên H100.

## Cấu trúc bốn chương

Chương đầu (`02-overlap-collective-va-compute`) phân tích cơ chế overlap trên CUDA stream, vì sao AllGather của FSDP có thể "ẩn" sau compute, và cách prefetch tốt.

Chương hai (`03-activation-checkpoint-va-bo-nho`) đi sâu activation checkpointing: full checkpoint, selective checkpoint, và CPU offload checkpoint. Tradeoff cụ thể.

Chương ba (`04-profiling-va-mfu`) hướng dẫn dùng PyTorch Profiler, Nsight, và cách tính MFU cho LLM training. Biết được code đang chậm ở đâu là 90% công việc tối ưu.

Chương cuối (`05-debug-common-pitfalls`) liệt kê các bug thường gặp với TP + FSDP, triệu chứng và cách fix. Đây là chương "tra cứu" khi bạn gặp lỗi placement mismatch, NaN loss, hoặc OOM bất thường.

## Mục tiêu sau Phần 9

1. Hiểu CUDA stream và cách collective overlap với compute.
2. Cấu hình activation checkpoint phù hợp với hardware.
3. Profiling và đọc trace để biết bottleneck ở đâu.
4. Diagnose nhanh các bug TP + FSDP phổ biến.

Phần 9 không có lý thuyết mới, chủ yếu là kinh nghiệm thực tế và checklist. Đây là phần bạn quay lại đọc nhiều lần khi training thật, không phải đọc một lần rồi quên.

## Cảnh báo

Performance tuning trên cluster lớn là nghệ thuật, không phải khoa học. Cùng một plan TP có thể chạy nhanh trên A100 và chậm trên H100, hoặc ngược lại. Bandwidth NVLink, kích thước batch, độ dài sequence, tất cả ảnh hưởng. Phần 9 cho bạn principles, nhưng bạn vẫn phải profile trên hardware mục tiêu của mình.

Đừng tin con số benchmark từ paper hoặc blog 100%. Hãy chạy thử trên cluster của bạn.
