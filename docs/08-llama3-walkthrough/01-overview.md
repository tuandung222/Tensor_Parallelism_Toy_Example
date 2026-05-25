---
title: Tổng quan Phần 8
---

# Phần 8: Walkthrough đầy đủ Llama-3 toy

Đến phần này, bạn đã có tất cả lý thuyết: TP cho MLP và Attention, Embedding/LM Head với loss_parallel, Sequence Parallel, 2D parallelism với FSDP, mixed precision, activation checkpoint. Phần 8 là một bài học tổng hợp: đọc toàn bộ `02_large_language_model/` từ A đến Z, gắn từng dòng vào khái niệm đã học.

## Mục tiêu

Sau Phần 8, bạn:

1. Đọc được toàn bộ `model.py`, `parallelism.py`, `train.py`, `data.py` không bí mật chỗ nào.
2. Biết những chỗ nào "production-ready" và chỗ nào là "toy code simplification".
3. Biết cách port plan TP từ toy code sang model production thật (Llama-3 70B, Mistral, Qwen).
4. Biết các điểm mở rộng nếu muốn build framework training riêng.

## Cấu trúc bốn chương

Chương đầu (`02-model-py`) đọc `model.py`: kiến trúc Transformer Llama, RMSNorm, RoPE, Attention với GQA, SwiGLU FFN, Transformer.

Chương hai (`03-parallelism-py`) đọc `parallelism.py`: hàm `parallelize`, plan TP đầy đủ, FSDP wrap, mixed precision, checkpoint.

Chương ba (`04-train-py`) đọc `train.py`: Lightning module, configure_model, training_step với loss_parallel, optimizer, data loader, Lightning Trainer.

Chương cuối (`05-end-to-end-walk`) chạy ngược một full step end-to-end: từ tokens vào, qua mọi block, ra logits, loss, backward, optimizer step. Mô tả mỗi tensor ở mỗi điểm với placement chính xác.

## Khi nào quay lại Phần 8

Phần 8 là tài liệu reference. Khi viết một plan TP cho model mới, bạn sẽ tra Phần 8 để nhớ:

- Block transformer wrap thế nào (`PrepareModuleInput` ở đâu, `RowwiseParallel(output_layouts=Shard(1))` cho cái nào).
- Khi nào dùng FSDP, khi nào không.
- Mỗi tham số trong `MixedPrecisionPolicy` nghĩa gì.

Phần 8 cũng là điểm bắt đầu nếu bạn muốn customize cho kiến trúc khác. Ví dụ thay SwiGLU bằng GeGLU, hoặc thay GQA bằng MQA, hoặc thêm cross attention. Pattern Megatron vẫn áp dụng, bạn chỉ điều chỉnh `n_heads`, `n_kv_heads`, plan keys.

## Ngữ cảnh thực tế

Toy code `02_large_language_model/` là phiên bản giáo dục, dựa trên `torchtitan` (PyTorch reference implementation cho LLM training). Nó simplify một số phần:

- Không có checkpoint save/load.
- Không có data preprocessing thực (chỉ random tokens).
- Không có evaluation loop.
- Không có gradient accumulation.

Nhưng phần TP/SP/FSDP **đầy đủ và đúng**. Đây là cốt lõi của framework training LLM thật, chỉ bị bao bọc bởi Lightning để giản tiện. Bạn có thể tin tưởng dùng làm template.

Sẵn sàng, chúng ta vào `model.py`.
