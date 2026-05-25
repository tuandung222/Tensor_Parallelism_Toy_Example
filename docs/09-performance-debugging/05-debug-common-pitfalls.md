---
title: Bug thường gặp với TP và FSDP
---

# Bug thường gặp với TP và FSDP

Chương cuối Phần 9 là tra cứu thực dụng. Triệu chứng, nguyên nhân gốc, và cách fix cho các bug phổ biến nhất khi train LLM với TP + FSDP.

## Bug 1: "expected Replicate but got Partial"

**Triệu chứng**: lỗi placement mismatch khi gọi một module hoặc operation trên DTensor.

**Nguyên nhân**: Output của một linear (thường RowwiseParallel) là Partial, nhưng module kế tiếp expect Replicate (hoặc Shard).

**Fix**:

1. Kiểm tra `output_layouts` của RowwiseParallel. Nếu bạn muốn output Replicate (default), không cần option. Nếu Shard(1), thêm `output_layouts=Shard(1)`.
2. Đảm bảo plan TP đầy đủ. Một module quan trọng không có trong plan có thể giữ tensor ở placement không mong muốn.
3. Dùng `tensor.full_tensor()` để force convert sang tensor thường khi cần.

## Bug 2: Loss NaN ngay step đầu

**Triệu chứng**: forward chạy, loss ra `NaN` hoặc `inf` ngay step 1.

**Nguyên nhân tiềm năng**:

1. `loss_parallel` không được bọc cả forward và backward.
2. Mixed precision: parameter init quá lớn, overflow bf16.
3. Quên `n_heads //= tp_size`, view trả về garbage tensor.
4. Sampler không replicate batch trong TP group: rank khác nhau nhận batch khác, all-reduce trộn lung tung.

**Fix**:

1. Bọc cả forward (`with loss_parallel():`) và backward (override `backward`).
2. Check init std của weight, đặc biệt với layer sâu. Llama-3 dùng `0.02 / sqrt(2 * (layer_id + 1))`.
3. Kiểm tra `attn_layer.n_heads` sau `parallelize_module` đúng giá trị local.
4. Đảm bảo Lightning's `ModelParallelStrategy` được dùng, hoặc tự cấu hình `DistributedSampler` với `num_replicas=dp_size`.

## Bug 3: OOM bất thường

**Triệu chứng**: OOM mặc dù tính toán cho thấy đủ bộ nhớ.

**Nguyên nhân**:

1. PyTorch allocator fragmentation. Memory thực không free vì allocator giữ block.
2. Quên checkpoint mọi block: một block không checkpoint, activation tăng vọt.
3. Gradient accumulation: gradient không free giữa các micro-step, peak tăng.
4. FSDP `reshard_after_forward=False` cho nhiều block: parameter giữ lại đầy đủ.

**Fix**:

1. `torch.cuda.empty_cache()` định kỳ. Hoặc set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
2. Verify mọi block trong `for layer_id, transformer_block in model.layers.items()` được checkpoint.
3. Free gradient sau mỗi step (PyTorch tự, nhưng nếu accumulation, chú ý).
4. Đặt `reshard_after_forward=True` cho hầu hết block.

## Bug 4: Hang ở collective

**Triệu chứng**: training treo ở một collective, không tiến.

**Nguyên nhân**:

1. Một rank exception, các rank khác đợi.
2. Sampler tạo batch khác nhau trên các rank cần đồng nhất (data corruption).
3. NCCL timeout: collective thực sự chậm hơn timeout (mặc định 10 phút).

**Fix**:

1. Check log mọi rank, tìm exception. Dùng `NCCL_DEBUG=INFO` để có thêm log.
2. Verify sampler. In token IDs đầu mỗi step trên mỗi rank để xác nhận.
3. Tăng NCCL timeout: `NCCL_TIMEOUT=1800` (30 phút) trong env. Nhưng đây là patch, không phải fix.

## Bug 5: MFU thấp bất ngờ

**Triệu chứng**: MFU 20-30% thay vì kỳ vọng 50-60%.

**Nguyên nhân**:

1. NCCL không dùng ring/tree algorithm tối ưu. NCCL tự chọn theo topology nhưng có thể sai.
2. Collective không overlap với compute. Có barrier không cần.
3. Kernel CUDA chưa fused.

**Fix**:

1. `NCCL_ALGO=ring` hoặc `tree`. Thử cả hai, đo.
2. Profile trace, tìm barrier. Thường do `.item()`, `.cpu()`, hoặc `loss.backward()` không async.
3. Dùng FlashAttention: `torch.nn.functional.scaled_dot_product_attention` đã có FlashAttention backend, đảm bảo PyTorch version mới.

## Bug 6: Checkpoint không restore đúng

**Triệu chứng**: load checkpoint, loss khác xa baseline.

**Nguyên nhân**:

1. Save trên rank 0 chỉ, không sync với các rank khác.
2. DTensor checkpoint save full tensor, load không shard lại đúng.
3. Optimizer state shard không khớp với DTensor sharding mới.

**Fix**:

1. Dùng `torch.distributed.checkpoint` API, không phải `torch.save`. API mới handle distributed.
2. Đảm bảo save và load dùng cùng plan parallelism.
3. Nếu thay đổi TP size, phải resharding checkpoint. Đây là task riêng, không tự động.

## Bug 7: Gradient không update một số parameter

**Triệu chứng**: training chạy, một số layer không học (weight không thay đổi).

**Nguyên nhân**:

1. Một parameter không nằm trong `model.parameters()`.
2. `requires_grad=False` accident.
3. DTensor parameter không được FSDP track.

**Fix**:

1. In `[(n, p.requires_grad) for n, p in model.named_parameters()]`, verify mọi parameter `requires_grad=True`.
2. Verify FSDP wrap đúng module chứa parameter đó. `fully_shard(model)` ở root nên cover tất cả.
3. Quan sát gradient norm mỗi parameter: `grad_norm = p.grad.full_tensor().norm()`. Nếu 0 hoặc NaN, có bug.

## Checklist debug

Khi gặp vấn đề, theo thứ tự:

1. **Reproduce trên 1 GPU**: nếu lỗi không xảy ra, đây là vấn đề parallelism.
2. **Reproduce trên 2 GPU TP=2**: nếu lỗi ở đây, plan TP sai.
3. **Reproduce trên 2 GPU FSDP=2**: nếu lỗi ở đây, plan FSDP hoặc trật tự sai.
4. **Reproduce trên 4 GPU 2x2**: nếu lỗi ở đây mà 1, 2, 3 OK, đây là 2D parallelism bug.
5. **Print placement của tensor**: ở các điểm nghi vấn, in `tensor.placements` và `tensor.shape`. Đối chiếu với lý thuyết.

Print template:

```python
def debug_dtensor(name, t):
    if hasattr(t, 'placements'):
        print(f"{name}: placements={t.placements}, shape={t.shape}, local_shape={t.to_local().shape}")
    else:
        print(f"{name}: regular tensor, shape={t.shape}")
```

Đặt `debug_dtensor("after wq", xq)` ở nơi nghi vấn, run trên một rank. So sánh với expected placement.

## Một lời khuyên cuối

Bug parallelism khó hơn bug "thường" vì:

- Không reproducible đơn giản (cần $\ge 2$ GPU).
- Stack trace thường chỉ ra một dòng nhưng nguyên nhân ở chỗ khác.
- Race condition giữa các rank.

Hãy:

- Test trên cấu hình nhỏ nhất có thể (2 GPU thường đủ).
- Log nhiều, đặc biệt placement và shape.
- So sánh với baseline single-GPU bất cứ khi nào có thể (loss ở step 0 phải khớp, dù TP hay không).

Phần 9 và toàn bộ chuỗi bài giảng kết thúc ở đây. Bạn đã đi từ "Tensor Parallelism là gì" đến đọc được code production LLM training. Chúc bạn áp dụng tốt.
