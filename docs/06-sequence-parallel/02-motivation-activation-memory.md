---
title: Motivation, activation memory
---

# Vì sao Sequence Parallel quan trọng

Trước khi giới thiệu cơ chế SP, ta cần thấy rõ vấn đề mà nó giải. Phần này phân tích chi tiết activation memory cho Transformer LLM với TP cổ điển và tại sao chiều sequence trở thành bottleneck mới.

## Activation memory đến từ đâu

Trong một forward pass với autograd, framework lưu các tensor trung gian để dùng cho backward. Các "checkpoint" này tạo nên activation memory.

Trong một Transformer block, các activation chính cần lưu:

1. **Input của attention** (sau norm 1): shape $(B, S, K)$.
2. **Q, K, V** (sau projection): shape $(B, S, H, d_k)$.
3. **Attention scores**: shape $(B, H, S, S)$. **Đây là cái lớn nhất với sequence dài**.
4. **Output của attention** (trước $W_O$): shape $(B, S, K)$.
5. **Residual sau attention**: shape $(B, S, K)$.
6. **Input của FFN** (sau norm 2): shape $(B, S, K)$.
7. **Hidden FFN** (sau $W_1, W_3$): shape $(B, S, H_{ffn})$.
8. **Output FFN**: shape $(B, S, K)$.
9. **Residual sau FFN**: shape $(B, S, K)$.

Tổng order per block ở fp32: $\Theta(B \cdot S \cdot K) + \Theta(B \cdot H \cdot S^2)$. Với $L$ block, nhân $L$ lần.

## TP cổ điển giúp gì

TP shard parameter $P$ lần và shard một số activation $P$ lần (Q, K, V, hidden FFN). Nhưng **residual và input/output của mỗi block vẫn replicate**: chúng được duy trì ở placement Replicate giữa các block để có thể feed vào block sau.

Cụ thể, sau AllReduce của Row Parallel cuối mỗi linear, tensor là replicate. Residual cũng replicate. Norm cũng replicate. Vậy:

- Bộ nhớ Q, K, V: chia $P$.
- Bộ nhớ scores: chia $P$ (chia theo head).
- Bộ nhớ hidden FFN: chia $P$.
- **Bộ nhớ residual + norm + input/output block: KHÔNG chia**, vẫn full $(B, S, K)$.

Với LLM context dài, residual + norm + input/output (tổng cộng vài tensor $(B, S, K)$ mỗi block) chiếm phần lớn activation memory.

## Ước lượng cụ thể

Llama-3 8B với $L = 32$, $K = 4096$, $H = 32$, $d_k = 128$. Context $S = 8192$, batch $B = 1$.

Bộ nhớ residual + norm + input/output mỗi block (4 tensor $(B, S, K)$ ở fp32):

$$
4 \cdot 1 \cdot 8192 \cdot 4096 \cdot 4 \text{ bytes} = 0.5 \text{ GB}
$$

Nhân $L = 32$ block: $16$ GB.

Đây là phần **không** được TP giảm. Với $S = 32000$ (context dài Llama-3): $64$ GB. Không vừa GPU lớn nhất.

## SP giảm phần này $P$ lần

SP shard luôn chiều sequence ở các zone residual + norm + input/output. Activation tại các zone này chỉ chiếm $1/P$ kích thước. Với TP = SP = 8:

- $S = 8192$: 16 GB / 8 = 2 GB. Vừa GPU 24 GB.
- $S = 32000$: 64 GB / 8 = 8 GB. Vừa với một chút margin.

Đây không phải con số nhỏ, nó là khác biệt giữa "không train được" và "train được" cho context dài.

## Phần activation đã shard sẵn

Một số activation đã được TP cổ điển shard:

- **Q, K, V** shard theo head: $\Theta(B \cdot S \cdot K) / P$ mỗi rank.
- **Scores** shard theo head: $\Theta(B \cdot H \cdot S^2) / P$ mỗi rank.
- **Hidden FFN** shard cuối: $\Theta(B \cdot S \cdot H_{ffn}) / P$ mỗi rank.

Đây là phần "may mắn" của TP. Nhưng các tensor này chỉ tồn tại bên trong attention/FFN, không kéo dài giữa các block. Phần kéo dài (residual) chính là phần SP nhắm vào.

## Activation memory tổng kết với và không SP

| Component | Không TP, không SP | TP=8, không SP | TP=8 + SP=8 |
|-----------|--------------------|-----------------|-------------|
| Residual/norm (per block) | 0.5 GB | 0.5 GB | 0.06 GB |
| Q,K,V (per block) | 0.4 GB | 0.05 GB | 0.05 GB |
| Scores (per block) | 8 GB | 1 GB | 1 GB |
| Hidden FFN (per block) | 1.3 GB | 0.16 GB | 0.16 GB |
| **Tổng per block** | ~10 GB | ~1.7 GB | ~1.3 GB |
| **Tổng L=32 block** | 320 GB | 54 GB | 42 GB |

Cộng thêm activation checkpointing (Phần 9) chia thêm vài lần nữa, ta có thể fit context dài vào GPU 80 GB hoặc thậm chí 40 GB. Không có SP, gần như bất khả thi.

## Một nhận xét tinh tế: scores không shard sequence

Mặc dù SP shard sequence ở residual, **scores** $(B, H, S, S)$ không thể shard sequence theo cùng cách. Lý do: softmax tính trên chiều $S$ thứ hai (key dim). Nếu shard chiều $S$, mỗi rank chỉ có một phần keys, softmax sai. Đây là giới hạn nội tại của attention.

Các kỹ thuật như FlashAttention không shard scores nhưng không lưu hết scores, chỉ stream chúng. Đây là một hướng khác để giảm activation memory, song song với SP. FlashAttention + SP + TP là combination thực dụng.

Trong Phần 6 này, ta tập trung vào SP cho phần ngoài attention. FlashAttention sẽ được nhắc ở Phần 9.

Chương tiếp: cơ chế SP cho RMSNorm/LayerNorm.
