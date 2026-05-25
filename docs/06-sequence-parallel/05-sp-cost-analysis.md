---
title: Phân tích chi phí SP
---

# Phân tích chi phí Sequence Parallel

Chương cuối Phần 6 tổng kết chi phí giao tiếp, bộ nhớ, và compute của SP, đối chiếu trực tiếp với TP cổ điển. Mục tiêu: cho bạn công cụ để quyết định có cần bật SP cho project của mình hay không.

## Chi phí giao tiếp

Một block với $P = $ TP/SP size, kích thước tensor sequence $T = B \cdot S \cdot K$.

**TP cổ điển**: 2 AllReduce mỗi block mỗi forward. Tổng forward + backward: 4 AllReduce. Mỗi AllReduce ring chuyển $\frac{2(P-1)}{P} T$ phần tử qua mạng.

**SP + TP**: 2 AllGather + 2 ReduceScatter mỗi block forward. Tổng forward + backward: 4 AllGather + 4 ReduceScatter. Mỗi AllGather/ReduceScatter ring chuyển $\frac{P-1}{P} T$ phần tử.

So sánh tổng bytes chuyển:

- TP cổ điển: $4 \cdot \frac{2(P-1)}{P} T = \frac{8(P-1)}{P} T$.
- SP + TP: $4 \cdot \frac{P-1}{P} T + 4 \cdot \frac{P-1}{P} T = \frac{8(P-1)}{P} T$.

**Bằng nhau.** SP không làm tăng giao tiếp.

Lý do bản chất: AllReduce trong ring algorithm = ReduceScatter (chuyển $\frac{P-1}{P} T$) + AllGather (chuyển $\frac{P-1}{P} T$). SP chỉ tách AllReduce thành hai bước riêng, giữa hai bước tensor ở placement Shard giúp tiết kiệm activation.

## Chi phí bộ nhớ activation

Đây là phần SP thắng đậm.

Định nghĩa activation footprint của một block là tensor lớn nhất tồn tại cùng lúc trong block đó.

**TP cổ điển**:

- Residual $(B, S, K)$ replicate: $T$ phần tử mỗi rank.
- Hidden FFN $(B, S, H_{ffn})$ shard cuối: $T \cdot H_{ffn}/K / P$ mỗi rank, thường $\approx 4T/P$ (với $H_{ffn} = 4K$).
- Q/K/V $(B, S, K)$ shard head: $T/P$ mỗi rank.
- Scores $(B, H, S, S)$ shard head: $B H S^2 / P$ mỗi rank.

Tổng activation per block per rank: $\Theta(T) + \Theta(T/P) + \Theta(B H S^2 / P)$. Term đầu **không** chia $P$.

**SP + TP**:

- Residual $(B, S/P, K)$ shard sequence: $T/P$ mỗi rank.
- Hidden FFN: vẫn $\Theta(4T/P)$.
- Q/K/V: vẫn $T/P$.
- Scores: vẫn $\Theta(B H S^2 / P)$.

Tổng: $\Theta(T/P) + \Theta(4T/P) + \Theta(B H S^2 / P)$. **Mọi term chia $P$**.

Tiết kiệm chính: term residual giảm $P$ lần. Đây là term lớn nhất nếu $S$ vừa hoặc $S$ rất dài (term scores chỉ thắng nếu $S^2 H \gg K T = K B S K$, tức $S H \gg B K^2$, hiếm khi).

## Chi phí compute

Compute tổng không đổi: phép tính trong block vẫn giống nhau, chỉ phân bố khác.

Tổng FLOPs mỗi block:

- QKV projection + output projection: $\Theta(4 B S K^2)$.
- Attention $Q K^\top$ và $\mathrm{score} V$: $\Theta(2 B S^2 K)$.
- FFN: $\Theta(2 B S K H_{ffn})$.

Mỗi rank gánh $1/P$ tổng FLOPs, vẫn vậy với SP. Compute không đổi.

Norm: bản thân nó chỉ $\Theta(B S K)$, nhỏ. SP làm norm chạy trên tensor $(B, S/P, K)$, nhanh hơn $P$ lần, nhưng vì norm là phần nhỏ FLOPs, lợi ích không đáng kể.

## Khi nào SP đáng dùng

SP đáng dùng khi residual activation chiếm tỉ trọng lớn. Cụ thể:

- **Sequence rất dài**: $S \ge 4K$ thường đã đủ động lực, $S = 32K$ là bắt buộc.
- **Batch nhỏ**: với LLM training, batch thường nhỏ trên mỗi GPU (sau khi chia DP), nên residual càng dominates.
- **Hidden dim không quá lớn**: $K = 4096$ (Llama-3 8B), $K = 8192$ (lớn hơn) đều SP-friendly.

SP không lợi nhiều khi:

- $S$ nhỏ ($S < 1024$): residual không phải bottleneck.
- Model rất rộng ($K = 16384$ trở lên): hidden FFN và scores chiếm phần lớn, residual nhỏ tương đối.

Tỉ lệ thông minh:

$$
\text{ratio} = \frac{T}{T/P + 4T/P + B H S^2 / P} \approx \frac{1}{(5 + B H S / K) / P}
$$

Nếu $B H S / K \ge 1$ (sequence dài), residual không dominate, SP ít lợi. Nếu ngược lại (sequence vừa, batch nhỏ), SP rất lợi.

## Tích hợp với checkpoint

SP và Activation Checkpointing (Phần 9) compose tốt. Activation Checkpoint giảm activation thêm $\sqrt{L}$ lần (rule of thumb), SP giảm residual $P$ lần. Hai kỹ thuật áp dụng trên các zone khác nhau:

- SP nhắm residual (zone ngoài attention/FFN).
- Checkpoint nhắm Q/K/V, scores, hidden FFN (zone trong).

Tổng tiết kiệm: $P \cdot \sqrt{L}$. Với $P = 8, L = 32$: tiết kiệm $45$ lần activation memory tổng. Đây là combination chuẩn để train Llama-3 8B với context dài trên GPU 80 GB.

## Tóm tắt Phần 6

| Đại lượng | TP | TP + SP |
|-----------|------|----------|
| Parameter mỗi rank | $1/P$ | $1/P$ |
| Activation residual mỗi rank | full | $1/P$ |
| Activation in-block mỗi rank | $1/P$ | $1/P$ |
| Bytes chuyển mỗi block per step | $\frac{8(P-1)}{P} T$ | $\frac{8(P-1)}{P} T$ |
| Compute mỗi rank | $1/P$ | $1/P$ |
| Số collective | 4 mỗi block | 8 mỗi block |

SP **không miễn phí** ở góc nhìn số collective (gấp đôi), nhưng tổng bytes chuyển bằng. Trên NCCL, mỗi collective có overhead khởi tạo nhỏ ($\sim$ vài $\mu s$), nên SP thường chậm hơn TP một chút ($\le 5\%$) ở các block ngắn. Với block dài, hiệu ứng này tan biến.

Khuyến nghị: bật SP cho mọi LLM training với $S \ge 4K$, hoặc bất cứ khi nào activation memory là bottleneck.

Phần 6 kết thúc. Phần 7 sẽ chuyển sang 2D parallelism: kết hợp TP/SP với FSDP để chia thêm parameter và optimizer state theo chiều data parallel.
