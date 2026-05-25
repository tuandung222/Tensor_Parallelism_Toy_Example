---
title: Weight tying trong TP
---

# Weight tying trong TP

Weight tying là kỹ thuật chia sẻ trọng số giữa Embedding (input) và LM Head (output): cùng một ma trận $E \in \mathbb{R}^{V \times K}$ vừa dùng để lookup token, vừa dùng để chiếu hidden thành logit (lúc đó dùng $E^\top$). Đây là kỹ thuật phổ biến với mô hình nhỏ (GPT-2, BERT) vì tiết kiệm $V \cdot K$ tham số và giúp regularize.

Trong TP, weight tying gặp một vấn đề: Embedding shard hàng còn LM Head shard cột. Hai placement này không tương thích cho cùng một tensor vật lý.

## Vì sao xung đột placement

Embedding tự nhiên shard hàng $V$ (mỗi rank giữ một subset vocab). LM Head tự nhiên shard cột $V$ (cùng nhận xét).

Nhìn về placement của ma trận $E$:

- Trong vai trò Embedding lookup: $E$ shape $(V, K)$, shard chiều 0 ($V$).
- Trong vai trò LM Head matmul: $E^\top$ shape $(K, V)$, shard chiều 1 ($V$).

Chiều $V$ trong cả hai trường hợp là chiều shard. Nếu ta đặt $E$ với placement $\mathrm{Shard}(0)$, thì:

- Embedding: đúng, shard hàng. OK.
- LM Head: cần $E^\top$. Transpose của tensor shard chiều 0 trở thành tensor shard chiều 1. OK về toán, nhưng PyTorch phải xử lý transpose của DTensor sao cho placement phù hợp.

Tin tốt: cả hai phép đều shard chiều $V$, không xung đột. Tin chưa tốt: cách PyTorch implement Embedding RowwiseParallel và Linear ColwiseParallel có thể không hoàn toàn nhất quán với cùng một tensor.

## Cách giải đúng

Cách đầu, dùng `parametrize` hoặc share parameter qua một wrapper. Đặt $E$ với placement chuẩn của Embedding (Shard(0)). Khi cần dùng làm LM Head, gọi forward bằng cách thực hiện $H \cdot E^\top$ bằng matmul thuần, để PyTorch DTensor tự xử lý transpose.

Cách hai, đơn giản hơn: **không tie** trong TP. Tách $E_{in}$ và $W_{out}$ thành hai tensor riêng, có thể vẫn share initialization nhưng training tách rời. Llama-3, Mistral, Qwen đều dùng untied. Lý do: với $V$ rất lớn, tiết kiệm parameter từ tying ít hơn so với phức tạp của TP-aware sharing.

Cách ba: tie nhưng accept một phép all-gather trong LM Head để có $E$ đầy đủ rồi transpose, hi sinh hiệu năng. Không khuyến nghị.

## Trường hợp thực tế

GPT-2 tied. Khi port GPT-2 sang TP, cộng đồng thường untie luôn, accept tăng tham số. Mô hình nhỏ ($V \cdot K$ vài chục triệu) chấp nhận được.

Llama-3, Mistral, Qwen, DeepSeek đều untied. Với $V \cdot K = 5 \cdot 10^8$ tham số ở Llama-3 8B, chỉ chiếm $5/8 \cdot 100\% \approx 6\%$ tổng tham số mô hình. TP shard chia 8 còn $0.7\%$, không phải bottleneck.

Decoder của T5 và một vài mô hình encoder-decoder vẫn tie. Khi train với TP, cần wrapper riêng.

## Cài đặt khi cần tie

Pseudocode khái niệm:

```python
class TiedEmbedding(nn.Module):
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(vocab_size, dim))

    def embed(self, tokens):
        return F.embedding(tokens, self.weight)

    def logits(self, hidden):
        return hidden @ self.weight.t()
```

Trong plan TP:

```python
plan = {
    "tied_emb.weight": ... # shard hàng vocab
}
```

Phép `self.weight.t()` trên DTensor `Shard(0)` cho ra DTensor `Shard(1)`, đúng cho LM Head Column Parallel. PyTorch hỗ trợ `transpose` tự nhiên.

Backward gradient với `self.weight` đến từ hai nguồn: backward qua `embed` (chỉ chạm các row tương ứng token_id) và backward qua `logits` (chạm mọi row, vì matmul đầy đủ). PyTorch tự cộng gradient từ hai đường.

## Tóm tắt khuyến nghị

| Tình huống | Khuyến nghị |
|------------|--------------|
| LLM hiện đại lớn ($V \ge 32000$) | Untied, dùng plan TP đơn giản |
| Model nhỏ port từ GPT-2 | Untied luôn khi áp TP |
| Tied bắt buộc (T5 decoder) | Wrapper custom, transpose DTensor |

Phần lớn dự án thực tế chọn untied. Đây là một trade-off giữa "vài phần trăm tham số" và "đơn giản TP" mà cộng đồng đã chọn đơn giản TP.

Chương tiếp ta walkthrough `tok_embeddings` và `output` trong toy code Llama.
