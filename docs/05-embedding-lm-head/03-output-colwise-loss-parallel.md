---
title: LM Head và loss_parallel
---

# LM Head và `loss_parallel`

LM Head là linear cuối cùng của một LLM, biến hidden state shape $(B, S, K)$ thành logits shape $(B, S, V)$:

$$
\mathrm{logits} = H \cdot W_{out}, \quad W_{out} \in \mathbb{R}^{K \times V}
$$

Trên LLM lớn, $V$ rất lớn (Llama-3: 128K, Qwen: 256K), nên LM Head chiếm vài trăm MB params và cho ra tensor logits rất lớn. Tensor logits $(B, S, V)$ với $B = 8, S = 2048, V = 128000$ chiếm khoảng 8 GB ở fp32. Đây là lý do TP cho LM Head là bắt buộc.

## Shard cột (vocab)

Pattern tự nhiên: shard $W_{out}$ theo cột $V$. Mỗi rank giữ $W_{out}^{(r)} \in \mathbb{R}^{K \times V/P}$, là một slice của vocab dim.

Forward trên rank $r$, với input $H$ replicate (hoặc shard sequence + đã gathered):

$$
\mathrm{logits}^{(r)} = H \cdot W_{out}^{(r)} \in \mathbb{R}^{B \times S \times V/P}
$$

Đây là Column Parallel. Output shard cuối ($V/P$).

**Lựa chọn quan trọng**: ta có thể all-gather logits để được tensor $(B, S, V)$ đầy đủ, rồi tính loss bình thường. Nhưng:

1. Tensor đầy đủ rất lớn (vài GB), tốn bộ nhớ peak.
2. AllGather tốn bandwidth.
3. Sau softmax + cross-entropy ta chỉ cần một scalar loss, không cần full logits.

Vậy có cách nào tính cross-entropy trực tiếp trên logits shard cuối, mà không cần all-gather. Câu trả lời: `loss_parallel`.

## Cross-entropy phân tán

Cross-entropy với one-hot label tại token $t^*$:

$$
\mathcal{L} = -\log p(t^*) = -\log \frac{\exp(z_{t^*})}{\sum_v \exp(z_v)} = -z_{t^*} + \log \sum_v \exp(z_v)
$$

Số hạng $z_{t^*}$ là logit tại vị trí label, một scalar. Số hạng $\log \sum_v \exp(z_v)$ là log-sum-exp (LSE) trên toàn vocab.

Ý tưởng `loss_parallel`: tách LSE thành hai bước.

Bước 1, mỗi rank tính LSE local trên shard của mình:

$$
\mathrm{lse}^{(r)} = \log \sum_{v \in \text{rank } r} \exp(z_v)
$$

Bước 2, để gom các LSE local thành LSE toàn cục:

$$
\log \sum_v \exp(z_v) = \log \sum_r \exp(\mathrm{lse}^{(r)})
$$

Đẳng thức này đúng nhờ luật $\log \sum \exp$. Để tính, ta AllReduce với operation max (cho ổn định số) hoặc dùng một phép all-reduce phức tạp hơn. Trong thực tế PyTorch cài bằng cách: thu thập tất cả `lse^(r)`, rồi tính `log sum exp` trên chúng.

Bước 3, term $z_{t^*}$: chỉ rank nắm vocab range chứa $t^*$ có giá trị này. Các rank khác đóng góp 0. AllReduce gom lại được $z_{t^*}$ đầy đủ. (Đây là một scalar mỗi position, kích thước nhỏ.)

Cuối cùng, loss = $-z_{t^*} + \mathrm{LSE}$, một scalar mỗi position, tổng hoặc trung bình ra một loss.

Toàn bộ quá trình chỉ tốn AllReduce trên các đại lượng nhỏ (kích thước $B \cdot S$, không phải $B \cdot S \cdot V$). Không cần tensor logits đầy đủ.

## `loss_parallel` trong PyTorch

```python
from torch.distributed.tensor.parallel import loss_parallel

with loss_parallel():
    return F.cross_entropy(output.reshape(-1, output.size(-1)), labels.reshape(-1))
```

Đây là context manager. Khi `cross_entropy` được gọi trên DTensor logits shard cuối, PyTorch tự thay implementation thường bằng implementation parallel.

Backward cũng cần `loss_parallel` để biết đường tính gradient đúng:

```python
def backward(self, *args, **kwargs):
    with loss_parallel():
        super().backward(*args, **kwargs)
```

Đây chính là code trong `02_large_language_model/train.py`.

## Configurations trong toy code

```python
"output": ColwiseParallel(
    input_layouts=Shard(1),
    output_layouts=Shard(-1),
    use_local_output=False,
),
```

Giải nghĩa từng option:

- `input_layouts=Shard(1)`: input của LM head đến từ `norm`, ở dạng shard chiều sequence (do Sequence Parallel, Phần 6). LM Head trước tiên all-gather để có input replicate, rồi mới matmul. (Nếu input đã replicate thì không cần.)
- `output_layouts=Shard(-1)`: output logits shard theo chiều cuối ($V$). Đây là Column Parallel chuẩn.
- `use_local_output=False`: giữ output dưới dạng DTensor, không unwrap về tensor thường. Cần thiết để `loss_parallel` nhận diện được tensor shard và áp implementation parallel.

## Bộ nhớ tiết kiệm bao nhiêu

Logits đầy đủ $(B, S, V) = (8, 2048, 128000) \approx 8$ GB ở fp32. Với TP=8: $8/8 = 1$ GB mỗi rank. Đây là tiết kiệm activation memory đáng kể, không tính tiết kiệm parameter.

Hơn nữa, gradient của logits cũng cùng shape. Không có `loss_parallel`, gradient $\nabla L$ với respect logits là một tensor $(B, S, V)$ đầy đủ phải truyền backward. Có `loss_parallel`, gradient cũng được giữ shard, $1/P$ kích thước.

## Sai lầm thường gặp

Sai lầm 1: gọi `cross_entropy` không có `loss_parallel`. PyTorch sẽ implicit all-gather logits để tính. Hoặc trả lỗi placement mismatch.

Sai lầm 2: quên `use_local_output=False`. Output unwrap về tensor thường ngay, `loss_parallel` không hoạt động.

Sai lầm 3: quên gọi `loss_parallel` trong backward. Forward chạy ra loss đúng, nhưng backward không biết gradient của logits đang shard cuối, gradient tính sai.

Sai lầm 4: dùng custom loss (label smoothing, focal loss, ...) trên DTensor shard cuối mà không adapt. Custom loss thường gọi `softmax` rồi nhân với label, nếu không `loss_parallel`-ize sẽ all-gather. Phải implement parallel-aware nếu cần performance.

Chương tiếp ta bàn về weight tying.
