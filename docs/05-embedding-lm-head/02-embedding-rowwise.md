---
title: Embedding shard theo vocab
---

# Embedding shard theo vocab

`nn.Embedding(V, K)` chứa một bảng tra $E \in \mathbb{R}^{V \times K}$. Khi gọi `E(token_ids)` với `token_ids` shape $(B, S)$, kết quả là tensor shape $(B, S, K)$ trong đó row thứ $i$ của output là $E[\text{token\_id}_i]$. Đây là phép gather theo index.

## Vì sao shard theo hàng (vocab) là tự nhiên

Hai cách shard bảng $E$: theo hàng (chiều $V$) hoặc theo cột (chiều $K$).

Shard theo cột (chia $K$ thành $K/P$ phần). Mỗi rank giữ một subset chiều embedding của mọi token. Lookup vẫn cho ra tensor đầy đủ về vocab, nhưng shape là $(B, S, K/P)$ trên mỗi rank. Vấn đề: output cần shard cuối, đi vào block đầu tiên (đang chờ replicate hoặc shard sequence). Để feed vào attention, ta phải all-gather toàn bộ embedding. Tốn collective lớn.

Shard theo hàng (chia $V$ thành $V/P$ phần). Rank $r$ giữ $E^{(r)} \in \mathbb{R}^{V/P \times K}$, là các hàng từ $r \cdot V/P$ đến $(r+1) V/P - 1$. Khi lookup `token_id`, chỉ rank "đúng" (rank $\lfloor \text{token\_id} \cdot P / V \rfloor$) có row đó, các rank khác trả về 0. Sau lookup, AllReduce (cộng) trên các rank gom lại được embedding đầy đủ. Một collective AllReduce, tensor $(B, S, K)$.

Shard hàng tốt hơn về chi phí (AllReduce nhỏ vì shape $K$, không phải $K/P$ rồi all-gather thành $K$). Đây là pattern chuẩn.

## Forward derivation cho Embedding Row Parallel

Cho `token_id` (shape $(B, S)$) replicate trên TP mesh.

Trên rank $r$:

$$
E^{(r)}[t] = \begin{cases} E[t] & \text{nếu } r \cdot V/P \le t < (r+1) \cdot V/P \\ 0 & \text{nếu không} \end{cases}
$$

Mỗi rank tự thực hiện lookup. Token ID nằm ngoài range của rank này trả về 0 (PyTorch xử lý bằng mask).

Sau lookup, tensor local $H^{(r)} \in \mathbb{R}^{B \times S \times K}$, có chỉ một phần các vị trí $(b, s)$ chứa giá trị thực (nơi token thuộc range của rank). Phần còn lại là 0.

Tổng tất cả các $H^{(r)}$ trên các rank cho ra embedding đầy đủ:

$$
H = \sum_r H^{(r)}
$$

vì với mỗi $(b, s)$, đúng một rank đóng góp giá trị thực, các rank khác đóng góp 0. AllReduce thực hiện phép tổng này. Sau AllReduce, $H$ replicate trên mọi rank.

## Trong code PyTorch

```python
"tok_embeddings": RowwiseParallel(input_layouts=Replicate()),
```

Đây là dòng trong `parallelism.py` của toy code. `RowwiseParallel` cho `nn.Embedding`:

- Shard `weight` theo dim 0 (chiều $V$).
- Input layout `Replicate` (token_ids giống nhau trên mọi rank).
- Output layout mặc định: shard hoặc partial tùy cấu hình; với toy code, output sẽ được PrepareModuleInput của layer đầu xử lý tiếp.

PyTorch DTensor xử lý phép out-of-range bằng cách map token id thành 0 trong lookup local, không gây index out of bounds.

## Backward

Gradient của embedding output là tensor shape $(B, S, K)$ replicate (đến từ block kế tiếp). Trên mỗi rank, gradient này được scatter ngược trở lại vào $\partial L / \partial E^{(r)}$ tại các row tương ứng với token_id thuộc range của rank, các row khác giữ 0.

Vì mỗi rank chỉ giữ shard $E^{(r)}$, gradient của $E$ tự nhiên cũng shard hàng. Không có collective trong backward của embedding lookup, ngoại trừ nếu cần broadcast gradient input cho upstream (không có ở đây vì token_ids không có gradient).

## Bộ nhớ và chi phí

Parameter mỗi rank: $V K / P$. Với $V = 128000, K = 4096, P = 8$: $V K / P = 65$ M params, $\approx 260$ MB ở fp32. Baseline không TP: $524$ M params, $\approx 2$ GB. Tiết kiệm $8$ lần.

Forward collective: 1 AllReduce trên tensor $(B, S, K)$, kích thước $B \cdot S \cdot K \cdot 4$ bytes. Với $B = 8, S = 2048, K = 4096$: 256 MB mỗi forward. Đây là cùng order với AllReduce của MLP và Attention, không phải bottleneck đặc biệt.

## Sai lầm thường gặp

Sai lầm 1: shard embedding theo cột ($K$). Như đã phân tích, sẽ tốn all-gather thay vì all-reduce, và làm hỏng pattern Sequence Parallel sau này.

Sai lầm 2: quên `input_layouts=Replicate()`. Nếu không nói, PyTorch có thể giả định input đang shard, dẫn đến lỗi placement.

Sai lầm 3: dùng `Embedding` thay `nn.Embedding` (tức module custom). `RowwiseParallel` chỉ hiểu một số module chuẩn. Với module custom, bạn phải tự implement DTensor parameter và forward wrapper.

Sai lầm 4: nghĩ rằng Embedding rowwise + AllReduce là đắt. Thực ra với $V$ rất lớn (đến 256K), tiết kiệm bộ nhớ thường vượt xa chi phí AllReduce.

Chương tiếp ta sang LM Head và `loss_parallel`.
