---
title: GQA và MQA với Tensor Parallel
---

# GQA và MQA với Tensor Parallel

Llama-3, Mistral, Qwen, và phần lớn LLM hiện đại không dùng Multi-Head Attention thuần (MHA), mà dùng Grouped Query Attention (GQA) hoặc, ở thái cực, Multi-Query Attention (MQA). Đây là biến thể tiết kiệm bộ nhớ K/V cache khi inference, và cũng có ảnh hưởng đến cách shard TP.

## Nhắc lại GQA và MQA

Trong MHA thuần, mỗi Q head có một K head và V head riêng. Số $n_{kv\_heads} = n_{heads}$.

Trong GQA, ta giữ $n_{heads}$ Q head nhưng giảm xuống $n_{kv\_heads} < n_{heads}$ K/V head. Mỗi K/V head được chia sẻ giữa $n_{rep} = n_{heads} / n_{kv\_heads}$ Q head liên tiếp. Llama-3 8B có $n_{heads} = 32, n_{kv\_heads} = 8$, tức mỗi K/V head phục vụ 4 Q head.

Trong MQA, $n_{kv\_heads} = 1$. Tất cả Q head dùng chung một cặp K/V.

Lợi ích chính: K/V cache khi inference giảm $n_{rep}$ lần. Với context dài 100K token, đây là khác biệt giữa OOM và chạy được.

## `repeat_kv` trong code

Trong `model.py` của repo:

```python
def repeat_kv(x, n_rep):
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )
```

Hàm này nhân bản K/V cho đủ với $n_{heads}$ Q head bằng `expand` rồi `reshape`. Đây là cách rẻ vì `expand` không sao chép memory thực (chỉ thay stride), `reshape` cũng chỉ thay view.

Quan trọng: `repeat_kv` xảy ra **sau khi** $W_K, W_V$ đã tính, nhưng **trước khi** vào `scaled_dot_product_attention`. Nó không phải là một linear, mà chỉ là một biến hình tensor.

## Điều kiện ràng buộc cho TP với GQA

Với TP $P$, ta chia $n_{heads}$ Q head cho các rank: $n_{heads}^{local} = n_{heads} / P$. Tương tự, ta chia $n_{kv\_heads}$ cho các rank: $n_{kv\_heads}^{local} = n_{kv\_heads} / P$.

Ràng buộc số học bắt buộc:

$$
P \mid n_{heads}, \qquad P \mid n_{kv\_heads}
$$

Cả hai phép chia phải lấy kết quả nguyên. Nếu một trong hai không chia hết, TP không hợp lệ.

Ví dụ với Llama-3 8B ($n_{heads} = 32, n_{kv\_heads} = 8$):

- $P = 2$: hợp lệ (32/2=16, 8/2=4)
- $P = 4$: hợp lệ (32/4=8, 8/4=2)
- $P = 8$: hợp lệ (32/8=4, 8/8=1)
- $P = 16$: $n_{kv\_heads}/P = 0.5$, **không hợp lệ**.

Ràng buộc thứ hai bắt nguồn từ GQA: TP size bị giới hạn bởi $n_{kv\_heads}$, không phải $n_{heads}$. Với MQA ($n_{kv\_heads} = 1$), TP chỉ chạy được với $P = 1$, tức không TP. Đó là một bất lợi của MQA mà GQA ra đời để khắc phục.

## Plan TP cho GQA

Pattern Col-Col-Col-Row vẫn áp dụng nguyên vẹn:

```python
"attention.wq": ColwiseParallel(),
"attention.wk": ColwiseParallel(),
"attention.wv": ColwiseParallel(),
"attention.wo": RowwiseParallel(),
```

Khác biệt duy nhất so với MHA: shape của $W_K, W_V$ khác. $W_K \in \mathbb{R}^{K \times n_{kv\_heads} \cdot d_k}$, không phải $\mathbb{R}^{K \times K}$. Số cột của $W_K$ là $n_{kv\_heads} \cdot d_k = K \cdot n_{kv\_heads} / n_{heads}$, nhỏ hơn $K$ một hệ số $n_{rep}$.

Khi shard Column cho $W_K$ trên $P$ rank: mỗi rank giữ $W_K^{(r)} \in \mathbb{R}^{K \times (n_{kv\_heads}/P) \cdot d_k}$. Để chia đều, cần $P \mid n_{kv\_heads}$ như đã nêu.

## Cập nhật `n_heads` và `n_kv_heads`

Như chương trước đã nhắc:

```python
attn_layer.n_heads = attn_layer.n_heads // tp_mesh.size()
attn_layer.n_kv_heads = attn_layer.n_kv_heads // tp_mesh.size()
```

Cả hai phải được cập nhật. Đặc biệt với GQA, nếu bạn quên giảm `n_kv_heads`, hàm `repeat_kv` sẽ tạo ra tensor có $n_{kv\_heads} \cdot n_{rep} \neq n_{heads}^{local}$ đối với code local, crash ngay.

Cũng phải xem `n_rep` được tính lại đúng chưa. Trong `Attention.__init__`:

```python
self.n_rep = self.n_heads // self.n_kv_heads
```

Vì cả $n_{heads}$ và $n_{kv\_heads}$ cùng bị chia $P$, tỉ số $n_{rep}$ vẫn giữ nguyên. Nhưng nếu code tính lại `n_rep` sau khi đã chỉnh `n_heads`, kết quả cũng đúng. May mắn là toy code chỉ tính `n_rep` một lần trong `__init__`, sau đó giữ.

## `repeat_kv` trong môi trường DTensor

Câu hỏi tinh tế: `repeat_kv` dùng `expand` và `reshape`, các phép này có hoạt động đúng trên DTensor không.

Trả lời: vì $K$ và $V$ là tensor thường (sau khi gọi `.view` và `transpose` để bỏ wrap DTensor), `repeat_kv` chạy trên tensor local của mỗi rank. PyTorch không cần biết tensor đó từng là DTensor, chỉ cần biết shape local hiện tại. Với điều kiện ràng buộc trên, mỗi rank có $K, V$ shape $(B, S, n_{kv\_heads}/P, d_k)$, sau `repeat_kv` trở thành $(B, S, n_{heads}/P, d_k)$. Phù hợp để vào `scaled_dot_product_attention`.

Đây là chỗ DTensor "ẩn mình": sau khi đã shard và view ra tensor thường, ta làm gì cũng được trên tensor local. PyTorch chỉ can thiệp khi ta gọi linear lại (như `wo`), khi đó tensor được tự động wrap lại thành DTensor để xử lý Row Parallel.

## Sai lầm thường gặp

Sai lầm 1: dùng MQA với TP $P > 1$. Vì $n_{kv\_heads} = 1$, không chia hết. PyTorch sẽ trả lỗi tại `parallelize_module`.

Sai lầm 2: chọn $P$ chia hết $n_{heads}$ nhưng không chia hết $n_{kv\_heads}$. Ví dụ $n_{heads} = 32, n_{kv\_heads} = 4, P = 8$. PyTorch crash khi shard $W_K$.

Sai lầm 3: quên cập nhật `n_kv_heads` sau parallelize. Forward chạy được nhưng `repeat_kv` tạo tensor sai shape, hoặc tệ hơn là sai số học mà không trả lỗi rõ ràng.

Sai lầm 4: nghĩ rằng GQA "nhẹ" hơn nên TP rẻ hơn. Sai, vì các Q head vẫn shard đầy đủ và phần lớn FLOPs nằm ở $Q K^\top$ và $\mathrm{score} \cdot V$, được tính theo $n_{heads}$. GQA chỉ giảm bộ nhớ K/V, không giảm compute.

## Tổng kết

Pattern TP cho GQA = pattern MHA + điều kiện chia hết kép. Một nguyên tắc nhỏ nhưng quan trọng khi thiết kế kiến trúc cho TP thân thiện: chọn $n_{kv\_heads}$ là bội số lớn của số TP rank dự kiến. Llama-3 chọn $n_{kv\_heads} = 8$ cho phép TP 1, 2, 4, 8. Mistral chọn $n_{kv\_heads} = 8$ tương tự. Đây không phải tình cờ.

Chương tiếp theo ta walkthrough `Attention.forward` đầy đủ, kết nối tất cả các điểm: rotary, repeat_kv, scaled_dot_product_attention, view, transpose, đều thông qua lăng kính DTensor.
