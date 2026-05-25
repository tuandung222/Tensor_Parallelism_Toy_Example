---
title: Tổng quan Phần 4
---

# Phần 4: Tensor Parallelism cho Self-Attention

Sau Phần 3, bạn đã quen pattern Column-then-Row trên MLP. Phần 4 áp dụng cùng nguyên tắc cho khối Self-Attention, khối thứ hai của một Transformer block. Tin tốt: Self-Attention vẫn dùng pattern Column-Column-Column-Row giống MLP. Tin chưa tốt: nó có thêm một chiều mới (head) và một loạt phép biến đổi shape (view, transpose, rotary embedding) làm cho việc theo dõi placement khó hơn.

Mục tiêu Phần 4 là làm cho bạn nhìn vào Multi-Head Attention và "thấy" ngay: $W_Q, W_K, W_V$ là Column Parallel theo head, $W_O$ là Row Parallel, và mọi phép trong giữa (softmax, scaled dot product, rotary) tự nhiên commute với việc shard theo head.

## Vì sao Self-Attention thân thiện với TP

Quan sát then chốt: trong Multi-Head Attention, các head độc lập với nhau. Cụ thể, head thứ $h$ tính:

$$
\mathrm{head}_h = \mathrm{softmax}\!\left( \frac{Q_h K_h^\top}{\sqrt{d_k}} \right) V_h
$$

mà không cần thông tin từ bất kỳ head nào khác. Phép concatenation cuối:

$$
\mathrm{MHA}(X) = \mathrm{Concat}(\mathrm{head}_1, \dots, \mathrm{head}_H) W_O
$$

chỉ ghép các head lại theo chiều cuối. Đây là một tài nguyên parallelism sẵn có miễn phí: nếu ta chia $H$ head cho $P$ rank ($P$ chia hết $H$), mỗi rank tự tính $H/P$ head độc lập, không cần collective ở giữa.

Chính cấu trúc multi-head này làm cho Self-Attention có thể shard theo "chiều head", và shard đó tương đương với shard cột trên $W_Q, W_K, W_V$ (vì các cột này nhóm tự nhiên theo head).

## Mục tiêu sau Phần 4

Sau bốn chương:

Thứ nhất, bạn vẽ được sơ đồ TP cho Multi-Head Attention với input $X$ replicate và output $Y$ replicate, chỉ rõ rank nào giữ head nào, và collective duy nhất nằm ở đâu.

Thứ hai, bạn giải thích được vì sao chia head, chứ không phải chia head_dim, là lựa chọn tự nhiên. Và vì sao nếu chia head_dim thì softmax bị phá.

Thứ ba, bạn xử lý được trường hợp Grouped Query Attention (GQA) và Multi-Query Attention (MQA), nơi số K/V head nhỏ hơn số Q head. Bạn biết được điều kiện ràng buộc nào trên $n_{kv\_heads}$ và $P$ để TP còn hợp lệ.

Thứ tư, bạn đọc được kế hoạch TP trong `02_large_language_model/parallelism.py` cho attention block và biết tại sao có dòng `attn_layer.n_heads = attn_layer.n_heads // tp_mesh.size()`.

## Cấu trúc bốn chương

Chương đầu (`02-multi-head-shard`) phát biểu rõ ràng phép shard theo head trên $W_Q, W_K, W_V, W_O$, derive forward và backward, đếm collective.

Chương hai (`03-qkv-output-pattern`) giải thích lựa chọn Column-Column-Column-Row, lý do vì sao $W_O$ shard hàng là duy nhất hợp lệ. Bao gồm phân tích bộ nhớ và FLOPs.

Chương ba (`04-gqa-mqa-kv-heads`) mở rộng sang GQA và MQA. Đây là tình huống thực tế của Llama-3, Mistral, Qwen. Ta sẽ chỉ ra cách `repeat_kv` ảnh hưởng đến TP, và cách cài đặt sao cho mỗi rank vẫn có đủ K/V heads.

Chương cuối (`05-attention-walkthrough`) walkthrough đoạn `Attention.forward` trong `02_large_language_model/model.py` và mapping về plan TP. Tập trung vào các điểm tinh tế: rotary embedding, repeat_kv, scaled_dot_product_attention, và sao chúng đều tương thích với DTensor.

## Liên hệ tới các phần trước

Phần 4 dùng trực tiếp pattern Column-then-Row của Phần 3 nhưng với thêm chiều "head". Mọi quy tắc về placement vẫn áp dụng. Phần 4 cũng giới thiệu khái niệm "shard theo chiều logic, không phải chiều tensor": shard theo head trong sense tự nhiên là một số block cột liên tiếp trên ma trận $W_Q$, đúng tương đương với Column Parallel.

Sau Phần 4, bạn đã có đủ vũ khí cho toàn bộ phần "intra-block" của Transformer. Hai phần còn lại của block là Embedding (đầu vào) và LM head (đầu ra), được Phần 5 lo. Phần 6 đến Phần 9 sẽ mở rộng sang sequence parallelism, 2D parallelism, walkthrough đầy đủ và performance.

Sẵn sàng, ta vào Multi-Head Attention shard theo head.
