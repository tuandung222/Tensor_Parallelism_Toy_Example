---
title: Tổng quan Phần 3
---

# Phần 3: Tensor Parallelism cho MLP

Phần này là điểm hội tụ. Toán Phần 1 và primitives Phần 2 sẽ được lắp vào nhau để giải một bài toán cụ thể: làm sao chạy được khối Feed-Forward Network (FFN, hay còn gọi là MLP) của một Transformer khi ma trận trọng số quá lớn để vừa trong một GPU.

Tôi chọn MLP làm bước đầu tiên không phải ngẫu nhiên. Trong một Transformer hiện đại như Llama hay GPT, FFN chiếm khoảng hai phần ba tổng số tham số. Nếu bạn shard được FFN, bạn đã giải quyết phần lớn áp lực bộ nhớ. Hơn nữa, MLP đơn giản hơn Self-Attention vì không có ràng buộc giữa các head, không có softmax, không có rotary embedding. Toàn bộ phép tính chỉ là hai (hoặc ba) phép nhân ma trận và một activation phi tuyến. Đây là sân tập lý tưởng để áp dụng pattern Column-then-Row mà ta đã derive.

## Mục tiêu sau Phần 3

Khi kết thúc bốn chương của Phần 3, bạn cần đạt được những điều sau.

Thứ nhất, bạn phải vẽ lại được sơ đồ Megatron MLP cho cả forward và backward, chỉ rõ rank nào giữ shard nào, và collective nào nằm ở đâu. Đây là kiến thức nền, mọi pattern TP phức tạp hơn (Self-Attention, Embedding, Sequence Parallel) đều là biến thể của nó.

Thứ hai, bạn phải giải thích được vì sao kiến trúc SwiGLU sinh ra ba ma trận $W_1, W_2, W_3$ thay vì hai như MLP ReLU truyền thống, và vì sao cả $W_1$ và $W_3$ đều shard cột còn $W_2$ shard hàng.

Thứ ba, bạn phải đọc được từng dòng `01_simple_model/train.py` và biết chính xác câu lệnh `parallelize_module(self.model, tp_mesh, plan)` đang làm gì bên dưới: nó wrap mỗi linear thành DTensor nào, ai gọi collective nào trong forward, ai gọi trong backward.

Nếu bạn đạt được ba điều trên, bạn có thể tự viết một plan TP cho bất kỳ kiến trúc Feed-Forward biến thể nào, kể cả GeGLU, ReGLU, hoặc Mixture-of-Experts gating đơn giản.

## Cấu trúc bốn chương

Chương đầu (`02-megatron-mlp-pattern`) trình bày pattern kinh điển Column-then-Row mà Megatron-LM đã đặt nền móng. Ta sẽ derive lại từ Phần 1 nhưng lần này dán nó vào ngữ cảnh MLP hai tầng $Y = \sigma(X W_1) W_2$. Câu hỏi trung tâm: một forward MLP qua TP cần bao nhiêu collective, ở đâu, và vì sao đó là số tối thiểu.

Chương hai (`03-swiglu-ba-linear`) mở rộng sang biến thể SwiGLU mà Llama, Mistral, và phần lớn LLM hiện đại đang dùng. Ba linear $W_1, W_2, W_3$ tưởng làm pattern phức tạp hơn, nhưng thực ra nó tuân theo cùng một nguyên tắc. Ta sẽ chỉ ra nguyên tắc đó dưới dạng định lý nhỏ về tính song song giữa các nhánh, và rút ra quy tắc đặt placement cho phép element-wise multiply.

Chương ba (`04-toy-simple-model-walkthrough`) là cầu nối lý thuyết và code. Ta sẽ mở `01_simple_model/train.py`, đọc từng dòng, ánh xạ về sơ đồ ở chương trước, và làm rõ vai trò của `ColwiseParallel()`, `RowwiseParallel()`, `parallelize_module()`, cùng cách Lightning quản lý `device_mesh`.

Chương cuối (chưa có ở phiên bản này, sẽ là Phần 4 trở đi) sẽ bước sang Self-Attention, nhưng trước đó bạn cần Phần 3 đứng vững.

## Mối liên hệ tới các phần khác

Phần 3 sử dụng trực tiếp ba kết quả sau từ Phần 1: (i) Column Parallel với input replicate và output shard, (ii) Row Parallel với input shard và output replicate sau all-reduce, (iii) bất biến rằng all-reduce trong forward kéo theo identity (no-op) trong backward, và ngược lại.

Phần 3 sử dụng trực tiếp ba primitives sau từ Phần 2: (i) `DeviceMesh` với chiều `"tensor_parallel"` để gom các rank cùng nhóm TP, (ii) `parallelize_module` để chuyển một `nn.Linear` thường thành `nn.Linear` với DTensor parameter, (iii) ngữ nghĩa `Placement` (`Replicate`, `Shard`, `Partial`) để hiểu output của mỗi tầng trông như thế nào về mặt phân bố dữ liệu.

Nếu bạn cảm thấy chưa chắc tay với Phần 1 hoặc Phần 2, hãy quay lại đọc kỹ chương Column/Row derivation trong `01-toan-hoc` và chương DTensor trong `02-pytorch-distributed` trước khi đi tiếp.

## Một lưu ý quan trọng về batch và sequence

Trong Phần 3, để giữ trọng tâm, ta sẽ giả thiết input có shape $(B, K)$, tức coi cả batch và sequence như một chiều duy nhất $B$. Trong thực tế, input MLP của Transformer có shape $(B, S, K)$ với $S$ là sequence length. Phép nhân $X W$ hoạt động giống hệt vì nó được thực hiện trên hai chiều cuối, hai chiều đầu chỉ là chiều "batch dimension".

Sự khác biệt $(B, K)$ và $(B, S, K)$ chỉ trở nên quan trọng khi ta sang Phần 6 (Sequence Parallelism), nơi ta shard luôn chiều $S$. Ở Phần 3, hãy tạm gộp $B$ và $S$ thành một chiều, sơ đồ và derivation sẽ sạch hơn.

Sẵn sàng, ta bước vào Megatron MLP pattern.
