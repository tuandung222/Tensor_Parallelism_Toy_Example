---
title: Tổng quan Phần 0
---

# Tổng quan Phần 0

Trước khi nói về Tensor Parallelism, ta phải trả lời câu hỏi nền: vì sao một mô hình lớn lại đặt ra vấn đề kỹ thuật. Nếu mô hình bé, một GPU đủ và mọi câu chuyện này không tồn tại. Khi mô hình lớn, mỗi loại parallelism là một câu trả lời cho một nút thắt khác nhau. Tensor Parallelism là câu trả lời cho một nút thắt rất cụ thể, và Phần 0 sẽ chỉ ra đúng nút thắt đó.

## Câu hỏi dẫn dắt

Hãy giữ ba câu hỏi sau trong đầu suốt Phần 0.

Câu hỏi thứ nhất. Khi nói “mô hình quá lớn”, ta đang nói lớn về cái gì cụ thể: tham số, gradient, optimizer state, activation, hay KV cache.

Câu hỏi thứ hai. Có bao nhiêu cách phân chia mô hình giữa nhiều GPU, và mỗi cách chuyển bài toán bộ nhớ thành bài toán băng thông như thế nào.

Câu hỏi thứ ba. Tensor Parallelism giải quyết bộ phận nào trong ba câu trên, và nó tốt hơn các phương án khác ở tình huống nào.

Phần 0 không có công thức nặng. Nhưng cấu trúc tinh thần ở đây sẽ quyết định bạn có cảm thấy phần sau “tự nhiên” hay không.

## Cấu trúc Phần 0

Bốn chương ngắn, ý tưởng liên kết theo trình tự.

Chương 2 phân tích bộ nhớ một GPU. Ta sẽ tính cụ thể vì sao một mô hình 7B đã bị nghẹt trên một GPU 24 GB, và đâu là “con voi” thực sự trong bộ nhớ.

Chương 3 phân biệt bốn loại parallelism phổ biến: Data Parallel, Tensor Parallel, Pipeline Parallel, và FSDP. Cùng với chúng là Sequence Parallel như một biến thể bổ trợ.

Chương 4 trả lời câu hỏi quan trọng: khi nào TP là lựa chọn đúng và khi nào nó tạo nhiều overhead hơn lợi ích.

## Mục tiêu sau Phần 0

Sau Phần 0, bạn không cần viết được code TP. Bạn cần làm được ba việc trí tuệ.

Một, vẽ được trên một tờ giấy bộ nhớ của một mô hình bất kỳ trên một GPU, tách thành các thành phần param, grad, opt state, activation.

Hai, đứng trước một lựa chọn “DP hay TP hay PP hay FSDP”, lập luận được từng phương án giải quyết bộ phận nào trong các thành phần đó.

Ba, lấy ví dụ Llama-3 3.5B trong toy repo của chúng ta, ước lượng được tại sao Lightning chọn TP=2 và FSDP=2 trên 4 GPU 24 GB, và không phải tổ hợp khác.

Khi bạn làm được ba việc đó, bạn đã sẵn sàng cho Phần 1, nơi ta sẽ thực sự đi sâu vào toán.
