---
title: Giới thiệu chuỗi bài giảng
slug: /intro
---

# Giới thiệu chuỗi bài giảng

Chào bạn. Tôi sẽ là giảng viên đồng hành với bạn trong chuỗi bài giảng này, và mục tiêu duy nhất của chúng ta là hiểu tường tận một kỹ thuật: Tensor Parallelism. Không phải “biết cách gọi API”, không phải “copy pattern từ Lightning docs”, mà hiểu vì sao mỗi dòng code phải như vậy, vì sao mỗi collective operation xảy ra ở đúng vị trí đó, và vì sao Megatron-LM đã chọn pattern này thay vì pattern khác.

## Phong cách bài giảng

Mỗi khái niệm sẽ được giải thích theo ba lớp.

Lớp một là trực giác hình học. Trước khi viết công thức, ta nhìn ma trận và phép nhân như những hình hộp, các phép shard như cách cắt khối thành lát.

Lớp hai là công thức toán. Ta viết rõ shape, derivation, và chứng minh tính tương đương giữa phiên bản phân tán và phiên bản đơn lẻ.

Lớp ba là code PyTorch thật. Mọi pattern đều ánh xạ về `torch.distributed.tensor.parallel`, `DeviceMesh`, `DTensor`, hoặc về toy code Llama-3 đi kèm repo. Đọc xong bạn không chỉ hiểu, mà có thể chạy lại trên 4 GPU.

## Bản đồ chuỗi bài giảng

Chuỗi này được thiết kế tuyến tính. Phần sau dựa trên phần trước. Đừng nhảy qua, nếu nhảy bạn sẽ thấy code phía sau như magic.

Phần 0 trả lời câu hỏi nền tảng nhất: vì sao cần parallelism, và trong các loại parallelism, vì sao chọn Tensor Parallelism. Phần này nhẹ về toán, nặng về trực giác.

Phần 1 là phần toán học. Tôi sẽ derive đầy đủ phép nhân ma trận khi shard cột và shard hàng, chỉ ra lúc nào cần all-reduce, lúc nào không. Đây là phần quan trọng nhất, đừng bỏ qua.

Phần 2 là PyTorch primitives. ProcessGroup, NCCL backend, DeviceMesh, DTensor với các loại Placement. Đây là vocabulary để đọc được code TP thật.

Phần 3 áp dụng vào MLP. Ta học Megatron pattern Column-then-Row, hiểu vì sao SwiGLU có ba linear, và walkthrough từng dòng `01_simple_model/train.py`.

Phần 4 mở rộng pattern Col-then-Row sang Self-Attention, làm rõ vì sao shard theo head chứ không phải head_dim, và xử lý GQA/MQA của Llama-3.

Phần 5 đi vào Embedding (shard vocab) và LM Head với `loss_parallel`, kỹ thuật để tính cross-entropy mà không cần all-gather full logits.

Phần 6 giới thiệu Sequence Parallelism: shard luôn chiều sequence ở zone norm và residual, giảm activation memory thêm $P$ lần mà không tăng giao tiếp.

Phần 7 kết hợp TP với FSDP thành 2D parallelism, công thức training LLM chuẩn ở scale lớn. Bao gồm mixed precision policy và activation checkpoint.

Phần 8 là walkthrough đầy đủ `02_large_language_model/`: đọc kỹ `model.py`, `parallelism.py`, `train.py`, và chạy ngược một full step end-to-end với placement chính xác.

Phần 9 cuối cùng là performance và debugging thực tế: overlap collective với compute, profiling và MFU, và checklist các bug thường gặp với TP/FSDP.

## Toy code đi kèm

Repo gốc có hai ví dụ chạy được:

`01_simple_model/train.py` minh họa Tensor Parallel cho một FeedForward đơn giản với `dim=8192`. Đây là toy nhỏ nhất, lý tưởng để xác nhận thiết lập NCCL đúng và hiểu pattern Col-then-Row.

`02_large_language_model/` là Llama-3 thu nhỏ khoảng 3.5B tham số, dùng 2D parallelism (TP cho layer trong, FSDP cho dữ liệu) trên 4 GPU 24 GB. Đây là production-realistic, dùng cho mọi case study sau này.

Tất cả ví dụ yêu cầu GPU với CUDA và `nccl` backend. Nếu bạn không có GPU, vẫn có thể đọc và hiểu, nhưng để chạy thực tế cần ít nhất 2 GPU.

## Kỳ vọng kiến thức nền

Bạn nên đã quen với PyTorch ở mức trung bình: viết được một `nn.Module`, đào tạo được một model nhỏ, hiểu loss và backward. Bạn không cần biết trước về distributed training. Mọi thứ về `ProcessGroup`, NCCL, DeviceMesh sẽ được giải thích từ đầu trong Phần 2.

Một số kiến thức toán nhẹ sẽ giúp: phép nhân ma trận, ma trận chuyển vị, gradient của một biến đối với một biến khác. Tôi sẽ ôn nhanh khi cần.

## Cách đọc tối ưu

Đọc xong mỗi chương, hãy tự trả lời câu hỏi: nếu mất kết nối giữa các GPU trong 1 mili giây ở vị trí này, kết quả model còn đúng không. Câu hỏi này buộc bạn nghĩ về dependency giữa các process, là cái lõi của distributed training.

Khi đọc code, hãy chỉ ra rank nào đang giữ tensor nào, và shape của tensor sau mỗi collective. Đừng chấp nhận “PyTorch lo việc đó”. Bạn phải vẽ được sơ đồ.

Sẵn sàng chưa? Vào Phần 0.
