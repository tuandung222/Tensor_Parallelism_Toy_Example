---
title: Tổng quan Phần 2
---

# Tổng quan Phần 2

Sau Phần 1, bạn có công thức toán. Phần 2 cho bạn từ vựng để biến công thức thành code PyTorch thật. Bốn primitives ta cần làm chủ: `ProcessGroup` và NCCL backend, `DeviceMesh`, `DTensor` với các loại `Placement`, và API `parallelize_module`.

## Vì sao có nhiều lớp trừu tượng

PyTorch Distributed có nhiều API ở các mức độ trừu tượng khác nhau, có thể gây bối rối nếu không phân biệt. Dưới đây là thứ tự từ thấp lên cao.

Tầng thấp nhất là `torch.distributed` cổ điển: `init_process_group`, `all_reduce`, `broadcast`. Bạn gọi collective bằng tay. Đây là API của 2018, vẫn được dùng cho code điều khiển mịn.

Tầng giữa là `DeviceMesh`. Một mesh là một grid logic của các rank, có chiều và tên. Ví dụ mesh 2D với chiều `data_parallel` và `tensor_parallel`. Đây là thứ Phần 2 sẽ tập trung.

Tầng giữa-cao là `DTensor`. DTensor là một tensor có metadata "tôi đang được phân bố như thế nào trên mesh". Bạn không phải nghĩ về rank, chỉ nghĩ về placement.

Tầng cao là `parallelize_module` plus các lớp như `ColwiseParallel`, `RowwiseParallel`, `SequenceParallel`. Bạn khai báo plan, PyTorch tự apply DTensor và chèn collective.

Trong chuỗi này ta sẽ dùng chủ yếu tầng giữa-cao và cao. Tầng thấp xuất hiện chỉ khi cần hiểu sâu hoặc debug.

## Cấu trúc Phần 2

Chương 2 nói về `ProcessGroup` và NCCL: làm sao nhiều process trên cùng máy (hoặc nhiều máy) tìm thấy nhau và bắt đầu nói chuyện. Đây là tầng móng.

Chương 3 giới thiệu `DeviceMesh`: cách tổ chức rank thành grid có chiều và tên. Đây là khái niệm quan trọng nhất cho 2D parallelism.

Chương 4 đi vào `DTensor` và các `Placement`: `Replicate`, `Shard`, `Partial`. Đây là cầu nối giữa công thức toán Phần 1 và code.

## Mục tiêu sau Phần 2

Sau Phần 2, bạn nên làm được ba việc.

Một, đọc một đoạn code có `DeviceMesh` và `DTensor`, vẽ ra trên giấy được tensor nằm ở đâu, shard theo chiều nào.

Hai, viết được một script minh họa nhỏ: 4 process, mỗi process tạo một tensor shard, dùng `to_local()` và collective để kiểm tra giá trị.

Ba, đọc dòng `tp_mesh = self.device_mesh["tensor_parallel"]` trong toy code mà không phải tra cứu, hiểu ngay nó đang lấy gì và để làm gì.

Khi bạn ở mức đó, Phần 3 sẽ chỉ là áp dụng.
