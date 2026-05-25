---
title: train.py với Lightning
---

# `train.py`, training loop với Lightning

File `train.py` đóng vai trò "main": ráp model, plan parallelism, optimizer, data, và trainer thành một flow training. Lightning được dùng làm framework để giảm boilerplate. Chương này phân tích từng phần.

## Lớp `Llama3` (LightningModule)

```python
class Llama3(L.LightningModule):
    def __init__(self, n_layers=16):
        super().__init__()
        self.model_args = ModelArgs(vocab_size=32000, n_layers=n_layers)
        self.model = Transformer(self.model_args)
```

`__init__` khởi tạo model thuần Llama-3, không có parallelism. Tham số `n_layers` mặc định 16 (thay vì 32 của Llama-3 8B) để toy dễ chạy hơn. Vocab size 32000 cho gọn (Llama-3 thật là 128K).

Trong `__init__`, model là CPU tensor thường (hoặc meta device tùy strategy). Chưa shard.

## `configure_model`

```python
def configure_model(self):
    parallelize(self.model, device_mesh=self.device_mesh)
```

Hook do Lightning gọi sau khi mesh đã được tạo. Đây là điểm áp dụng parallelism. `self.device_mesh` là DeviceMesh 2D do Lightning's `ModelParallelStrategy` tạo, có hai chiều `data_parallel` và `tensor_parallel`.

Gọi `parallelize` thay model bằng phiên bản đã shard. Sau dòng này, mọi forward gọi `self.model(...)` sẽ chạy với TP + FSDP.

## `on_train_start`

```python
def on_train_start(self):
    self.model.init_weights()
```

Khởi tạo weight sau khi parallelism đã áp. Lưu ý thứ tự: parallel trước, init sau.

Lý do: nếu init trước, weight được tạo trên CPU rồi mới shard. Tốn bộ nhớ peak. Init sau khi đã shard, mỗi rank chỉ init shard của mình, không cần gom.

Implementation của `init_weights` trong `model.py` xử lý DTensor parameter trong suốt: gọi `nn.init.trunc_normal_(linear.weight, ...)` trên DTensor, PyTorch tự áp init trên local shard. Magic.

## `training_step`

```python
def training_step(self, batch):
    inputs = batch[:, :-1]
    labels = batch[:, 1:]
    output = self.model(inputs)
    with loss_parallel():
        return F.cross_entropy(output.reshape(-1, output.size(-1)), labels.reshape(-1))
```

Phép shift để tạo (input, label) cho language modeling: input = token $[t_1, ..., t_{S-1}]$, label = token $[t_2, ..., t_S]$. Mô hình dự đoán token kế.

Forward `self.model(inputs)` cho ra logits shape $(B, S-1, V)$, DTensor Shard(-1) (vocab shard).

`loss_parallel` context: `F.cross_entropy` tự nhận diện DTensor shard, tính LSE parallel, AllReduce thông minh. Output là scalar loss (DTensor replicate).

`output.reshape(-1, output.size(-1))`: reshape thành 2D cho cross_entropy. Lưu ý đây là phép trên DTensor, PyTorch xử lý reshape của Shard(-1) tensor đúng cách.

## `backward` override

```python
def backward(self, *args, **kwargs):
    with loss_parallel():
        super().backward(*args, **kwargs)
```

Override để bọc backward trong `loss_parallel`. Cần thiết vì backward của cross_entropy với logits Shard cần dùng implementation parallel. Không có dòng này, backward sẽ implicit all-gather logits, lãng phí.

Lightning gọi `self.backward(loss)` thay vì `loss.backward()` trực tiếp, cho phép user override như vầy.

## `configure_optimizers`

```python
def configure_optimizers(self):
    return torch.optim.AdamW(self.model.parameters(), lr=3e-3, foreach=True)
```

AdamW chuẩn, learning rate cao bất thường $3 \cdot 10^{-3}$ (cho toy training, không phải khuyến nghị production).

`foreach=True`: dùng "foreach" implementation của AdamW, nhanh hơn vì tự vectorize. Quan trọng cho training nhanh.

Với DTensor parameter (đã shard), AdamW xử lý optimizer state cũng dưới dạng DTensor shard. Optimizer step chạy local trên mỗi rank, mỗi rank update shard của mình.

## `train_dataloader`

```python
def train_dataloader(self):
    dataset = RandomTokenDataset(
        size=512,
        vocab_size=self.model_args.vocab_size,
        seq_length=128,
    )
    return DataLoader(dataset, batch_size=8, num_workers=4)
```

`RandomTokenDataset` là `data.py`: dataset toy tạo 512 sample, mỗi sample 129 token (128 + 1 cho shift). Seed cố định 42, mọi rank tạo cùng dataset.

Lightning's `ModelParallelStrategy` tự áp `DistributedSampler` với `num_replicas=dp_size, rank=dp_rank`. Vậy:

- Trong cùng một TP group, mọi rank nhận cùng batch (input replicate).
- Khác TP group (khác DP rank), nhận batch khác (data parallel).

Setup đúng cho 2D parallelism.

## Hàm `train`

```python
def train(data_parallel_size, tensor_parallel_size, n_layers):
    strategy = ModelParallelStrategy(
        data_parallel_size=data_parallel_size,
        tensor_parallel_size=tensor_parallel_size,
    )

    trainer = L.Trainer(
        accelerator="cuda",
        devices=4,
        strategy=strategy,
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
    )

    with trainer.init_module(empty_init=True):
        model = Llama3(n_layers=n_layers)

    trainer.fit(model)
```

`ModelParallelStrategy(data_parallel_size, tensor_parallel_size)` cấu hình Lightning để tạo mesh 2D. Với `data_parallel_size="auto", tensor_parallel_size="auto"`, Lightning áp heuristic: TP intra-node, DP inter-node.

`trainer.init_module(empty_init=True)`: khởi tạo model trên **meta device** (không cấp phát memory thực). Sau khi `configure_model` shard và gọi `init_weights`, mới cấp phát memory cho shard local. Đây là cách init model lớn mà không OOM ở giai đoạn init.

`trainer.fit(model)` chạy training. Lightning lo gọi `configure_model` (áp parallel), `on_train_start` (init), rồi vòng lặp training với `training_step` và `backward`.

## Chạy thực tế

```bash
torchrun --nproc_per_node=4 train.py
```

Hoặc dùng default `data_parallel_size="auto", tensor_parallel_size="auto"`:

```bash
torchrun --nproc_per_node=4 train.py --tensor-parallel-size 4
```

(TP=4, DP=1, không có data parallelism, chỉ TP thuần.)

Hoặc 2D:

```bash
torchrun --nproc_per_node=4 train.py --data-parallel-size 2 --tensor-parallel-size 2
```

(TP=2, DP=2.)

## Một số lựa chọn thiết kế đáng ghi nhớ

1. **Plan parallel tách rời model**: `parallelism.py` không sửa `model.py`. Một số kiến trúc phải sửa code model để TP-aware, không lý tưởng. PyTorch DTensor cho phép wrap mà không sửa.

2. **Lightning hide complexity**: Lightning lo sampler, init meta, mesh setup, distributed launch. Cùng code có thể chạy 1 GPU hoặc 32 GPU.

3. **`empty_init=True`**: trick quan trọng để init model lớn. Bạn nên hiểu nguyên tắc, dù dùng framework khác.

4. **`loss_parallel` cả forward và backward**: phải nhớ cả hai context. Quên backward sẽ gradient sai.

Chương cuối Phần 8 sẽ chạy ngược một step end-to-end với placement chính xác.
