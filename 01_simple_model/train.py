import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel
from torch.distributed.tensor.parallel import parallelize_module

import lightning as L
from lightning.pytorch.demos.boring_classes import RandomDataset
from lightning.pytorch.strategies import ModelParallelStrategy


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class LitModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = FeedForward(8192, 8192)

    def configure_model(self):
        if self.device_mesh is None:
            return

        # Lightning will set up a `self.device_mesh` for you
        tp_mesh = self.device_mesh["tensor_parallel"]
        # Use PyTorch's distributed tensor APIs to parallelize the model
        plan = {
            "w1": ColwiseParallel(),
            "w2": RowwiseParallel(),
            "w3": ColwiseParallel(),
        }
        parallelize_module(self.model, tp_mesh, plan)

    def training_step(self, batch):
        output = self.model(batch)
        loss = output.sum()
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=3e-3)

    def train_dataloader(self):
        # Trainer configures the sampler automatically for you such that
        # all batches in a tensor-parallel group are identical
        dataset = RandomDataset(8192, 64)
        return torch.utils.data.DataLoader(dataset, batch_size=8, num_workers=2)


strategy = ModelParallelStrategy()
trainer = L.Trainer(
    accelerator="cuda",
    devices=4,
    strategy=strategy,
    max_epochs=1,
    logger=False,
    enable_checkpointing=False,
)

model = LitModel()
trainer.fit(model)

trainer.print(f"Peak memory usage: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")