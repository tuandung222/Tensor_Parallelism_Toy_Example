import time
import lightning as L
import torch
import torch.nn.functional as F
from data import RandomTokenDataset
from lightning.pytorch.strategies import ModelParallelStrategy
from model import ModelArgs, Transformer
from parallelism import parallelize
from torch.distributed.tensor.parallel import loss_parallel
from torch.utils.data import DataLoader
from argparse import ArgumentParser


class Llama3(L.LightningModule):
    def __init__(self, n_layers: int = 16):
        super().__init__()
        self.model_args = ModelArgs(vocab_size=32000, n_layers=n_layers)
        self.model = Transformer(self.model_args)

    def configure_model(self):
        # User-defined function that applies the desired parallelizations specific to the model
        # (TP, FSDP2, activation checkpointing, ...)
        parallelize(self.model, device_mesh=self.device_mesh)

    def on_train_start(self) -> None:
        self.model.init_weights()

    def training_step(self, batch):
        inputs = batch[:, :-1]
        labels = batch[:, 1:]
        output = self.model(inputs)
        # Optional: Parallelize loss computation across class dimension (see parallelism.py)
        with loss_parallel():
            return F.cross_entropy(output.reshape(-1, output.size(-1)), labels.reshape(-1))

    def backward(self, *args, **kwargs):
        with loss_parallel():
            super().backward(*args, **kwargs)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=3e-3, foreach=True)

    def train_dataloader(self):
        dataset = RandomTokenDataset(
            size=512,
            vocab_size=self.model_args.vocab_size, 
            seq_length=128
        )
        # Trainer configures the sampler automatically for you such that
        # all batches in a tensor-parallel group are identical
        return DataLoader(dataset, batch_size=8, num_workers=4)


def train(data_parallel_size, tensor_parallel_size, n_layers):
    
    strategy = ModelParallelStrategy(
        # Define the size of the 2D parallelism
        # Set to "auto" to apply TP intra-node and DP inter-node
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

    # Initialize the model
    with trainer.init_module(empty_init=True):
        model = Llama3(n_layers=n_layers)

    trainer.print(f"Number of model parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.1f} B")
    trainer.print("Starting training ...")
    
    t0 = time.perf_counter()

    trainer.fit(model)
    
    t1 = time.perf_counter()
    trainer.print(f"Training time: {t1 - t0:.02f} seconds")
    trainer.print(f"Average iteration time: {(t1 - t0) / len(trainer.train_dataloader):.02f} seconds")
    trainer.print(f"Average iters/sec: {len(trainer.train_dataloader) / (t1 - t0):.02f}")
    trainer.print(f"Peak memory usage: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")



if __name__ == "__main__":
    assert torch.cuda.device_count() >= 4, "This example requires at least 4 GPUs with 24 GB of memory each."
    
    parser = ArgumentParser()
    parser.add_argument("--data-parallel-size", type=int, required=False)
    parser.add_argument("--tensor-parallel-size", type=int, required=False)
    parser.add_argument("--n-layers", type=int, default=16)
    args = parser.parse_args()
    args.data_parallel_size = args.data_parallel_size or "auto"
    args.tensor_parallel_size = args.tensor_parallel_size or "auto"
    
    torch.set_float32_matmul_precision("high")
    train(args.data_parallel_size, args.tensor_parallel_size, args.n_layers)