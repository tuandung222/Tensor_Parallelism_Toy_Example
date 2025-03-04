import torch
from torch.utils.data import Dataset


class RandomTokenDataset(Dataset):
    def __init__(self, vocab_size: int, seq_length: int, size: int = 128):
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.size = size
        self.tokens = torch.randint(
            self.vocab_size,
            size=(self.size, self.seq_length + 1),
            # Set a seed to make this toy dataset the same on each rank
            # Fabric will add a `DistributedSampler` to shard the data correctly
            generator=torch.Generator().manual_seed(42),
        )

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, item: int):
        return self.tokens[item]
