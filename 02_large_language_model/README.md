## Tensor Parallel and 2D Parallel

This example shows how to apply tensor-parallelism to your model (here Llama 3) with the `ModelParallelStrategy`, and how it can be combined with FSDP (2D parallelism).
A Studio with at least 4 GPUs and 24 GB memory each are required to run this example.

Navigate to this example folder and run the training script:

```bash
python train.py
```

You should see an output like this:

```
Initializing distributed: GLOBAL_RANK: 0, MEMBER: 1/4
Initializing distributed: GLOBAL_RANK: 3, MEMBER: 4/4
Initializing distributed: GLOBAL_RANK: 2, MEMBER: 3/4
Initializing distributed: GLOBAL_RANK: 1, MEMBER: 2/4
----------------------------------------------------------------------------------------------------
distributed_backend=nccl
All distributed processes registered. Starting with 4 processes
----------------------------------------------------------------------------------------------------

Number of model parameters: 3.5 B
Starting training ...
...
Iteration 119 time: 0.83 seconds
Iteration 120 time: 0.83 seconds
Iteration 121 time: 0.83 seconds
Iteration 122 time: 0.83 seconds
Iteration 123 time: 0.83 seconds
Iteration 124 time: 0.83 seconds
Iteration 125 time: 0.83 seconds
Iteration 126 time: 0.83 seconds
Iteration 127 time: 0.83 seconds

Training time: 107.59 seconds
Average iteration time: 0.84 seconds
Average iters/sec: 1.19
Peak memory usage: 18.21 GB
```

> \[!NOTE\]
> The `ModelParallelStrategy` is experimental and subject to change. Report issues on [GitHub](https://github.com/Lightning-AI/pytorch-lightning/issues).
