# Experiment Notes

## Estimating how much data to download

To maximise memory usage, our model will ingest "packed" batches, i.e. batches
of maximum length (1024 tokens) where each element in the batch may contain
multiple tokenized samples from our dataset, separated by EOS tokens. This keeps
the GPU at max utilization throughout the training run by minimizing padding.

With this knowledge, we can estimate how much data we are going to need. This is
estimated with

$$
\text{Total tokens} = \frac{tokens}{second} \cdot
  \text{Training time}
$$

$tokens/second$, while the training time is essentially the maximum amount of
time allowed by our cluster (5 days). We train on a single GPU, using mixed
precision.

To make these estimates, we first need to determine what batch size we'll use.
We'll use whatever batch size is the largest that fits in memory. This is
computed in [tune.py](../claficle/run/tune.py). We find that tensors with
attention masks of mostly ones, the largest batch size we can fit is 2. Of
course we can achieve larger effective batch size with gradient accumulation.

Having fully defined our setup, we can estimate $token/second$ by running
`Trainer.fit` on enough batches (these can just contain random integers) to
include one validation run. This is computed in
[tune.py](../claficle/run/tune.py). We find this to be roughly 7000 tokens per
second.

We can now estimate the total amount of tokens we need to download.

$$
\text{Total tokens} =  \frac{7000}{1} \cdot 5 \cdot 24 \cdot 60 \cdot 60
                    = 3B \text{tokens}.
$$
