# Experiment Notes

## Estimating how much data to download

To maximise memory usage, our model will ingest "packed" batches, i.e. batches
of maximum length (1024 tokens) where each element in the batch may contain
multiple tokenized samples from our dataset, separated by EOS tokens. This keeps
the GPU at max utilization throughout the training run by minimizing padding.

With this knowledge, we can estimate how much data we are going to need. This is
estimated with

$$
\text{Total entries} = \frac{entries}{GB}\frac{GB}{token} \cdot \frac{token}{second} \cdot
  \text{Training time}
$$

We can estimate $entries/GB$, $GB/token$ and $token/second$, while the training
time is essentially the maximum amount of time allowed by our cluster (5 days).
We train on a single GPU.

To make these estimates, we first need to determine what batch size we'll use.
We'll use whatever batch size is the largest that fits in memory. This is
computed in [tune.py](../claficle/run/tune.py). We find that tensors with
attention masks of mostly ones, the largest batch size we can fit is 2. Of
course we can achieve larger effective batch size with gradient accumulation.

Having fully defined our setup, we can estimate $token/second$ by running
`Trainer.fit` on enough batches (these can just contain random integers) to
include one validation run. This is computed in
[tune.py](../claficle/run/tune.py). We find this to be 2500 tokens per second.

We can estimate $entries/GB$ and $GB/tokens$ by downloading 1 GB of data,
counting the number of entries and tokenizing pit without truncation or padding
(since we will not be throwing away any data with our packed batches) and count
the number of tokens. We measure $GB/tokens$ in
[notebooks/tokens_per_gb.ipynb](../notebooks/tokens_per_gb.ipynb). We find this
to be between 200M and 400M tokens per GB, depending on the language. We'll opt
for the more conservative measure of 200M for our estimates. Similarly we find
$entries/GB$ to be around 400000.

We can now estimate the total amount of data we need to download.

$$
\text{Total entries} = 400000 \cdot \frac{1}{200M} \cdot \frac{2500}{1} \cdot
  5 \cdot 24 \cdot 60 \cdot 60  = 2.1M \text{entries}
$$

To be sure, we will round up and download 2.4M entries of training data for
each language. This is roughly in line to the amount of data used in the
WECHSEL paper, although it is unclear (since the definitions of 'step' may
differ) whether they train for many more steps than we will be able to.
