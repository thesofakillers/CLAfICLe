from transformers import AutoModelForCausalLM
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained("gpt2-large")
model.to(device)


for bs in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    print(f"Batch size: {bs}")
    batch = torch.randint(0, 50000, (bs, 1024))
    attention_mask = torch.ones((bs, 1024))
    batch, attention_mask = batch.to(device), attention_mask.to(device)

    optimizer.zero_grad()
    print("Forward pass")
    output = model(input_ids=batch, attention_mask=attention_mask, labels=batch)
    loss = output.loss
    print("Backward pass")
    loss.backward
    print("Optimizer step")
    optimizer.step()

    # clear things from memory
    del batch, attention_mask, optimizer, output, loss
