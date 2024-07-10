import torch
from accelerate import Accelerator

accelerator = Accelerator()


# device = "cuda"
device = accelerator.device
model.to(device)


model, optimizer, training_dataloader, scheduler = accelerator.prepare(
    model, optimizer, training_dataloader, scheduler
)

for batch in training_dataloader:
    optimizer.zero_grad()
    inputs, targets = batch

    # inputs = inputs.to(device)
    # targets = targets.to(device)
    outputs = model(inputs)
    loss = loss_function(outputs, targets)
    # loss.backward()
    accelerator.backward(loss)
    optimizer.step()
    scheduler.step()
