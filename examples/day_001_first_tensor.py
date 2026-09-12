import torch

# Day 1.1: TL-001 create our first PyTorch tensor from familiar Python numbers.
numbers = [10, 20, 30]
tensor = torch.tensor(numbers)

# Day 1.2: TL-001 print the tensor so we can verify what PyTorch created.
print(tensor)

# Day 2.1: TL-002 inspect the tensor shape to see how many values it contains.
print(tensor.shape)
