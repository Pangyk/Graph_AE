import torch

a = torch.tensor([1, 2, 3, 4])
b = torch.tensor([0, 0, 0, 0])
c = torch.tensor([5, 5, 5, 5])

s = [a, b, c]
sk = torch.stack(s, dim=1)
print(sk)
