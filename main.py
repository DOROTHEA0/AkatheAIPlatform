import torch

x = torch.randn(3, 12, 512)
y = torch.randn(3, 12, 512)

print(x.shape)
print(y.shape)

print(torch.stack((x, y)).shape)
a, b = torch.chunk(torch.stack((x, y)), 2)
print(a.shape, b.shape)