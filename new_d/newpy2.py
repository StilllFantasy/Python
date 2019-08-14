import torch
import matplotlib.pyplot as plt 

k = torch.linspace(0, 1, 130)
# y = x.pow(2) + 0.2 * torch.rand(x.size())
y = 57
m = 324
f = 1114

z = (k * f) / (y + m*k)

plt.scatter(k.data.numpy(), z.data.numpy())
plt.show()
    
