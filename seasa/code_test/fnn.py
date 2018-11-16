import torch
import torch.nn as nn


model = nn.Sequential(nn.Linear(5,2))


a = torch.rand(3,5)

out = model(a)

print(out)
print(out[:,1])