import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_size=1, output_size=1):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        y = self.linear(x)
        return y


torch.manual_seed(0)
x = torch.rand(100, 1)
y = 2 * x + 5 + torch.rand(100, 1)

lr = 0.1
iters = 100
model = Model()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

for i in range(iters):
    y_hat = model(x)
    loss = F.mse_loss(y, y_hat)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if i % 10 == 0:
        print(loss.item())

print(loss.item())

for param in model.parameters():
    print(param)
