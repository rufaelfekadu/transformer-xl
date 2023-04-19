"""Even the first partitions have finished to process, the partition before
    the failed partition should be killed as soon as possible.
    """
import torch
from torch import nn

from torchgpipe import GPipe

class ExpectedException(Exception):
    pass

class Pass(nn.Module):
    def forward(self, x):
        print(torch.cuda.current_device())
        return x

model = nn.Sequential(Pass(), Pass(), Pass())
model = GPipe(model, [2, 1], devices=[2, 3], chunks=3)
data = torch.rand(3).to('cuda')
result = model(data)
print('results')
print(result)