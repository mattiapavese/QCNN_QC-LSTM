############Testing
import torch
from QCNN import QConv2d
from pennylane import numpy as np

in_channels=5
x = torch.tensor( np.random.rand(1, in_channels, 8,8), requires_grad=True ).float()

conv2d =QConv2d(in_channels=in_channels, kernel_size=(3,3))
out=conv2d(x)

print('out:', out.shape)
print('end')