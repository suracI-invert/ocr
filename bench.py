import torch
from time import time
import numpy as np
from tqdm import tqdm

iters = [100, 1_000]

for its in iters:
    print(f'Total iterations: {its}')
    for i in tqdm(range(its), desc= 'CPU'):
        test = [[1] * 512]
        for i in range(127):
            inp = torch.randint(low= 4, high= 227, size= (512,), device= 'cuda')
            inp = inp.cpu()
            test.append(inp)
        test = torch.tensor(test).T
    for i in tqdm(range(its), desc= 'GPU'):
        test = torch.tensor([[1] * 512] + [[0] * 512] * 127, dtype= torch.int64, device= 'cuda')
        for i in range(127):
            inp = torch.randint(low= 4, high= 227, size= (512,), device= 'cuda')
            test[i + 1,:] = inp
        test = test.T
