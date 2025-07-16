import torch
import time
while 1 != 0:
    i = torch.randn(100)
    if i[0] > 1.:
        time.sleep(1)
