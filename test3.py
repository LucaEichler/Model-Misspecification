import torch
import time

from classical_models import NonLinear

dx=1
dy=1
dh=2
gt = NonLinear(dx, dy, dh)
gt2 = NonLinear(dx, dy, dh)

X = torch.randn(4).unsqueeze(1).unsqueeze(0)

a = gt(X)
b = gt2(X)
c = gt(torch.cat((X,X),dim=0), torch.cat((gt.get_W().unsqueeze(0), gt2.get_W().unsqueeze(0)), dim=0))

print(a)
print(c[0])
print(b)
print(c[1])


