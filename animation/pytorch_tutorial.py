import numpy as np
import torch

dtype = torch.float
device = torch.device("cpu")

# N is batch size, D_in is input dimension
# H is hidden dimension, D_out is output dimension
N, D_in, H, D_out = 64, 1000, 100, 10

# input, output data
x = torch.rand(N, D_in)
y = torch.rand(N, D_out)

# initialize weights
w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
w2 = torch.rand(H, D_out, device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-6
for t in range(500):
    # forward
    y_pred = x.mm(w1).clamp(min=0).mm(w2)
    loss = (y_pred - y).pow(2).sum()
    if t % 100 == 99:
        print(t, loss.item())

    loss.backward()

    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

    # h = x.dot(w1)
    # h_relu = np.maximum(h, 0)
    # y_pred = h_relu.dot(w2)
    #
    # loss = np.square(y_pred - y).sum()
    # print(t, loss)
    #
    # # backward
    # grad_y_pred = 2.0 * (y_pred - y)
    # grad_w2 = h_relu.T.dot(grad_y_pred)
    # grad_h_relu = grad_y_pred.dot(w2.T)
    # grad_h = grad_h_relu.copy()
    # grad_h[h<0] = 0
    # grad_w1 = x.T.dot(grad_h)
    #
    # w1 -= learning_rate * grad_w1
    # w2 -= learning_rate * grad_w2