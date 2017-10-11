# ReLU netowrk with one hidden layer and no biases
# predicts y from x using Euclidean error
# From tutorials on pytorch.org
import numpy as np

# N = batch is
# D_in = input dimension
# D_out = output dimension
# H = hidden dimension
N = 64
D_in = 1000
D_out = 10
H = 100

# Generate random input and output data
x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)

# Randomly initialise the weights
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

learning_rate = 1e-6
for t in range(500):
	# Forward pass - compute predicted y
	h = x.dot(w1)
	h_relu = np.maximum(h, 0)
	y_pred = h_relu.dot(w2)

	# Compute and print loss
	loss = np.square(y_pred -y).sum()
	print(t, loss)

	# Backdrop to compute gradients of w1 and w2 with respect to loss
	grad_y_pred = 2.0 * (y_pred - y)
	grad_w2 = h_relu.T.dot(grad_y_pred)
	grad_h_relu = grad_y_pred.dot(w2.T)
	grad_h = grad_h_relu.copy()
	grad_h[h < 0] = 0
	grad_w1 = x.T.dot(grad_h)

	# Update the weights
	w1 -= learning_rate * grad_w1
	w2 -= learning_rate * grad_w2


