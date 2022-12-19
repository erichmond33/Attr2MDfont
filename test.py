import torch

# Define two neural networks
def net1(x):
    return x * 2

def net2(x):
    return x * 3

# Define a tensor and set requires_grad=True to track the gradient
x = torch.tensor([1.0], requires_grad=True)

# Use the tensor as input to both neural networks
y1 = net1(x)
y2 = net2(x)

# Compute the loss
loss = y1 + y2

# Backpropagate the gradients
loss.backward()

# Print the gradient of x
print(x.grad)