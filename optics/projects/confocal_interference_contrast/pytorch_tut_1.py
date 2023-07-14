# a tutorial from https://www.projectpro.io/recipes/optimize-function-sgd-pytorch
import torch

# batch size, input layers, hidden layers, output layers
batch, dim_in, dim_h, dim_out = 64, 1000, 100, 10

# define inputs and outputs
input_x = torch.randn(batch, dim_in)
output_y = torch.randn(batch, dim_out)

# define model and loss function
SGD_model = torch.nn.Sequential(
    torch.nn.Linear(dim_in, dim_h),
    torch.nn.ReLU(),
    torch.nn.Linear(dim_h, dim_out),
)

loss_fn = torch.nn.MSELoss(reduction='sum')

rate_learning = 0.1

# initialize optimizer
optim = torch.optim.SGD(SGD_model.parameters(), lr=rate_learning, momentum=0.9)

# forward pass
for values in range(500):
    pred_y = SGD_model(input_x)
    loss = loss_fn(pred_y, output_y)
    if values % 100 == 99:
        print(values, loss.item())

# zero all gradients before backward pass
optim.zero_grad()

# backward pass (we calculate gradients of the loss wrt model params)
loss.backward()

# update the optimizer parameters
optim.step()
