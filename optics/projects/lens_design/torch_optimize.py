import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
# from lbfgs import LBFGS


torch.manual_seed(0)


def error_function(intensity_distribution):
    # A simple focusing cost function
    peak_intensity = torch.amax(intensity_distribution)
    total_intensity = torch.sum(intensity_distribution)
    return 1.0 - peak_intensity / total_intensity
    # # Rosenbrock function: f(x, y) = (a - x)^2 + b(y - x^2)^2
    # return (1 - p[0])**2 + 100 * (p[1] - p[0]**2)**2


class LensModel(nn.Module):
    def __init__(self, nb_variables):
        super().__init__()
        self.lens_scattering = torch.nn.Parameter(torch.randn(nb_variables) + torch.randn(nb_variables) * 1j)

    def forward(self):
        return torch.abs(torch.fft.fftn(self.lens_scattering)) ** 2  # Simulate intensity after a lens

    def normalize(self):
        with torch.no_grad():
            self.lens_scattering /= torch.linalg.norm(self.lens_scattering)


if __name__ == '__main__':
    nb_variables = 20
    model = LensModel(nb_variables)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.5)
    # optimizer = LBFGS(model.parameters(), history_size=10, max_iter=40)

    def display_function(model, error):
        print(f'p = {model.lens_scattering.detach().numpy()} => {error.detach().numpy():.3f}')  # detach from graph before converting to numpy!

    errors = []
    def calc_error_and_graph():
        optimizer.zero_grad()  # Reset gradient calculations
        error = error_function(model())  # Calculate the cost (and build the calculation graph used for the gradient calculation)
        # display_function(model, error)
        error.backward()  # Work backwards through the graph to determine the error gradient directly as a function of the model's parameters
        # Keep intermediate results for reporting
        errors.append(error)
        return error

    for _ in range(1000):
        optimizer.step(calc_error_and_graph)


    #
    # Calculate error for final parameters
    #
    with torch.no_grad():
        model.normalize()
        intensity_distribution = model()
        error = error_function(intensity_distribution)
    errors.append(error)
    errors = np.asarray([_.detach().numpy() for _ in errors])

    #
    # Display solution
    #
    print(f'Final result: {intensity_distribution.detach().numpy()}')
    display_function(model, error)

    plt.semilogy(errors)
    plt.show(block=True)

