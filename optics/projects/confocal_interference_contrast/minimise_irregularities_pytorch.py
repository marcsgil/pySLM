import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import numpy.lib.scimath as sm

from optics import log

from optics.utils.ft import Grid
from optics.utils.display import complex2rgb
# from projects.confocal_interference_contrast.interference_image_processing import data_locations, get_file


class LSDIC(torch.nn.Module):
    def __init__(self, initial_estimate):
        super().__init__()
        # Pre-calculate the factors
        self.interference_factors = torch.tensor([1, 1j, -1, -1j], dtype=torch.complex128)  #.cuda()
        for _ in range(initial_estimate.ndim):
            self.interference_factors = self.interference_factors.unsqueeze(-1)

        self.__estimate = torch.nn.Parameter(initial_estimate)

    @property
    def estimate(self) -> torch.nn.Parameter:
        return self.__estimate

    @estimate.setter
    def estimate(self, new_value):
        if not isinstance(new_value, torch.Tensor):
            new_value = torch.from_numpy(new_value)
        if isinstance(new_value, torch.nn.Parameter):
            new_value = new_value.data
        self.__estimate.data = new_value.detach().clone()

    def __h_fun(self, x):
        target = x[1:]
        ref = x[:-1]
        return ref + self.interference_factors * target

    def forward(self, x=None):
        if x is None:
            x = self.estimate
        return torch.abs(self.__h_fun(x)) ** 2


if __name__ == '__main__':
    # Define a ground truth
    sphere_radius = 5e-6
    noise_level = 0.05
    scan_shape = [50, 50]
    grid = Grid(shape=scan_shape, step=200e-9)
    thickness = 2 * sphere_radius * sm.sqrt(1 - sum((_ / sphere_radius) ** 2 for _ in grid)).real
    ground_truth_opd = (1.5151 - 1.3317) / 633e-9 * thickness
    ground_truth = torch.tensor(np.exp(2j * np.pi * ground_truth_opd), dtype=torch.complex128)
    # Test data
    # cp_image = get_file('0026CP', data_locations['bead scans'], file_format='npy')[0]
    # folder = r'C:\\Users\\lvalantinas\\OneDrive - University of Dundee\\Work\\Data\\Scans\\Tomographic scans of beads\\'
    # folder = r'C:\Users\dshamoon001\OneDrive - University of Dundee\LS PhC scans of beads\\'
    # cp_image = get_file('0047CP', folder, file_format='npy')[0]#np.load('gt.npy') #
    # cp_image = torch.tensor(cp_image) / np.amax(np.abs(cp_image))
    # ground_truth = cp_image
    scan_shape = ground_truth.shape

    # The initial estimate
    initial_estimate = torch.ones(scan_shape, dtype=torch.complex64)
    # initial_estimate = ground_truth.detach().clone()

    # Define what the microscope does
    model = LSDIC(initial_estimate)
    # model.cuda()

    # Determine an example measurement for testing
    with torch.no_grad():
        measurement = model(ground_truth)  #.cuda())
        measurement += noise_level * torch.randn(measurement.shape) * torch.max(measurement)

    def mse(_):
        return torch.mean(torch.abs(_)**2)

    def me(_):
        return torch.mean(torch.abs(_))

    def grad_fun(var):
        return [torch.diff(torch.angle(var), 1, dim=_) for _ in range(len(scan_shape))]

    col_weights = 0 / 100
    row_weights = torch.ones([scan_shape[0], 1]) / 100
    row_weights[[0, -1]] = 1000

    def grads_mse(var):
        grads = grad_fun(var)
        weights = [col_weights, row_weights]
        return sum(mse(g * w) for g, w in zip(grads, weights))

    def grads_norm(var):
        grads = grad_fun(var)
        weights = [0.0001/40] + [1/40] * (len(grads) - 1)
        return sum((torch.linalg.norm(g) * w) ** 2 for g, w in zip(grads, weights)) ** 0.5

    def loss_fun():
        prediction = model()
        loss_model = mse(measurement - prediction)
        loss_grad = grads_mse(model.estimate)
        # log.info(f'Loss model: {loss_model:.2f};  loss grad: {loss_grad:.2f}')
        # loss_borders = torch.linalg.norm(model.estimate.imag[0, :]) + torch.linalg.norm(model.estimate.imag[-1, :]) \
        #                + torch.linalg.norm(model.estimate.imag[:, 0]) + torch.linalg.norm(model.estimate.imag[:, -1])
        return loss_model + loss_grad  #+ 10*loss_borders

    # The ground truth estimate
    model.estimate = ground_truth
    ground_truth_loss = loss_fun()
    log.info(f'Ground truth loss = {ground_truth_loss}')
    model.estimate = initial_estimate   # start testing from the ground truth # todo: should not be commented

    # optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Calculation done in each iterations (calculating the loss and the gradient)
    losses = []
    def closure():
        loss = loss_fun()
        optimizer.zero_grad()
        loss.backward()
        losses.append(loss.detach().numpy())
        return loss

    fig, axs = plt.subplots(1, 2)

    def display_error():
        loss = losses[-1]
        with torch.no_grad():
            error_with_ground_truth = torch.linalg.norm(model.estimate - ground_truth) / torch.linalg.norm(ground_truth)
            log.info(f'{_}: loss={loss:0.4f}, error={error_with_ground_truth:0.4f}')

            axs[0].imshow(complex2rgb(ground_truth, 1))
            axs[0].set_title('DIC complex phase data')
            axs[1].imshow(complex2rgb(model.estimate, 1))
            axs[1].set_title('Estimated complex phase')
            plt.pause(0.001)
            plt.show(block=False) #

    # Iterative calculation starts
    start = time.perf_counter()
    for _ in range(10000):
        optimizer.step(closure)
        if _ % 1000 == 0:
            display_error()

    log.info(f'Minimization time: {time.perf_counter() - start:0.3f}s')

    # Showing results
    display_error()
    plt.show() # block=True



