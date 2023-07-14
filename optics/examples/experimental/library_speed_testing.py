import numpy as np
import scipy as sci
import torch
import torch.fft

from optics.utils import ft

# import sys
# sys.path.append(r'C:\Users\Laurynas\AppData\Roaming\Python\Python38\Scripts')
# import tensorflow as tf

from timeit import default_timer as timer

# Testing data
rnd = np.random.seed(1)
random_matrix1 = np.random.rand(1000, 1000)
random_matrix2 = np.random.rand(1000, 1000)

# Timing variables
add_timer = np.zeros(2)
mult_timer = np.zeros(2)
fft_timer = np.zeros(2)


print('testing numpy speed...')

test_matrix1 = random_matrix1.copy()
test_matrix2 = random_matrix2.copy()
add_timer[0] = timer()
for _ in range(1000):
    if _ % 2 == 0:
        test_matrix1 += test_matrix2
    else:
        test_matrix2 += test_matrix1
add_timer[1] = timer()
print('addition: Done...')

test_matrix1 = random_matrix1.copy()
test_matrix2 = random_matrix2.copy()
mult_timer[0] = timer()
for _ in range(1000):
    if _ % 2 == 0:
        test_matrix1 *= test_matrix2
    else:
        test_matrix2 *= test_matrix1
mult_timer[1] = timer()
print('multiplication: Done...')

test_matrix1 = random_matrix1.copy()
fft_timer[0] = timer()
for _ in range(1000):
    test_matrix1 = np.fft.fft2(test_matrix1)
fft_timer[1] = timer()
print('fft: Done...')

print(f'numpy speed for addition {add_timer[1] - add_timer[0]}s, multiplication {mult_timer[1] - mult_timer[0]}s, fft {fft_timer[1] - fft_timer[0]}s')

#####################################
print('testing fft with pyfftw')
test_matrix1 = random_matrix1.copy()
fft_timer[0] = timer()
for _ in range(1000):
    test_matrix1 = ft.fft2(test_matrix1)
fft_timer[1] = timer()
print('fft: Done...')

print(f'pyfftw speed fft {fft_timer[1] - fft_timer[0]}s')


#####################################
print('testing pytorch speed...')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'used device: {device}...')

random_matrix1 = torch.tensor(random_matrix1).to(device)
random_matrix2 = torch.tensor(random_matrix2).to(device)

test_matrix1 = random_matrix1
test_matrix2 = random_matrix2
add_timer[0] = timer()
for _ in range(1000):
    if _ % 2 == 0:
        test_matrix1 += test_matrix2
    else:
        test_matrix2 += test_matrix1
add_timer[1] = timer()
print('addition: Done...')

test_matrix1 = random_matrix1
test_matrix2 = random_matrix2
mult_timer[0] = timer()
for _ in range(1000):
    if _ % 2 == 0:
        test_matrix1 *= test_matrix2
    else:
        test_matrix2 *= test_matrix1
mult_timer[1] = timer()
print('multiplication: Done...')

test_matrix1 = random_matrix1
fft_timer[0] = timer()
for _ in range(1000):
    test_matrix1 = torch.fft.fftn(test_matrix1)
fft_timer[1] = timer()
print('fft: Done...')
print(f'torch tensor speed for addition {add_timer[1] - add_timer[0]}s, multiplication {mult_timer[1] - mult_timer[0]}s, fft {fft_timer[1] - fft_timer[0]}s')







