import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog as fd

input_directories = ['E:/polarization_memory_effect/processed_data/']
paths = fd.askopenfilenames(initialdir=input_directories[0])
file_data = np.load(paths[0])
complex_shift_pols = file_data['complex_shift_pols']
complex_pol_memory = file_data['complex_pol_memory']
positions = file_data['positions']
pol_names = ['Horizontal', 'Vertical', 'Diagonal', 'Anti diagonal', 'Right circular', 'Left circular']


#display

fig = plt.figure()
for _ in range(5):
    plt.plot(positions[:,0], np.abs(complex_pol_memory[_]),  '^', label=pol_names[_+1])

plt.legend()


fig2 = plt.figure()
for _ in range(6):
    plt.plot(positions[:,0], np.abs(complex_shift_pols[_]),  '^', label=pol_names[_])

plt.legend()

plt.show()
