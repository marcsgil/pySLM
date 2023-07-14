# from projects.confocal_interference_contrast.polarization_memory_effect.process_memory_effect_data_v1 import show_results
from tkinter import filedialog as fd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# data = np.load(r"D:\scan_pol_memory\results\memory_effect_data\sample_2.5c_thick_0.5mm_step 0.200um_input_polatization_circular2023-05-01_18-55-36.npz")
#
# pol_me_images = data['pol_me_images']
# imgs = []
# for _ in range(5):
#     imgs.append(pol_me_images[20+_])
# np.savez(r"D:\scan_pol_memory\results\memory_effect_data\test_sample_2.5c_id_2023-05-01_18-55-36.npz", imgs = np.array(imgs), concentrations = 2.5)
#
# exit()




# input_directories = [r'D:\scan_pol_memory\results\memory_effect_data']
# reference_path = list(fd.askopenfilenames(initialdir=input_directories[0], title='Images to compare data:'))
# reference_path_2 = list(fd.askopenfilenames(initialdir=input_directories[0], title='Images to compare data:'))
#
# for _ in reference_path_2:
#     reference_path.append(_)
reference_path = ['D:/scan_pol_memory/results/memory_effect_data/new_coefficients_2.5c_thick_0.5mm_step 0.200um_input_polatization_circular2023-05-01_18-55-36.npz']
entries = Path(r"D:\scan_pol_memory\results\memory_effect_data\coefficients_old")
files = []
file_names = []
for entry in entries.iterdir():
    if 'coefficients' in entry.name:
        files.append(entry.absolute())
        file_names.append(entry.name.split('2023-')[1])
idx = []

for reference in reference_path:
    id = reference.split('2023-')[1]
    try:
        idx.append(file_names.index(id))
    except:
        print(f'{id} was not found at the list')

fig, axs = plt.subplots(2, 1, sharex=True, sharey=True)
dif = []
for _ in range(len(reference_path)):
    concentration = reference_path[_].split('c_')[0].split('coefficients_')[1]
    # print(f'loading {reference_path[_]}')
    data = np.load(reference_path[_])
    shift_me_coefficients = np.array(data['shift_me_coefficients'])
    pol_me_coefficients = np.array(data['pol_me_coefficients'])
    positions = data['positions']
    time = data['time']
    statistics = data['statistics']
    pol_name = data['pol_name']

    # print(f'loading {files[idx[_]]}')
    data_old = np.load(files[idx[_]])
    shift_me_coefficients_old = np.array(data_old['shift_me_coefficients'])
    pol_me_coefficients_old = np.array(data_old['pol_me_coefficients'])
    statistics_old = data_old['statistics']
    print(f'___{_}___')
    print(f'File path:  {reference_path[_]}')
    print(f'input polarization: {pol_name}')
    print(f'Statistics concentration {concentration} c:  mean dif  {np.mean(np.abs(pol_me_coefficients) - np.abs(pol_me_coefficients_old))} +- {np.std(np.abs((np.abs(pol_me_coefficients) - np.abs(pol_me_coefficients_old))))}')
    dif.append(np.mean(np.abs(pol_me_coefficients) - np.abs(pol_me_coefficients_old)))
    print(f'Polarization memory effect mean absolute value: new = {statistics[1]} +- {statistics[2]}; old = {statistics_old[1]} +- {statistics_old[2]}')

    axs[0].plot(positions*1e6, np.abs(shift_me_coefficients[0]), '.b', label=f'Horizontal pol., {concentration}c, new')
    axs[0].plot(positions*1e6, np.abs(shift_me_coefficients[1]), '.r', label=f'Vertical pol., {concentration}c, new')
    axs[1].plot(positions*1e6, np.abs(pol_me_coefficients), '.b', label=f'{concentration}c, new, mean ={statistics[1]} $\pm$ {statistics[2]} ')

    axs[0].plot(positions*1e6, np.abs(shift_me_coefficients_old[0]), '*b', label=f'Horizontal pol., {concentration}c, old')
    axs[0].plot(positions*1e6, np.abs(shift_me_coefficients_old[1]), '*r', label=f'Vertical pol., {concentration}c, old')
    axs[1].plot(positions*1e6, np.abs(pol_me_coefficients_old), '*r', label=f'{concentration}c, old, mean ={statistics_old[1]} $\pm$ {statistics_old[2]} ')


axs[0].set_xlabel('x (um)')
axs[0].set_ylabel('Abs. value of coefficient (arb. units)')
axs[0].set_title('Shift memory effect')
axs[1].set_xlabel('x (um)')
axs[1].set_ylabel('Abs. value of coefficient (arb. units)')
axs[1].set_title('Polarization memory effect')
axs[0].legend(loc='lower right')
axs[1].legend(loc='lower right')
plt.show()
dif = np.array(dif)
print(dif)
print(np.amin(dif))

