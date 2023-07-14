import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog as fd
from optics.utils.ft import Grid
from optics.utils.ft.czt import interp
from optics.utils.display import grid2extent, complex2rgb



#funcs


def creates_display(img1, img2, img3, factor):
    fig, axs = plt.subplots(1, 3)#, sharex='all', sharey='all')
    axs[0].imshow(complex2rgb(img1, 1))
    axs[1].imshow(complex2rgb(img2, 1))
    axs[2].imshow(complex2rgb(img3, 1))
    axs[0].set_title(f'Scan low resolution {img1.shape}')
    axs[1].set_title(f'Interpolated {img2.shape} [factor {factor:0.3}] ')
    axs[2].set_title(f'Scan high resolution {img3.shape}')
    [ax.set_axis_off() for ax in axs.ravel()]
    plt.get_current_fig_manager().window.showMaximized()
    plt.pause(0.01)
    output_file = r'C:\Users\lab\OneDrive - University of Dundee\share_files\figures' + f'/complex_amp_factor_{factor:0.3}.png'
    fig.savefig(output_file)
    plt.show()


def creates_intensities_display(lr_intensities, interp_intensities, hr_intensities, factor):
    intensities = [lr_intensities, interp_intensities, hr_intensities]
    max_intensity = np.zeros(3)
    for _ in range(3):
        max_intensity[_] = np.amax(intensities[_])

    fig, axs = plt.subplots(3, 4)
    pol_names = ['Diagonal', 'Left', 'Anti diagonal', 'Right']
    data_names = ['Original data', 'Interpolated data', 'HR reference data']
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])

    for line in range(3):
        for c in range(4):
            img = axs[line, c].imshow(intensities[line][c] / np.amax(max_intensity), vmin=0, vmax=1)
            axs[line, c].set_title(f'{pol_names[c]} - {data_names[line]} {intensities[line][c].shape}')
            axs[line, c].set_axis_off()
            fig.colorbar(img, cax=cbar_ax)

    plt.get_current_fig_manager().window.showMaximized()
    plt.pause(0.01)
    output_file = r'C:\Users\lab\OneDrive - University of Dundee\share_files\figures' + f'/intesities_factor_{factor:0.3}.png'
    fig.savefig(output_file)
    plt.show()

def save_intensities_from_data(data_path):
    data = np.load(data_path)
    complex_amplitude = data['complex_amplitude']
    step_size = data['step_size']
    scan_size = data['scan_size']
    camera_data = data['camera_data']
    centre = np.array(camera_data[0, 0].shape) // 2
    intensity_d = np.mean(camera_data[..., :centre[0], :centre[1]], axis=(-1, -2))
    intensity_l = np.mean(camera_data[..., centre[0]:, :centre[1]], axis=(-1, -2))
    intensity_a = np.mean(camera_data[..., :centre[0], centre[1]:], axis=(-1, -2))
    intensity_r = np.mean(camera_data[..., centre[0]:, centre[1]:], axis=(-1, -2))
    intensities = np.array([intensity_d, intensity_l, intensity_a, intensity_r])
    extra_info = 'intensities_'
    np.savez(data_path.split('DPC')[0] + extra_info + data_path.split('DPC_scan_')[1],
         complex_amplitude = complex_amplitude, intensities = intensities, scan_size = scan_size, step_size = step_size)

def interp_amps(lr_intensities, factor):
    interp_intensities = []
    for _ in range(lr_intensities.shape[0]):
        interp_intensities.append(np.abs(interp(lr_intensities[_], factor)))

    interp_intensities = np.asarray(interp_intensities)
    polarization_phasors = np.exp(2j * np.pi / 4 * np.arange(4))
    average_intensity = np.mean(interp_intensities, axis=0)
    interp_complex_amplitude = sum(i * _ for i, _ in zip(interp_intensities, polarization_phasors)) / polarization_phasors.size #/ average_intensity
    interp_complex_amplitude[np.isnan(interp_complex_amplitude)] = 0.0
    return interp_complex_amplitude, interp_intensities


data_path = r'c:\Users\lab\OneDrive - University of Dundee\Documents\lab\code\python\optics\projects\confocal_interference_contrast\results\DPC_scan_64x200_2023-02-21_17-21-28.npz'
save_intensities_from_data(data_path)
exit()




data_path = r'C:\Users\lab\OneDrive - University of Dundee\share_files\intensities_50x50_2023-02-16_00-21-31.npz'
data = np.load(data_path)
lr_complex_amplitude = data['complex_amplitude']
step_size = data['step_size']
scan_size = data['scan_size']
lr_intensities = data['intensities']   # d, l , a , r
# scan_hr_data_path = r"C:\Users\braia\IdeaProjects\lab\results\scans\interpolation\intensities_complex_amplitude80x80_2023-01-31_18-01-04.npz"
# data_scan_hr = np.load(scan_hr_data_path)
# hr_complex_amplitude = data_scan_hr['complex_amplitude']
# hr_intensities = data_scan_hr['intensities']




factor = 2

interp_complex_amplitude, interp_intensities = interp_amps(lr_intensities, factor)
interp_complex_amplitude_2, interp_intensities_2 = interp_amps(lr_intensities, factor*2)

creates_intensities_display(lr_intensities, interp_intensities, interp_intensities_2, float(factor))
creates_display(lr_complex_amplitude, interp_complex_amplitude, interp_complex_amplitude_2, float(factor))

# factor = np.linspace(1, 2, 10)

# for _ in factor:
#     interp_complex_amplitude = interp_amps(lr_intensities, _)
#     creates_display(lr_complex_amplitude, interp_complex_amplitude, hr_complex_amplitude, _)


exit()

##### save intensities

# centre = np.array(camera_data[0, 0].shape) // 2
# intensity_d = np.mean(camera_data[..., :centre[0], :centre[1]], axis=(-1, -2))
# intensity_l = np.mean(camera_data[..., centre[0]:, :centre[1]], axis=(-1, -2))
# intensity_a = np.mean(camera_data[..., :centre[0], centre[1]:], axis=(-1, -2))
# intensity_r = np.mean(camera_data[..., centre[0]:, centre[1]:], axis=(-1, -2))
# intensities = np.array([intensity_d, intensity_l, intensity_a, intensity_r])
# print(intensities.shape)
#
# extra_info = 'intensities_complex_amplitude'
# np.savez(data_path.split('DPC')[0] + extra_info + data_path.split('DPC_scan_')[1],
#          complex_amplitude = lr_complex_amplitude, intensities = intensities, scan_size = scan_size, step_size = step_size)




###### save complex amplitude
#extra_info = 'complex_amplitude'
#camera_data = data['camera_data']
# np.savez(data_path.split('DPC')[0] + extra_info + data_path.split('DPC_scan_')[1],
#     complex_amplitude = complex_amplitude, scan_size = scan_size, step_size = step_size)


grid = Grid(extent=scan_size, step=step_size)

# print('Calculating a phase factor to fix the background (edge) to 0.')
# distance_from_edge = np.amin(
#     np.broadcast_arrays(*(np.minimum(_ - _.ravel()[0], _.ravel()[-1] - _) for _ in grid)),
#     axis=0
# )
# background_edge_width = 3e-6  # The edge over which to average to get the background phase
# weight = np.maximum(0, background_edge_width - distance_from_edge) / background_edge_width
# edge_phasor = np.dot(complex_amplitude.ravel(), weight.ravel())
# edge_phasor /= np.abs(edge_phasor)
# print(f'Background phase is {np.angle(edge_phasor) * 180 / np.pi:0.1f}°.')

