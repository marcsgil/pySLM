import numpy as np
import time
from optics.instruments.cam.ids_peak_cam import IDSPeakCam
import matplotlib.pyplot as plt
from optics.utils.roi import Roi
from projects.confocal_interference_contrast.get_memory_effect import MemoryEffect
from optics.utils.display import complex2rgb, grid2extent
from tqdm import tqdm
from datetime import datetime
from tkinter import filedialog as fd
from optics.instruments.stage.nanostage import NanostageLT3
from optics.utils.ft import Grid
# from optics.instruments.cam.ids_cam import IDSCam, IDSCamError

def imgs_acquisition(cam,number_of_acquisitions, time_delay = None):
    #time_delay: time between two acquisitions in seconds
    imgs = []
    times = []
    for _ in tqdm(range(number_of_acquisitions)):
        imgs.append(cam.acquire())
        if time_delay != None:
            time.sleep(time_delay)
        times.append(time.time())
    times = np.array(times)
    return np.array(imgs), times-times[0]

def split(images):
    images = np.array([images[..., :images.shape[-1] // 2], images[..., images.shape[-1] // 2:]],
                    dtype=np.float32)
    return images

def recombine(img0, img1):
    img = np.concatenate((img0, img1), axis=-1)
    return img

def show(imgs):
    fig, axs = plt.subplots(1, len(imgs))
    for _ in range(len(imgs)):
        axs[_].imshow(imgs[_])
    plt.show()

def show_complex_imgs(imgs):
    fig, axs = plt.subplots(2, 3, sharex=True, sharey= True)
    names = [['Intensity horizontal', 'Intensity vertical', 'Intensity(H.V*)'], ['Field horizontal', 'Field vertical', 'Field H.V*']]
    print(len(imgs))
    for _ in range(3):
        axs[0, _].imshow(np.real(imgs[0][_]))
        axs[0, _].set_label(names[0][_])
        axs[1, _].imshow(complex2rgb(imgs[1][_], normalization=True))
        axs[1, _].set_label(names[1][_])
    plt.show(block = True)

###### functions for errors estimation
def time_dependent_error_data(cam, time_delay, pol_name, sample_name):
    #pol_name: input polarization name
    number_of_acquisitions = 100
    imgs, times = imgs_acquisition(cam, number_of_acquisitions, time_delay)
    imgs_hv = split(imgs) # horizontal pol: [0, _], vertical pol: [1, _]

    if pol_name == 'horizontal':
        imgs = []
        for _ in range(number_of_acquisitions):
            imgs.append(recombine(imgs_hv[0, 0], imgs_hv[0,_]))
    elif pol_name == 'vertical':
        imgs = []
        for _ in range(number_of_acquisitions):
            imgs.append(recombine(imgs_hv[1, 0], imgs_hv[1,_]))

    file_path = r'D:\scan_pol_memory\results\time_dependent_error\sample_' + sample_name + '_pol_' + pol_name + datetime.now().strftime("%Y-%m-%d_%H-%M-%S.npz")
    print(f'Saving file : {file_path}')
    np.savez(file_path, imgs_to_compare = np.array(imgs), times = times, pol_name = pol_name)
    try:
        test = np.load(file_path)
        im = test['imgs_to_compare']
        times = test['times']
        test = 0
        im = 0
        times = 0
        print(f'Data validated {file_path}')
    except:
        print(f'Danger, corrupted file {file_path}')

# input polarizations : diagonal or left circular
def memory_effect(cam, stage, grid,  pol_name, sample_name, test_result):
    ### calculate the shift memory effect for horizontal and vertical polarizations
    ## To add more polarizations, need to change Laurynas' code

    stage.move(1, grid[1][0][0] * 1e6, 'um')
    while test_result != 'yes':
        single_img = cam.acquire()
        pol_m_effect = MemoryEffect(single_img)
        pol_m_effect_coefficients, c_imgs = pol_m_effect.calculate_coefficients()
        single_img = split(single_img)
        test_imgs = [[single_img[0], single_img[1], np.abs(c_imgs[0, 0]*np.conjugate(c_imgs[0, 1]))**2], [c_imgs[0, 0], c_imgs[0, 1], c_imgs[0, 0]*np.conjugate(c_imgs[0, 1])]]
        print(f'Pol memory effect abs mean = {np.abs(pol_m_effect_coefficients)}, phase mean = {np.angle(pol_m_effect_coefficients)}')
        show_complex_imgs(test_imgs)
        test_result = input('Stop callibration? yes/no')
        if test_result != 'yes':
            new_exposure = float(input('New exposure time (ms)'))*1e-3
            cam.exposure_time = new_exposure
            print(f'Camera exposure time: {cam.exposure_time}')
    imgs = []
    times = []
    for _ in tqdm(grid[1][0], 'Scanning ...'):
        stage.move(1, _ * 1e6, 'um')
        imgs.append(cam.acquire())
        times.append(time.time())
    times = np.array(times)
    hv_imgs = split(np.array(imgs))
    shift_me_images = [[], []]
    for _ in range(hv_imgs.shape[1]):
        shift_me_images[0].append(recombine(hv_imgs[0, int(hv_imgs.shape[1]/2)], hv_imgs[0, _]))
        shift_me_images[1].append(recombine(hv_imgs[1, int(hv_imgs.shape[1]/2)], hv_imgs[1, _]))

    file_path = r'Z:\memory_effect\sample_' + sample_name + '_input_polatization_' + pol_name + datetime.now().strftime("%Y-%m-%d_%H-%M-%S.npz")
    # file_path = r'D:\scan_pol_memory\results\memory_effect_data\sample_' + sample_name + '_input_polatization_' + pol_name + datetime.now().strftime("%Y-%m-%d_%H-%M-%S.npz")
    np.savez(file_path, shift_me_images = shift_me_images, pol_me_images = np.array(imgs), positions = np.array(grid[1][0]), time = times-times[0], pol_name = pol_name)

def shift_memory_effect(cam, stage, grid,  pol_name, sample_name):
    ### calculate the shift memory effect for horizontal and vertical polarizations
    ## To add more polarizations, need to change Laurynas' code
    imgs = []
    for _ in tqdm(grid[1][0], 'Scanning ...'):
        stage.move(1, _ * 1e6, 'um')
        time.sleep(0.1)
        imgs.append(cam.acquire())
    hv_imgs = split(np.array(imgs))

    if pol_name == 'horizontal':
        imgs = []
        for _ in range(hv_imgs.shape[1]):
            imgs.append(recombine(hv_imgs[0, 0], hv_imgs[0, _]))

    elif pol_name == 'vertical':
        imgs = []
        for _ in range(hv_imgs.shape[1]):
            imgs.append(recombine(hv_imgs[1, 0], hv_imgs[1, _]))

    elif pol_name == 'diagonal':
        imgs = []
        imgs_h = []
        imgs_v = []
        for _ in range(hv_imgs.shape[1]):
            imgs_h.append(recombine(hv_imgs[0, 0], hv_imgs[0, _]))
        for _ in range(hv_imgs.shape[1]):
            imgs_v.append(recombine(hv_imgs[1, 0], hv_imgs[1, _]))

    if pol_name == 'diagonal':
        file_path = r'D:\scan_pol_memory\results\shift_memory_effect\shift_sample_' + sample_name +'_input_pol_' + pol_name +'_output_' + 'horizontal_' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S.npz")
        time.sleep(0.1)
        np.savez(file_path, imgs_to_compare = np.array(imgs_h), positions = np.array(grid[1][0]), pol_name = pol_name)
        time.sleep(0.2)
        file_path = r'D:\scan_pol_memory\results\shift_memory_effect\shift_sample_' + sample_name +'_input_pol_' + pol_name +'_output_' + 'vertical_' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S.npz")
        time.sleep(0.1)
        np.savez(file_path, imgs_to_compare = np.array(imgs_v), positions = np.array(grid[1][0]), pol_name = pol_name)
        time.sleep(0.2)
    else:
        file_path = r'D:\scan_pol_memory\results\shift_memory_effect\shift_sample_' + sample_name + '_pol_' + pol_name + datetime.now().strftime("%Y-%m-%d_%H-%M-%S.npz")
        time.sleep(0.1)
        np.savez(file_path, imgs_to_compare = np.array(imgs), positions = np.array(grid[1][0]), pol_name = pol_name)
        time.sleep(0.2)
    try:
        test = np.load(file_path)
        imgs_to_compare = test['imgs_to_compare']
        test = 0
        imgs_to_compare = 0
        print(f'Data validated {file_path}')
    except:
        print(f'Danger, corrupted file {file_path}')

def hv_memory_effect(cam, number_of_acquisitions, sample_name):
    ## Calculate the difference between the horizontal and vertical polarizations.. We use the overlap integral to estimate the diffenrence.
    ## 100 images (for statistics) are taken over time at the same point of the sample.
    file_path = r'D:\scan_pol_memory\results\memory_effect_hv\sample_' + sample_name + '_pol_hv_' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S.npz")
    input('change to horizontal polarization')
    h_imgs, times = imgs_acquisition(cam, number_of_acquisitions, time_delay=None)
    input('change to vertical polarization')
    v_imgs, times_v = imgs_acquisition(cam, number_of_acquisitions, time_delay=None)
    h_images = split(h_imgs)
    # h_imgs = 0
    v_images = split(v_imgs)
    v_imgs = 0
    imgs_to_compare = []
    show([h_images[0, 0], v_images[1, 0]])
    for _ in range(number_of_acquisitions):
        imgs_to_compare.append(recombine(h_images[0,_],v_images[1,_])) # todo change back to h_images[0,_], v_images[1,_]
    np.savez(file_path, imgs_to_compare = np.array(imgs_to_compare), times = times)
    try:
        test = np.load(file_path)
        im = test['imgs_to_compare']
        times = test['times']
        test = 0
        im = 0
        times = 0
        print(f'Data validated {file_path}')
    except:
        print(f'Danger, corrupted file {file_path}')

def diagonal_memory_effect(cam, number_of_acquisitions, sample_name):
    ## Calculate the difference between the horizontal and vertical polarizations.. We use the overlap integral to estimate the diffenrence.
    ## 100 images (for statistics) are taken over time at the same point of the sample.
    file_path = r'D:\scan_pol_memory\results\memory_effect_hv\sample_' + sample_name  + datetime.now().strftime("%Y-%m-%d_%H-%M-%S.npz")
    imgs, times = imgs_acquisition(cam, number_of_acquisitions, time_delay=None)
    np.savez(file_path, imgs_to_compare = np.array(imgs), times = times)
    try:
        test = np.load(file_path)
        im = test['imgs_to_compare']
        times = test['times']
        test = 0
        im = 0
        times = 0
        print(f'Data validated {file_path}')
    except:
        print(f'Danger, corrupted file {file_path}')

def plot_m_effect_read_data(mode = 'position'):
    input_directories = [r'D:\scan_pol_memory\results']
    reference_path = fd.askopenfilenames(initialdir=input_directories[0], title='Images to compare data:')
    # times = []
    # m_effect_coefficientes = []
    # pol_names = []
    # reference_path = ['D:/scan_pol_memory/results/time_dependent_error/sample_test_3_pol_horizontal2023-04-13_18-24-20.npz']
    fig, ax = plt.subplots(1, 2)
    for _ in reference_path:
        print(f'loading {_}')
        data = np.load(_)
        imgs_to_compare = data['imgs_to_compare']
        if mode == 'position':
            x_axis = data['positions']
        else:
            mode = 'time'
            x_axis = data['times']
        pol_name = data['pol_name']
        memeffect = MemoryEffect(imgs_to_compare)
        m_effect_coefficientes = memeffect.calculate_coefficients()
        mean_abs = np.mean(np.abs(m_effect_coefficientes))
        std_abs = np.std(np.abs(m_effect_coefficientes))
        mean_phase = np.mean(np.angle(m_effect_coefficientes))
        std_phase = np.std(np.angle(m_effect_coefficientes))
        print(f'{pol_name} pol. memory effect mean absolute value = {mean_abs} +- {std_abs}')
        print(f'{pol_name} pol. memory effect mean phase value = {mean_phase} +- {std_phase}')
        ax[0].errorbar(x_axis * 1e6, np.abs(m_effect_coefficientes), label=pol_name, fmt = '.')   #, yerr= std_abs, label=pol_name, fmt = '.')
        ax[1].errorbar(x_axis * 1e6, np.angle(m_effect_coefficientes), label=pol_name, fmt = '.')# yerr= std_phase, label=pol_name, fmt = '.')
    ax[0].set_title("Amplitude")
    ax[0].set_xlabel("Time (seconds)")
    ax[0].set_ylabel('Overlap integral (abs. value)')
    ax[0].legend(loc='lower right')
    ax[1].set_title("Phase")
    ax[1].set_xlabel("Time (seconds)")
    ax[1].set_ylabel('Overlap integral (abs. value)')
    ax[1].legend(loc='lower right')

    plt.show()

def stage_precision():
    input_directories = [r'D:\scan_pol_memory\results']
    reference_path = fd.askopenfilenames(initialdir=input_directories[0], title='Images to compare data:')
    reference_data = np.load(reference_path[0])
    data = reference_data['imgs_to_compare']
    reference_hv = split(data)
    positions = reference_data['positions']
    c = 0
    for _ in reference_path[1::]:
        print(f'loading {_}')
        data = np.load(_)
        imgs_to_compare = data['imgs_to_compare']
        imgs_hv = split(imgs_to_compare)
        imgs = []
        for _ in range(reference_hv.shape[1]):
            imgs.append(recombine(reference_hv[1, _], imgs_hv[1, _]))
        file_path = r'D:\scan_pol_memory\results\time_dependent_error\shift_sample_pol_horizontal_'+str(c) +'_' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S.npz")
        time.sleep(0.5)
        np.savez(file_path, imgs_to_compare = np.array(imgs), positions = positions, pol_name = 'horizontal')
        time.sleep(0.5)
        try:
            test = np.load(file_path)
            im = test['imgs_to_compare']
            test = 0
            im = 0
            print(f'Data validated {file_path}')
        except:
            print(f'Danger, corrupted file {file_path}')
        c += 1

####polarizations coordinates

# Horizontal - 265.0480
# diagonal - 242.6283
# vertical - 219.6
# anti-diagonal - 197.2875

## Todo coordinates of the stage axis (z,x,y)
mode =3#3 # 0: cam test, 1: time dependent error, 2: shift, 3: memory effect, 4: diagonal memory 5: stage center
exposure_time = 6e-3
concentration = 0
pol_name = 'circular' # diagonal circular
if mode == 0:
    # camera test
    with IDSPeakCam(serial=4104465739, gain=0, exposure_time=exposure_time) as cam:
        cam.roi = Roi(top=100, shape=(1700, 3840))
        stage = NanostageLT3()
        stage.center_all()
        img = cam.acquire()
        plt.imshow(img)
        plt.show(block =True)
        imgs = split(img)
        show(imgs)

elif mode == 1:
    # time dependent error (h, v)
    with IDSPeakCam(serial=4104465739, gain=0, exposure_time=exposure_time) as cam:
        cam.roi = Roi(top=100, shape=(1700, 3840))
        stage = NanostageLT3()
        stage.center_all()
        time_delay = [None, None, None, 0.6, 0.6, 2*0.6, 5*0.6, 10*0.6]
        pol_name = ['horizontal', 'vertical', 'diagonal']
        for _ in time_delay:
            sample_name = f'{concentration}c_thick_0.5mm_time_delay_{_}s'
            time_dependent_error_data(cam, _, pol_name[0], sample_name) # todo change the pol

elif mode == 2:
    # shift memory effect (h, v, d)
    with IDSPeakCam(serial=4104465739, gain=0, exposure_time=exposure_time) as cam:
        cam.roi = Roi(top=100, shape=(1700, 3840))
        stage = NanostageLT3()
        stage.center_all()
        depth = 150e-6
        time.sleep(0.5)
        step_size = 0.01e-6
        grid = Grid(shape=(200, 1)[::-1], step=step_size, center=(150e-6, 150e-6)[::-1]) # (x,y)
        pol_name = ['horizontal', 'vertical', 'diagonal']
        sample_name = f'{concentration}c_thick_0.5mm_step_{step_size*1e6}um'
        # y_positions = [ 130, 170]   # um
        # for pos in y_positions:
        #     stage.move(2, pos, 'um')
        for _ in range (2):
            shift_memory_effect(cam, stage, grid, pol_name[1], sample_name)  # todo change the pol

elif mode == 3:
    # memory_effect
    with IDSPeakCam(serial=4104465739, gain=0, exposure_time=exposure_time) as cam:
        cam.roi = Roi(top=100, shape=(1700, 3840))
        stage = NanostageLT3()
        stage.center_all()
        step_size = [0.002e-6, 0.02e-6, 0.2e-6]
        y_pos = [130e-6, 150e-6, 170e-6]
        test_result = 'no'
        for _ in y_pos:
            for st in step_size:
                stage.move(0, _ * 1e6, 'um')
                grid = Grid(shape=(100, 1)[::-1], step=st, center=(150e-6, 150e-6)[::-1]) # (x,y)
                sample_name = f'{concentration}c_thick_0.5mm_step{st*1e6: .3f}um'
                memory_effect(cam, stage, grid,  pol_name, sample_name, test_result)
                test_result = 'yes'

elif mode == 4:
    # diagonal memory effect
    with IDSPeakCam(serial=4104465739, gain=0, exposure_time=exposure_time) as cam:
        cam.roi = Roi(top=100, shape=(1700, 3840))
        stage = NanostageLT3()
        stage.center_all()
        number_of_acquisitions = 100
        sample_name = f'{concentration}c_thick_0.5mm_diagonal_memory_'
        for _ in range (3):
            diagonal_memory_effect(cam, number_of_acquisitions, sample_name)

elif mode == 5:
    # stage test
    stage = NanostageLT3()
    stage.center_all()
    input('press enter')
    # step_size = 1e-6
    # grid = Grid(shape=(100, 1)[::-1], step=step_size, center=(150e-6, 150e-6)[::-1])
    # for _ in tqdm(grid[1][0], 'Scanning ...'):
    #     stage.move(2, _ * 1e6, 'um')
    #     time.sleep(0.2)

elif mode == 6:
    # shift memory effect (h, v, d)
    with IDSPeakCam(serial=4104465739, gain=0, exposure_time=exposure_time) as cam:
        cam.roi = Roi(top=100, shape=(1700, 3840))
        stage = NanostageLT3()
        stage.center_all()
        depth = 150e-6
        time.sleep(0.5)
        step_size = 0.01e-6
        grid = Grid(shape=(200, 1)[::-1], step=step_size, center=(150e-6, 150e-6)[::-1]) # (x,y)
        pol_name = ['horizontal', 'vertical', 'diagonal']
        #exposure_times = [1e-3, 5e-3, 10e-3, 20e-3, 30e-3, 40e-3, 50e-3, 60e-3]
        exposure_times = [10e-3]
        for _ in range(20):#exposure_times:
            cam.exposure_time = _
            sample_name = f'{concentration}c_thick_0.5mm_exp_time_{_}s_'
            # shift_memory_effect(cam, stage, grid, pol_name[0], sample_name)  # todo change the pol
            diagonal_memory_effect(cam, 10, sample_name)