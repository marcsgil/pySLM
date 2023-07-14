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
from pylablib.devices import Thorlabs


def split(images):
    images = np.array([images[..., :images.shape[-1] // 2], images[..., images.shape[-1] // 2:]],
                      dtype=np.float32)
    return images


def show_complex_imgs(imgs):
    fig, axs = plt.subplots(2, 3, sharex=True, sharey= True)
    names = [['Intensity horizontal', 'Intensity vertical', 'Intensity(H.V*)'], ['Field horizontal', 'Field vertical', 'Field H.V*']]
    for _ in range(3):
        axs[0, _].imshow(np.real(imgs[0][_]))
        axs[0, _].set_label(names[0][_])
        axs[1, _].imshow(complex2rgb(imgs[1][_], normalization=True))
        axs[1, _].set_label(names[1][_])
    plt.show(block = True)

###### functions for errors estimation


# input polarizations : diagonal or left circular
def raw_images(cam, stage, grid, sample_name):
    stage.center_all()
    stage.move(1, grid[1][0][0] * 1e6, 'um')
    test_result = 'a'
    while test_result != 'yes':
        single_img = cam.acquire()
        pol_m_effect = MemoryEffect(single_img)
        pol_m_effect_coefficients, c_imgs = pol_m_effect.metric_array, pol_m_effect.complex_images
        single_img = split(single_img)
        test_imgs = [[single_img[0], single_img[1], np.abs(c_imgs[0, 0]*np.conjugate(c_imgs[0, 1]))**2], [c_imgs[0, 0], c_imgs[0, 1], c_imgs[0, 0]*np.conjugate(c_imgs[0, 1])]]
        print(f'Pol memory effect abs mean = {np.abs(pol_m_effect_coefficients)}, phase mean = {np.angle(pol_m_effect_coefficients)}')
        show_complex_imgs(test_imgs)
        test_result = input('Stop calibration? yes/no')
        if test_result != 'yes':
            new_exposure = float(input('New exposure time (ms)'))*1e-3
            cam.exposure_time = new_exposure
            print(f'Camera exposure time: {cam.exposure_time}')

    pol_names = ['horizontal', 'vertical']#, 'diagonal', 'anti', 'left', 'right']
    hp_angles = [265, 221.8, 243.774, 198.539, 245.7703, 265.108]
    qp_angles = [265, 268.4, 268.25, 267.108, 132.22, 215.9]
    stage_hp = Thorlabs.kinesis.KinesisMotor("27602188", scale="PRM1-Z8", default_channel=1, is_rack_system=False)
    stage_qp = Thorlabs.kinesis.KinesisMotor("27265497", scale="PRM1-Z8", default_channel=2, is_rack_system=False)
    if not stage_hp.is_homed(channel=1):
        stage_hp.home(sync=True, force=False, channel=1, timeout=None)
    if not stage_qp.is_homed(channel=2):
        stage_qp.home(sync=True, force=False, channel=2, timeout=None)

    shift_m_e_imgs_all_polarizations = []
    times_all_pol = []


    ### pol_m_e
    pol_effect_positions = [50e-6, 100e-6, 150e-6, 200e-6, 250e-6]
    pol_m_e_imgs = []
    times_m_e = []
    for x in tqdm(pol_effect_positions, 'Scanning ...'):
        stage.move(1, x * 1e6, 'um')
        times = []
        imgs = [[], []]
        for p in range(len(pol_names)):
            print(f'Preparing {pol_names[p]} polarization')
            stage_hp.move_to(hp_angles[p], channel=1, scale=True)
            stage_qp.move_to(qp_angles[p], channel=2, scale=True)
            stage_hp.wait_move(channel=1, timeout=None)
            stage_qp.wait_move(channel=2, timeout=None)
            time.sleep(4)
            for n_acquisition in range(20):
                imgs[p].append(cam.acquire())
                times.append(time.time())

        times_m_e.append(np.array(times))
        pol_m_e_imgs.append(np.array(imgs))

    ##### shift
    y_pos = [150e-6, 200e-6]

    for y_0 in y_pos:
        stage.move(2, y_0 * 1e6, 'um')
        imgs_per_y = []
        for p in range(len(pol_names)):
            print(f'Preparing {pol_names[p]} polarization')
            stage_hp.move_to(hp_angles[p], channel=1, scale=True)
            stage_qp.move_to(qp_angles[p], channel=2, scale=True)
            stage_hp.wait_move(channel=1, timeout=None)
            stage_qp.wait_move(channel=2, timeout=None)
            time.sleep(4)
            imgs = [[], []]
            times = []
            for bv in range(2):
                for _ in tqdm(grid[1][0], 'Scanning ...'):
                    stage.move(1, _ * 1e6, 'um')
                    imgs[bv].append(cam.acquire())
                    times.append(time.time())
            imgs_per_y.append(np.array(imgs))
            times_all_pol.append(np.array(times))
        shift_m_e_imgs_all_polarizations.append(np.array(imgs_per_y))


    file_path = r'D:\transfer\26-05-23\sample_' + sample_name + datetime.now().strftime("%Y-%m-%d_%H-%M-%S.npz")
    control = 1
    while control == 1:
        print(f'Saving data {file_path} ....')
        np.savez(file_path, shift_m_e_images = shift_m_e_imgs_all_polarizations, pol_m_e_images = pol_m_e_imgs, shift_positions = np.array(grid[1][0]),
             pol_m_e_positions = pol_effect_positions ,times_shift = times_all_pol, times_pol_m_e =  times_m_e, pol_name = pol_names)
        try:
            test = np.load(file_path)
            keys = ['shift_m_e_images', 'pol_m_e_images', 'shift_positions', 'pol_m_e_positions', 'times_shift', 'times_pol_m_e', 'pol_name' ]
            for _ in keys:
                vari = test[_]
                print(f' {_} shape:  {np.asarray(vari).shape}')
            print(f'Data validated {file_path}')
            control = 0
        except:
            print(f'Danger, corrupted file {file_path}')
            control = 1





## Todo coordinates of the stage axis (z,x,y)
mode =1# 0: cam test, 1: memory effect, 2: stage center
exposure_time = 1.65e-3
concentration = 0.875
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


elif mode == 1:
    with IDSPeakCam(serial=4104465739, gain=0, exposure_time=exposure_time) as cam:
        cam.roi = Roi(top=100, shape=(1700, 3840))
        step_size = 0.05e-6
        grid = Grid(shape=(100, 1)[::-1], step=step_size, center=(150e-6, 150e-6)[::-1]) # (x,y)
        stage = NanostageLT3()
        sample_name = f'{concentration}c_thick_0.5mm_step_0.05um'
        t1= time.time()
        print(t1)
        raw_images(cam, stage, grid, sample_name)
        print(f'Total time = {(time.time()-t1)/60} minutes')


elif mode == 2:
    # stage test
    stage = NanostageLT3()
    stage.center_all()
    input('press enter')
    # step_size = 1e-6
    # grid = Grid(shape=(100, 1)[::-1], step=step_size, center=(150e-6, 150e-6)[::-1])
    # for _ in tqdm(grid[1][0], 'Scanning ...'):
    #     stage.move(2, _ * 1e6, 'um')
    #     time.sleep(0.2)

