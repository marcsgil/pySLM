import logging
import time
import numpy as np
import matplotlib.pyplot as plt
from projects.confocal_interference_contrast.get_memory_effect import MemoryEffect
from tkinter import filedialog as fd
from tqdm import tqdm
from scipy.optimize import curve_fit
from datetime import datetime


def split(images):
    images = np.array([images[..., :images.shape[-1] // 2], images[..., images.shape[-1] // 2:]], dtype=np.float32)
    return images

def recombine(img0, img1):
    img = np.concatenate((img0, img1), axis=-1)
    return img

def show(imgs):
    fig, axs = plt.subplots(1, len(imgs))
    for _ in range(len(imgs)):
        axs[_].imshow(imgs[_])
    plt.show()

def func(x, a, b):
    return a*np.exp(b*x)

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
        file_path = r'D:\scan_pol_memory\results\shift_memory_effect\stage_precision\stage_precision_sample_1c_pol_horizontal_'+str(c) +'_' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S.npz")
        np.savez(file_path, imgs_to_compare = np.array(imgs), positions = positions, pol_name = 'horizontal')
        try:
            test = np.load(file_path)
            im = test['imgs_to_compare']
            test = 0
            im = 0
            print(f'Data validated {file_path}')
        except:
            print(f'Danger, corrupted file {file_path}')
        c += 1

def process_raw_imgs():
    # input_directories = [r'D:\scan_pol_memory\results\memory_effect_data']
    input_directories = [r'B:\memory_effect_raw_imgs']
    reference_path = list(fd.askopenfilenames(initialdir=input_directories[0], title='Images to compare data:'))
    c = 0
    for _ in reference_path:
        start_time = time.time()
        print(f'[{c} of {len(reference_path)}]  loading {_}')
        data = np.load(_)
        keys = ['shift_m_e_images', 'pol_m_e_images', 'shift_positions', 'pol_m_e_positions', 'times_shift', 'times_pol_m_e', 'pol_name']
        # pol memory effect calculations
        pol_m_e_images = data[keys[1]]   # shape (5,6, img)
        t2 = time.time() - start_time
        print(f'Time spent for load the pme data {t2} seconds ')
        pol_names = ['horizontal', 'vertical', 'diagonal', 'anti', 'left', 'right']
        pol_names = pol_names[:2]  # Just looking at the first two polarizations, H and V
        list_of_comparisons = []
        list_of_coefficients_pme = []
        fringe_periods = None#[242, -65]  # TODO: Check that the spatial frequency is correct!
        for point in range(len(pol_m_e_images)):
            pme_coefficients_per_point = []
            for input_pol in range(len(pol_names)-1):
                for comparison in range(1+input_pol, len(pol_names)):
                    input_pol_1 = split(pol_m_e_images[point][input_pol])
                    input_pol_2 = split(pol_m_e_images[point][comparison])
                    if pol_names[input_pol] == 'horizontal' and pol_names[comparison] == 'vertical':
                        # note that: 0 - intensity image of horizontal component of field, 1 - intensity image of horizontal component of field
                        # for this specific case I will compare the horizontal component with the vertical
                        img_to_compare_h = recombine(input_pol_1[0], input_pol_2[1])
                        img_to_compare_v = recombine(input_pol_1[1], input_pol_2[0])
                        print('H-V PME Img')
                    else:
                        img_to_compare_h = recombine(input_pol_1[0], input_pol_2[0])
                        img_to_compare_v = recombine(input_pol_1[1], input_pol_2[1])
                    obj_me = [MemoryEffect(img_to_compare_h, fringe_periods=fringe_periods),
                              MemoryEffect(img_to_compare_v, fringe_periods=fringe_periods)]
                    pme_coefficients = [obj_me[0].metric_array, obj_me[1].metric_array]
                    pme_coefficients_per_point.append(np.asarray(pme_coefficients))
                    if point == 0:
                        list_of_comparisons.append(f'{pol_names[input_pol]} x {pol_names[comparison]}')
                    print(f'{pol_names[input_pol]} x {pol_names[comparison]} - H: {np.abs(pme_coefficients[0])}; v: {np.abs(pme_coefficients[1])} ')
            list_of_coefficients_pme.append(pme_coefficients_per_point)  # shape per point (15, 2, 1700, 3840)
        pol_m_e_images = 0  # just to clear ram space
        # shift memory effect calculations
        shift_m_e_images = data[keys[0]] # shape (12, 100, img): [2 y positions at sample, 6 input pol] = 12 , 100 points along a line in x direction
        print(f'Time spent for load the sme data {(time.time() - start_time) - t2} seconds ')
        print(f'len shift_m_e_images {len(shift_m_e_images)}, shape shift_m_e_images[0] {shift_m_e_images[0].shape}')
        line_per_input_pol = [[shift_m_e_images[0], shift_m_e_images[6]], [shift_m_e_images[1],shift_m_e_images[7]]] # shape [polarization][y_position]
        number_imgs_per_line = len(line_per_input_pol[0][0])
        print(f'number_imgs_per_line = {number_imgs_per_line}')
        list_of_coefficients_sme = []
        for y_pos in range(2):
            reference_img_h = split(line_per_input_pol[0][y_pos][number_imgs_per_line//2])
            reference_img_v = split(line_per_input_pol[1][y_pos][number_imgs_per_line//2])
            hh_img = [[],[]]
            vv_img = [[],[]]
            hv_img = [[],[]]
            vh_img = [[],[]]
            for img_number in range(number_imgs_per_line):
                h_pol_imgs = split(line_per_input_pol[0][y_pos][img_number])
                v_pol_imgs = split(line_per_input_pol[1][y_pos][img_number])
                hh_img[0].append(recombine(reference_img_h[0], h_pol_imgs[0]))
                hh_img[1].append(recombine(reference_img_h[1], h_pol_imgs[1]))
                vv_img[0].append(recombine(reference_img_v[0], v_pol_imgs[0]))
                vv_img[1].append(recombine(reference_img_v[1], v_pol_imgs[1]))
                hv_img[0].append(recombine(reference_img_h[0], v_pol_imgs[1]))
                hv_img[1].append(recombine(reference_img_h[1], v_pol_imgs[0]))
                vh_img[0].append(recombine(reference_img_v[0], h_pol_imgs[1]))
                vh_img[1].append(recombine(reference_img_v[1], h_pol_imgs[0]))
            print(f' shape hh = {np.asarray(hh_img).shape}')
            print(f' shape hh[0] = {np.asarray(hh_img[0]).shape}')
            obj_me = [[MemoryEffect(hh_img[0], fringe_periods=fringe_periods), MemoryEffect(hh_img[1], fringe_periods=fringe_periods)],
                      [MemoryEffect(vv_img[0], fringe_periods=fringe_periods), MemoryEffect(vv_img[1], fringe_periods=fringe_periods)],
                      [MemoryEffect(hv_img[0], fringe_periods=fringe_periods), MemoryEffect(hv_img[1], fringe_periods=fringe_periods)],
                      [MemoryEffect(vh_img[0], fringe_periods=fringe_periods), MemoryEffect(vh_img[1], fringe_periods=fringe_periods)]]
            sme_coefficients_per_line = [[obj_me[0][0].metric_array,obj_me[0][1].metric_array],
                                         [obj_me[1][0].metric_array,obj_me[1][1].metric_array],
                                         [obj_me[2][0].metric_array,obj_me[2][1].metric_array],
                                         [obj_me[3][0].metric_array,obj_me[3][1].metric_array]]
            list_of_coefficients_sme.append(np.array(sme_coefficients_per_line))

        file_path = _.split('sample')[0] + 'coefficients_sample' + _.split('sample')[1]
        np.savez(file_path, coefficients_pme=np.array(list_of_coefficients_pme), list_of_comparisons_pme=list_of_comparisons,
                 coefficients_sme=np.array(list_of_coefficients_sme),
                 shift_positions=data['shift_positions'], pol_m_e_positions=data['pol_m_e_positions'],
                 times_shift=data['times_shift'], times_pol_m_e=data['times_pol_m_e'], pol_name=data['pol_name'])
        print(f'Total Time spent {(time.time() - start_time)/60} minutes ')

def process_raw_imgs_v2():
    # input_directories = [r'D:\scan_pol_memory\results\memory_effect_data']
    input_directories = [r'B:\memory_effect_raw_imgs']
    reference_path = list(fd.askopenfilenames(initialdir=input_directories[0], title='Images to compare data:'))
    # reference_path = [r"D:\transfer\26-05-23\sample_1c_thick_0.5mm_step_0.05um2023-05-26_17-33-41.npz"]
    c = 0
    for ref_path in reference_path:
        start_time = time.time()
        print(f'[{c} of {len(reference_path)}]  loading {ref_path}')
        data = np.load(ref_path)
        keys = ['shift_m_e_images', 'pol_m_e_images', 'shift_positions', 'pol_m_e_positions', 'times_shift', 'times_pol_m_e', 'pol_name']
        # pol memory effect calculations
        pol_m_e_images = np.asarray(data[keys[1]])   # shape (5, 2, 20, 1700, 3840): [point at sample, pol (h;v), imgs_per_pol, img ]
        pol_names = ['horizontal', 'vertical', 'diagonal', 'anti', 'left', 'right']
        pol_names = pol_names[:2]  # Just looking at the first two polarizations, H and V
        list_of_comparisons = ['HH', 'VV', 'HV']
        list_of_coefficients_pme = []
        fringe_periods =[242, -65]  # TODO: Check that the spatial frequency is correct!
        list_benchmark_pme_coefficients = []
        for point in range(pol_m_e_images.shape[0]):
            pme_coefficients_per_point = []
            benchmark_pme_coefficients_per_point = []
            h_benchmark_ref = split(pol_m_e_images[point,0,0])[0]
            v_benchmark_ref = split(pol_m_e_images[point,1,0])[1]
            for img_number in tqdm(range(20), f'PME point {point}'): #todo  20
                h_input = split(pol_m_e_images[point,0,img_number])[0]   # h comp
                v_input = split(pol_m_e_images[point,1,img_number])[1]   # v comp
                img_to_compare_hv = recombine(h_input, v_input)
                img_to_compare_hh = recombine(h_benchmark_ref, h_input)
                img_to_compare_vv = recombine(v_benchmark_ref, v_input)
                obj_me = [MemoryEffect(img_to_compare_hv, fringe_periods=fringe_periods),
                          MemoryEffect(img_to_compare_hh, fringe_periods=fringe_periods),
                          MemoryEffect(img_to_compare_vv, fringe_periods=fringe_periods)]
                pme_coefficients = obj_me[0].metric_array
                benchmark_coefficients = [obj_me[1].metric_array, obj_me[2].metric_array]
                pme_coefficients_per_point.append(np.asarray(pme_coefficients))
                benchmark_pme_coefficients_per_point.append(np.asarray(benchmark_coefficients))

            list_of_coefficients_pme.append(pme_coefficients_per_point)
            list_benchmark_pme_coefficients.append(benchmark_pme_coefficients_per_point)
        pol_m_e_images = 0  # just to clear ram space
        # shift memory effect calculations
        shift_m_e_images = np.asarray(data[keys[0]]) # shape (2, 2, 2, 100, 1700, 3840): [y_pos, pol, line_scan_number,x_pos,img
        print(f'len shift_m_e_images {len(shift_m_e_images)}, shape shift_m_e_images[0] {shift_m_e_images[0].shape}')
        list_of_coefficients_sme_benchmark = []
        list_of_coefficients_spme = []
        list_of_coefficients_sme = []
        for y_pos in range(2):
            h_l1_ref = split(shift_m_e_images[y_pos,0 ,0 ,100//2])[0]
            h_l2_ref = split(shift_m_e_images[y_pos,0 ,1 ,100//2])[0]
            v_l1_ref = split(shift_m_e_images[y_pos,1 ,0 ,100//2])[1]
            v_l2_ref = split(shift_m_e_images[y_pos,1 ,1 ,100//2])[1]
            hh_benchmark_imgs = []
            vv_benchmark_imgs = []
            h1h1_sme_imgs = []
            v1v1_sme_imgs =[]
            h2h2_sme_imgs =[]
            v2v2_sme_imgs =[]
            h2v1_spme_imgs =[]
            h1v2_spme_imgs =[]
            v1h2_spme_imgs =[]
            v2h1_spme_imgs =[]
            coefficients_list = []
            for img_number in tqdm(range(100), f'SME y_pos = {y_pos}'): #todo 100
                h_l1 = split(shift_m_e_images[y_pos,0 ,0,img_number ])[0]
                h_l2 = split(shift_m_e_images[y_pos,0 ,1, img_number ])[0]
                v_l1 = split(shift_m_e_images[y_pos,1 ,0, img_number ])[1]
                v_l2 = split(shift_m_e_images[y_pos,1 ,1,img_number ])[1]
                hh_benchmark_imgs.append(recombine(h_l1_ref, h_l2)) #0
                vv_benchmark_imgs.append(recombine(v_l1_ref, v_l2))#1
                h1h1_sme_imgs.append(recombine(h_l1_ref, h_l1))#2
                v1v1_sme_imgs.append(recombine(v_l1_ref, v_l1))#3
                h2h2_sme_imgs.append(recombine(h_l2_ref, h_l2))#4
                v2v2_sme_imgs.append(recombine(v_l2_ref, v_l2))#5
                h2v1_spme_imgs.append(recombine(h_l2_ref, v_l1))#6
                h1v2_spme_imgs.append(recombine(h_l1_ref, v_l2))#7
                v1h2_spme_imgs.append(recombine(v_l1_ref, h_l2))#8
                v2h1_spme_imgs.append(recombine(v_l2_ref, h_l1))#9
            obj_me = [MemoryEffect(hh_benchmark_imgs, fringe_periods=fringe_periods), MemoryEffect(vv_benchmark_imgs, fringe_periods=fringe_periods),
                      MemoryEffect(h1h1_sme_imgs, fringe_periods=fringe_periods), MemoryEffect(v1v1_sme_imgs, fringe_periods=fringe_periods),
                      MemoryEffect(h2h2_sme_imgs, fringe_periods=fringe_periods), MemoryEffect(v2v2_sme_imgs, fringe_periods=fringe_periods),
                      MemoryEffect(h2v1_spme_imgs, fringe_periods=fringe_periods), MemoryEffect(h1v2_spme_imgs, fringe_periods=fringe_periods),
                      MemoryEffect(v1h2_spme_imgs, fringe_periods=fringe_periods), MemoryEffect(v2h1_spme_imgs, fringe_periods=fringe_periods)]
            for _ in obj_me:
                coefficients_list.append(_.metric_array)
            list_of_coefficients_sme_benchmark.append(np.array([coefficients_list[0], coefficients_list[1]]))
            list_of_coefficients_sme.append(np.array([coefficients_list[2], coefficients_list[3],coefficients_list[4], coefficients_list[5]]))
            list_of_coefficients_spme.append(np.array([coefficients_list[6], coefficients_list[7],coefficients_list[8], coefficients_list[9]]))
        print(f'list_of_coefficients_pme shape = {np.asarray(list_of_coefficients_pme).shape}')
        print(f'list_benchmark_pme_coefficients shape = {np.asarray(list_benchmark_pme_coefficients).shape}')
        print(f'list_of_coefficients_sme shape = {np.asarray(list_of_coefficients_sme).shape}')
        print(f'list_of_coefficients_sme_benchmark = {np.asarray(list_of_coefficients_sme_benchmark).shape}')
        print(f'list_of_coefficients_spme = {np.asarray(list_of_coefficients_spme).shape}')
        file_path = ref_path.split('sample')[0] + f'coefficients_fringe_{fringe_periods}_sample' + ref_path.split('sample')[1]
        np.savez(file_path, coefficients_pme=np.array(list_of_coefficients_pme), benchmark_pme_coefficients= np.array(list_benchmark_pme_coefficients),
                 list_of_comparisons_pme=list_of_comparisons,
                 coefficients_sme=np.array(list_of_coefficients_sme),
                 coefficients_spme = np.array(list_of_coefficients_spme),
                 coefficients_sme_benchmark = np.array(list_of_coefficients_sme_benchmark),
                 shift_positions=data['shift_positions'], pol_m_e_positions=data['pol_m_e_positions'],
                 times_shift=data['times_shift'], times_pol_m_e=data['times_pol_m_e'], pol_name=data['pol_name'])
        print(f'Total Time spent {(time.time() - start_time)/60} minutes ')

def process_raw_imgs_v2_stage():
    input_directories = [r'B:\memory_effect_raw_imgs']
    reference_path = list(fd.askopenfilenames(initialdir=input_directories[0], title='Images to compare data:'))
    c = 0
    for ref_path in reference_path:
        start_time = time.time()
        print(f'[{c} of {len(reference_path)}]  loading {ref_path}')
        data = np.load(ref_path)
        keys = ['shift_m_e_images', 'pol_m_e_images', 'shift_positions', 'pol_m_e_positions', 'times_shift', 'times_pol_m_e', 'pol_name']
        fringe_periods =None #[242, -65]  # TODO: Check that the spatial frequency is correct!
        shift_m_e_images = np.asarray(data[keys[0]]) # shape (2, 2, 2, 100, 1700, 3840): [y_pos, pol, line_scan_number,x_pos,img
        list_of_coefficients_sme_benchmark = []
        for y_pos in range(2):
            hh_benchmark_imgs = []
            vv_benchmark_imgs = []
            coefficients_list = []
            for img_number in tqdm(range(100), f'SME y_pos = {y_pos}'): #todo 100
                h_l1 = split(shift_m_e_images[y_pos,0 ,0,img_number ])[0]
                h_l2 = split(shift_m_e_images[y_pos,0 ,1, img_number ])[0]
                v_l1 = split(shift_m_e_images[y_pos,1 ,0, img_number ])[1]
                v_l2 = split(shift_m_e_images[y_pos,1 ,1,img_number ])[1]
                hh_benchmark_imgs.append(recombine(h_l1, h_l2)) #0
                vv_benchmark_imgs.append(recombine(v_l1, v_l2))#1
            obj_me = [MemoryEffect(hh_benchmark_imgs, fringe_periods=fringe_periods), MemoryEffect(vv_benchmark_imgs, fringe_periods=fringe_periods)]
            for _ in obj_me:
                coefficients_list.append(_.metric_array)
            list_of_coefficients_sme_benchmark.append(np.array([coefficients_list[0], coefficients_list[1]]))
        file_path = ref_path.split('sample')[0] + f'stage_precision_fringe_{fringe_periods}_sample' + ref_path.split('sample')[1]
        np.savez(file_path, stage_precision_coefficients = np.array(list_of_coefficients_sme_benchmark))
        print(f'Total Time spent {(time.time() - start_time)/60} minutes ')
        c += 1


def read_results_raw_imgs():
    # input_directories = [r'B:\memory_effect_raw_imgs']
    # reference_paths = list(fd.askopenfilenames(initialdir=input_directories[0], title='Images to compare data:'))
    reference_paths = [r"B:\memory_effect_raw_imgs\coefficients_sample_0c_thick_0.5mm_step_0.05um2023-05-17_15-15-27.npz",
                       r"B:\memory_effect_raw_imgs\coefficients_sample_0.5c_thick_0.5mm_step_0.05um2023-05-17_15-36-00.npz",
                       r"B:\memory_effect_raw_imgs\coefficients_sample_1c_thick_0.5mm_step_0.05um2023-05-17_15-56-29.npz",
                       r"B:\memory_effect_raw_imgs\coefficients_sample_1.5c_thick_0.5mm_step_0.05um2023-05-17_16-17-27.npz",
                       r"B:\memory_effect_raw_imgs\coefficients_sample_2c_thick_0.5mm_step_0.05um2023-05-17_16-37-48.npz",
                       r"B:\memory_effect_raw_imgs\coefficients_sample_2.5c_thick_0.5mm_step_0.05um2023-05-17_16-58-42.npz",
                       r"B:\memory_effect_raw_imgs\coefficients_sample_3c_thick_0.5mm_step_0.05um2023-05-17_17-20-31.npz",
                       r"B:\memory_effect_raw_imgs\coefficients_sample_3.5c_thick_0.5mm_step_0.05um2023-05-17_17-39-46.npz",
                       r"B:\memory_effect_raw_imgs\coefficients_sample_4c_thick_0.5mm_step_0.05um2023-05-17_18-00-07.npz",
                       r"B:\memory_effect_raw_imgs\coefficients_sample_4.5c_thick_0.5mm_step_0.05um2023-05-17_18-18-56.npz",
                       r"B:\memory_effect_raw_imgs\coefficients_sample_5c_thick_0.5mm_step_0.05um2023-05-17_18-39-02.npz"]
    fig, axs = plt.subplots(2, 2, sharex=True)
    fig2, axs2 = plt.subplots(2, 2, sharex=True)
    fig3, axs3 = plt.subplots(2, 2, sharex=True)
    concentrations_list = []
    pme_values_list = []
    for reference_path in reference_paths[:]:
        concentration = reference_path.split('sample_')[1].split('c_')[0]
        print(f'Data {reference_path}')
        print(f'Concentration {concentration} c ')
        concentrations_list.append(float(concentration))
        data = np.load(reference_path)
        coefficients_pme = data['coefficients_pme'] #  shape (5, 1, 2, 1, 4)
        list_of_comparisons_pme = data['list_of_comparisons_pme']
        coefficients_sme = data['coefficients_sme'] #array shape (2, 4, 2, 100, 4)   (y_pos, comparison (HH, VV, HV, vH), comp (h, v),   )
        shift_positions = data['shift_positions']
        pol_m_e_positions = data['pol_m_e_positions']
        times_shift = data['times_shift']
        times_pol_m_e = data['times_pol_m_e']
        pol_name = data['pol_name']
        coefficients_pme_mean = np.mean(np.abs(coefficients_pme), axis=0)
        coefficients_pme_std = np.std(np.abs(coefficients_pme), axis=0)
        pme_values_list.append(coefficients_pme_mean[0,0,0])
        for _ in range(len(list_of_comparisons_pme)):
            print(f'PME: concentration {concentration} c; Innet product -- {coefficients_pme_mean[0,0,0,1]} +- {coefficients_pme_std[0,0,0,1]}')
            print(f'PME: concentration {concentration} c; N inner product -- {coefficients_pme_mean[0,0,0,3]} +- {coefficients_pme_std[0,0,0,3]}')
        metric = 2
        coefficients_sme_mean = np.mean(np.abs(coefficients_sme), axis=0)   # shape (4, 2, 100, 4) - (comparison, field component, imgs, metric)
        coefficients_sme_std = np.std(np.abs(coefficients_sme), axis=0)
        good_comp = [coefficients_sme_mean[0,0,:,metric], coefficients_sme_mean[1,1,:,metric], coefficients_sme_mean[2,0,:,metric], coefficients_sme_mean[3,1,:,metric]]
        bad_comp = [coefficients_sme_mean[0,1,:,metric], coefficients_sme_mean[1,0,:,metric], coefficients_sme_mean[2,1,:,metric], coefficients_sme_mean[3,0,:,metric]]
        comparison_names = ['Ref: H, In: H','Ref: V, In: V','Ref: H, In: V','Ref: V, In: H']
        metrics = ['Difference', 'Inner product', 'Norm. Difference', 'Norm. Inner product']
        axs = axs.ravel()
        axs2 = axs2.ravel()
        for _ in range(4):
            axs[_].plot(shift_positions*1e6-150, good_comp[_], '-', label=f'{comparison_names[_]}, concentration {concentration}')
            axs[_].legend(loc='lower right')
            axs[_].set_title(comparison_names[_])
            axs[_].set_xlabel('x (um)')
            axs[_].set_ylabel(metrics[metric])
            axs2[_].plot(shift_positions*1e6-150, bad_comp[_], '-', label=f'{comparison_names[_]}, concentration {concentration}')
            axs2[_].legend(loc='lower right')
            axs2[_].set_title(comparison_names[_] + 'Cross talk')
            axs2[_].set_xlabel('x (um)')
            axs2[_].set_ylabel(metrics[metric])
    axs3 = axs3.ravel()
    for _ in range(4):
        axs3[_].plot(np.asarray(concentrations_list), np.asarray(pme_values_list)[:,_], '-', label=f'Metric: {metrics[_]}')
        axs3[_].set_title(f'PME - {metrics[_]} ')
        axs3[_].set_xlabel('Concentration')
        axs3[_].set_ylabel(metrics[metric])
    plt.show()
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')


def gaussian(x, amplitude, mean, stddev, v0):
    return amplitude * np.exp(-(x - mean)**2 / (2 * stddev**2)) + v0


def fit_data(x_data, y_data, pme_mean_value):
        try:
            # Fit the Gaussian curve to the data
            initial_guess = [1, 0, 0.5, 0.2]  # Initial guess for the parameters
            popt, pcov = curve_fit(gaussian, x_data, y_data, p0=initial_guess)
            y_fitted_2 = gaussian(x_data, *popt)
            corr_matrix = np.corrcoef(y_data, y_fitted_2)
            corr = corr_matrix[0, 1]
            R_sq = corr**2
            if R_sq <= 0.95 :
                logging.warning(f' R2 curve fit = {R_sq} ')
            x_fit = np.linspace(np.amin(x_data),np.amax(x_data), 10000)
            def find_nearest(array, value):
                array = np.asarray(array)
                idx = (np.abs(array - value)).argmin()
                return array[idx], idx
            y_fitted_hd = gaussian(x_fit,*popt)
            pme_nearest, idx = find_nearest(y_fitted_hd, pme_mean_value)
            x_shift = x_fit[idx]
            # x_shift = popt[1] + np.sqrt(2)*popt[2]*np.sqrt(2j*np.pi+np.log(-popt[0]/(popt[3]-pme_mean_value)))
            # plt.plot(x_data, y_fitted_2, label = 'Fitted data curve_fit')
        except:
            print('not possible')
            R_sq = 0
            x_shift = 0
            popt = np.asarray([1,1,1,1])
        # plt.plot(x_data, y_data,'.', label = 'Experimental data')
        # plt.plot(x_data, y_fitted, label = 'Fitted data polyfit')
        # plt.legend()
        # plt.show()
        return np.array([R_sq, np.abs(x_shift), *popt.tolist()])


def read_results_stage():
    input_directories = [r"C:\Users\tvettenburg\Downloads\coefficients_12-06-2023"]
    reference_paths = list(fd.askopenfilenames(initialdir=input_directories[0], title='Images to compare data:'))
    # reference_paths = [r"B:\memory_effect_raw_imgs\26-05-23\stage_precision_fringe_None_sample_1c_thick_0.5mm_step_0.05um2023-05-26_17-33-41.npz", r"B:\memory_effect_raw_imgs\26-05-23\stage_precision_fringe_None_sample_0.5c_thick_0.5mm_step_0.05um2023-05-26_16-52-33.npz"]
    mean_value=[]
    std_value = []
    metrics_name = ['difference', 'inner_product', 'normalized_difference', 'normalized_inner_product','h comp intensity', 'h comp intensity']
    metric = 2
    for r_p in reference_paths:
        concentration = r_p.split('sample_')[1].split('_thick')[0]
        data=np.load(r_p)
        cc = data['stage_precision_coefficients']

        mean_value.append(np.mean(cc, axis=(0,1,2))[metric])
        std_value.append(np.std(cc, axis=(0,1,2))[metric])
        print(f'Concentration: {concentration}, {metrics_name[metric]}: {mean_value[-1]} +- {std_value[-1]} ')
    print(f'All concentrations: {metrics_name[metric]}: {np.mean(np.asarray(mean_value))} +- {np.mean(np.asarray(std_value))}')



def read_results_raw_imgs_v2():
    input_directories = [r"C:\Users\tvettenburg\Downloads\coefficients_12-06-2023"]
    reference_paths = list(fd.askopenfilenames(initialdir=input_directories[0], title='Images to compare data:'))
    # reference_paths = [r"C:\Users\tvettenburg\Downloads\all coefficients 260523\coefficients_fringe_none_sample_1.5c_thick_0.5mm_step_0.05um2023-05-26_18-35-16.npz"]
    # fig, axs = plt.subplots(3, 1, sharex=True)
    # fig2, axs2 = plt.subplots(2, 2, sharex=True)
    # fig3, axs3 = plt.subplots(2, 2, sharex=True)
    concentrations_list = []
    pme_values_list = []
    # concentration_values = np.arange(0,2.25,0.25).tolist()
    concentration_values = np.arange(0,1,0.125).tolist()
    concentration_values.extend(np.arange(1,2.25,0.25).tolist())

    concentrations = []
    for _ in reference_paths:
        concentrations.append(float(_.split('sample_')[1].split('c_')[0]))

    idx_c = []
    for _ in concentration_values:
        idx_c.append([i for i,x in enumerate(concentrations) if x==_])

    dx_h_sme = []
    dx_v_sme = []
    dx_spme = []
    pme_values_c = []
    spme_values_c = []

    for c_number in range(len(concentration_values)):
        concentration = concentration_values[c_number]
        dx_h_sme_un = []
        dx_v_sme_un = []
        dx_spme_un = []
        pme_values_c_un = []
        spme_values_c_un = []
        for idx in idx_c[c_number]:
            reference_path = reference_paths[idx]
            print(f'Data {reference_path}')
            print(f'Concentration {concentration} c ')
            concentrations_list.append(float(concentration))
            data = np.load(reference_path)
            coefficients_pme = data['coefficients_pme']   # (5, 20, 1, 6) point, measuremets, hv, metrics
            benchmark_pme_coefficients = data['benchmark_pme_coefficients']  # (5, 20, 2, 1, 6) point, measuremets,0-hh; 1 vv, metrics
            list_of_comparisons_pme= data['list_of_comparisons_pme']

            pol_m_e_positions = data['pol_m_e_positions']
            times_shift = data['times_shift']
            times_pol_m_e = data['times_pol_m_e']
            pol_name = data['pol_name']
            shift_positions=data['shift_positions']

            mean_per_point_pme = []
            std_per_point_pme = []
            mean_per_point_pme_bench = []
            std_per_point_pme_bench = []

            for point in range(coefficients_pme.shape[0]):
                mean_per_point_pme.append(np.mean(coefficients_pme[point], axis=0))
                std_per_point_pme.append(np.std(coefficients_pme[point], axis=0))
                mean_per_point_pme_bench.append(np.mean(benchmark_pme_coefficients[point], axis=0))
                std_per_point_pme_bench.append(np.std(benchmark_pme_coefficients[point], axis=0))
            mean_per_point_pme = np.array(mean_per_point_pme)
            std_per_point_pme = np.array(std_per_point_pme)
            mean_per_point_pme_bench = np.array(mean_per_point_pme_bench)
            std_per_point_pme_bench = np.array(std_per_point_pme_bench)
            metrics_name = ['difference', 'inner_product', 'normalized_difference', 'normalized_inner_product','h comp intensity', 'h comp intensity']
            metric = 2
            pme_mean_value = np.mean(mean_per_point_pme, axis=0)[0, metric]

            coefficients_sme=data['coefficients_sme'] # (2, 4, 100, 6)
            coefficients_spme =data['coefficients_spme'] # (2, 4, 100, 6)
            coefficients_spme_benchmark =data['coefficients_sme_benchmark'] # (2, 2, 100, 6)

            # fig, axs = plt.subplots(3, 1, sharex=True)
            # axs = axs.ravel()
            sme_names = ['H1H1', 'V1V1', 'H2H2', 'V2V2']
            spme_names = ['H2V1', 'H1V2', 'V1H2', 'V2H1']
            bench_names = ['H1H2', 'V1V2']
            title_names = [f'Shift memory effect, {concentration}c','Shift polarization memory effect', 'Shift polarization memory effect benchmark' ]
            difraction_limit = 633e-9/(2*0.55) * 1e6

            fit_statistics_h_sme = []
            fit_statistics_v_sme = []
            fit_statistics_spme = []
            spme_mean_value = []
            for y_pos in range(2):
                for _ in range(4):
                    # axs[0].plot(shift_positions*1e6-150, coefficients_sme[y_pos,_,:,metric],'-', label=f'{sme_names[_]}, y_pos {y_pos}')
                    # axs[1].plot(shift_positions*1e6-150, coefficients_spme[y_pos,_,:,metric],'-', label=f'{spme_names[_]}, y_pos {y_pos}')
                    if _ in [0, 2]:
                        fit_statistics_h_sme.append(fit_data(shift_positions*1e6-150,coefficients_sme[y_pos,_,:,metric], pme_mean_value))

                    else:
                        fit_statistics_v_sme.append(fit_data(shift_positions*1e6-150,coefficients_sme[y_pos,_,:,metric], pme_mean_value))

            x_h_mean = [np.mean(np.asarray(fit_statistics_h_sme[0:2]),axis=0), np.mean(np.asarray(fit_statistics_h_sme[2:4]),axis=0)]
            x_v_mean = [np.mean(np.asarray(fit_statistics_v_sme[0:2]),axis=0), np.mean(np.asarray(fit_statistics_v_sme[2:4]),axis=0)]
            for y_pos in range(2):
                # print(f'Y pos {y_pos}: x shift h = {x_h_mean[y_pos][1]} um; x shift v = {x_v_mean[y_pos][1]} um')
                for _ in range(4):
                    fit_statistics_spme.append(fit_data(shift_positions*1e6-150,coefficients_spme[y_pos,_,:,metric], pme_mean_value))

                    ppot_spme = fit_statistics_spme[-1][2:6]      # fit data return  R_sq, x_shift, popt
                    spme_mean_value.append(gaussian((x_h_mean[y_pos][1]+x_v_mean[y_pos][1])/2, *ppot_spme))

            fit_statistics_h_sme=np.array(fit_statistics_h_sme)
            fit_statistics_v_sme =np.array(fit_statistics_v_sme)
            fit_statistics_spme =np.array(fit_statistics_spme)
            spme_mean_value = np.array(spme_mean_value)

            print(f'metric: {metrics_name[metric]}')

            print(f'PME mean:  {pme_mean_value:0.4f} +- {np.mean(std_per_point_pme, axis=0)[0,metric]:0.4f}; ')# point std  {np.std(mean_per_point_pme, axis=0)[0,metric]:0.4f}; intensity ratio(h/v) {np.mean(mean_per_point_pme, axis=0)[0,4]/np.mean(mean_per_point_pme, axis=0)[0,5]:0.4f}')
            # print(f'PME HH benchmarck mean:  {np.mean(mean_per_point_pme_bench, axis=0)[0,0,metric]:0.4f} +- {np.mean(std_per_point_pme_bench, axis=0)[0,0,metric]:0.4f};  point std  {np.std(mean_per_point_pme_bench, axis=0)[0,0,metric]:0.4f};')
            # print(f'PME VV benchmarck mean:  {np.mean(mean_per_point_pme_bench, axis=0)[1,0,metric]:0.4f} +- {np.mean(std_per_point_pme_bench, axis=0)[1,0,metric]:0.4f};  point std  {np.std(mean_per_point_pme_bench, axis=0)[1,0,metric]:0.4f};')
            print(f'H pol X shift mean: {np.mean(fit_statistics_h_sme, axis=0)[1]} +- {np.std(fit_statistics_h_sme, axis=0)[1]}')
            print(f'V pol X shift mean: {np.mean(fit_statistics_v_sme, axis=0)[1]} +- {np.std(fit_statistics_v_sme, axis=0)[1]}')
            print(f'SPME mean: {np.mean(spme_mean_value)} +- {np.std(spme_mean_value)}')
            print(f'SPME X shift mean: {np.mean(fit_statistics_spme, axis=0)[1]} +- {np.std(fit_statistics_spme, axis=0)[1]}')

            x_hd = np.linspace(np.amin(shift_positions*1e6-150),np.amax(shift_positions*1e6-150),1000)
            fit_h_sme=gaussian(x_hd,*np.mean(fit_statistics_h_sme, axis=0)[2:6])
            fit_v_sme=gaussian(x_hd,*np.mean(fit_statistics_v_sme, axis=0)[2:6])
            fit_spme=gaussian(x_hd,*np.mean(fit_statistics_spme, axis=0)[2:6])
            print(f'Std SME H: {np.mean(fit_statistics_h_sme, axis=0)[5]} ')
            print(f'Std SME V: {np.mean(fit_statistics_v_sme, axis=0)[5]} ')
            print(f'Std SPME: {np.mean(fit_statistics_spme, axis=0)[5]} ')

            sme_mean = np.mean(coefficients_sme, axis=0)
            h_sme_mean = np.mean(np.asarray([sme_mean[0],sme_mean[2]]),axis = 0)
            v_sme_mean = np.mean(np.asarray([sme_mean[1],sme_mean[3]]),axis = 0)
            spme_mean = np.mean(coefficients_spme, axis=(0,1))

            h_sme_std = np.std(np.asarray([sme_mean[0],sme_mean[2]]),axis = 0)
            v_sme_std = np.std(np.asarray([sme_mean[1],sme_mean[3]]),axis = 0)
            spme_std = np.std(coefficients_spme, axis=(0,1))

            R_sq_h_sme = np.corrcoef(h_sme_mean[:,metric], gaussian(shift_positions*1e6-150, *np.mean(fit_statistics_h_sme, axis=0)[2:6]))[0, 1]**2
            R_sq_v_sme = np.corrcoef(v_sme_mean[:,metric], gaussian(shift_positions*1e6-150, *np.mean(fit_statistics_v_sme, axis=0)[2:6]))[0, 1]**2
            R_sq_spme = np.corrcoef(spme_mean[:,metric], gaussian(shift_positions*1e6-150, *np.mean(fit_statistics_spme, axis=0)[2:6]))[0, 1]**2
            # fig2 = plt.figure()
            # plt.errorbar(shift_positions*1e6-150,h_sme_mean[:,metric],yerr=h_sme_std[:,metric],fmt='.b', label=f'Horizontal Shift Memory Effect')
            # plt.errorbar(shift_positions*1e6-150,v_sme_mean[:,metric],yerr=v_sme_std[:,metric],fmt='.r', label=f'Vertical Shift Memory Effect')
            # plt.errorbar(shift_positions*1e6-150,spme_mean[:,metric],yerr=spme_std[:,metric],fmt='.g', label=f'Shift Polarization Memory Effect',ecolor='lightgreen', elinewidth=1.5, capsize=0)
            # plt.errorbar(shift_positions*1e6-150,np.ones(100)*pme_mean_value,yerr=np.mean(std_per_point_pme, axis=0)[0,metric], fmt='-k', label=f'Polarization Memory Effect mean value',ecolor='lightgray', elinewidth=1, capsize=0)
            # plt.fill_between(shift_positions*1e6-150, pme_mean_value-np.mean(std_per_point_pme, axis=0)[0,metric],pme_mean_value+ np.mean(std_per_point_pme, axis=0)[0,metric], alpha=1,edgecolor= 'lightgray', facecolor='lightgray',
            #                  linewidth=1, linestyle='-', antialiased=True)
            # plt.plot(x_hd,fit_h_sme,'-b', label=f'Fit Horizontal SME, $R^2$ = {R_sq_h_sme:0.3f}',linewidth=2)
            # plt.plot(x_hd,fit_v_sme,'-r', label=f'Fit Vertical SME, $R^2$ = {R_sq_v_sme:0.3f}',linewidth=2)
            # plt.plot(x_hd,fit_spme,'-g', label=f'Fit SPME, $R^2$ = {R_sq_spme:0.3f}',linewidth=2)
            # plt.axvline(x=difraction_limit, color='y', label = 'Difraction limit')
            # plt.axvline(x=-difraction_limit, color='y')
            # plt.title(f'Concentration {concentration} c')
            # plt.xlabel('x (um)')
            # plt.ylabel(metrics_name[metric])
            # plt.legend(loc='upper right')
            # plt.show()
            dx_h_sme_un.append(np.mean(fit_statistics_h_sme, axis=0)[1])
            dx_v_sme_un.append(np.mean(fit_statistics_v_sme, axis=0)[1])
            # dx_spme_un.append()
            pme_values_c_un.append(pme_mean_value)
            spme_values_c_un.append(np.mean(spme_mean_value))

        dx_h_sme.append(np.mean(np.array(dx_h_sme_un)))
        dx_v_sme.append(np.mean(np.array(dx_v_sme_un)))
        dx_spme.append(np.mean(np.array(dx_spme_un)))
        pme_values_c.append(np.mean(np.array(pme_values_c_un)))
        spme_values_c.append(np.mean(np.array(spme_values_c_un)))

    # plt.plot(np.asarray(concentration_values), np.asarray(dx_h_sme),'*', label=f'Delta X horizontal pol')
    # plt.plot(concentration_values, np.asarray(dx_v_sme),'*', label=f'Delta X vertical pol')
    # plt.legend(loc='upper right')

    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    factor = difraction_limit
    plt.plot(np.asarray(concentration_values), np.asarray(pme_values_c)**2, '*', label=f'PME')
    plt.plot(np.asarray(concentration_values), np.asarray(spme_values_c)**2, '.k', label=f'SPME')
    plt.gca().set(xlabel='concentration', ylabel='$\epsilon^2$')
    # plt.plot(np.asarray(concentration_values), np.asarray(spme_values_c) / np.asarray(pme_values_c), '*', label=f'SPME/PME')
    # plt.gca().set(xlabel='concentration', ylabel='shift error relative error')
    plt.legend()

    plt.show()

# process_raw_imgs_v2()
# read_results_raw_imgs_v2()
# process_raw_imgs_v2_stage()
# read_results_raw_imgs()
read_results_stage()

# stage_precision()
# test_registration()
# calculate_coefficients()
# process_raw_imgs_one()
# show_results()
# process_raw_imgs()
# show_imgs()
# plot_m_effect_data()
# fit_data()