import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog as fd

x = np.linspace(0, 10, 11)
X = np.linspace(0,10,100)

linear_0c_amp = np.array([90.1, 86.5, 90.7, 89.1, 87.6])
linear_0c_angle = np.array([-9.4, -8.5, -8.1, -6.0, -3.9])
circular_0c_amp = np.array([93.8, 95.7, 92.1, 92.4, 94.2])
circular_0c_angle = np.array([-0.8, -5.5, -2.5, -3.2, -1.9])

linear_1c_amp = np.array([71.3, 70.4, 69.8, 71.1, 69.6])
linear_1c_angle = np.array([2.1, 1.4, 6.7, -1.6, -1.4])
circular_1c_amp = np.array([72.2, 72.8, 71.5, 72.6, 72.5])
circular_1c_angle = np.array([3.8, 1, -0.7, 5.4, 5.7])

linear_2c_amp = np.array([63.5, 61.1, 61.8, 63.4, 62.2])
linear_2c_angle = np.array([6.4, 2.7, 6.9, 9, 13.8])
circular_2c_amp = np.array([65.5, 68.1, 63.7, 65.7, 63.3])
circular_2c_angle = np.array([10.2, 6.3, 3.7, 12.7, 10.5])

linear_3c_amp = np.array([55.1, 53.2, 53.1, 54.2, 53.6])
linear_3c_angle = np.array([6.9, 6.2, 6.6, 9, 8.6])
circular_3c_amp = np.array([58.4, 55.4, 59, 57.9, 57.1])
circular_3c_angle = np.array([3.1, 7.5, 9, 8.7, 10])

linear_4c_amp = np.array([44.6, 42.8, 47.3, 43.3, 45.3])
linear_4c_angle = np.array([20.9, 16.9, 21.9, 22.8, 16.3])
circular_4c_amp = np.array([46.5, 47.7, 48.7, 47.1, 47.9])
circular_4c_angle = np.array([16.9, 11.9, 21.8, 24, 16.9])

linear_5c_amp = np.array([30.6, 35.6, 35, 36, 35.4])
linear_5c_angle = np.array([10.3, 6.4, 15.4, 11.6, 11.3])
circular_5c_amp = np.array([34.8, 38.5, 35.7, 36.7, 37.8])
circular_5c_angle = np.array([12.8, 8.8, 14.6, 10.9, 9.3])

linear_6c_amp = np.array([38.5, 41, 41.1, 41.8, 39.8])
linear_6c_angle = np.array([14.8, 15.2, 17.1, 15.8, 20.2])
circular_6c_amp = np.array([43.4, 44.4, 42.1, 42.8, 39.3])
circular_6c_angle = np.array([16.4, 17.3, 15.1, 12.9, 12.6])

linear_7c_amp = np.array([31.8, 29.7, 30.3, 30.3, 29.3])
linear_7c_angle = np.array([11.4, 23.6, 19.6, 27, 18.8])
circular_7c_amp = np.array([33.9, 30.2, 30.9, 34.5, 31.2])
circular_7c_angle = np.array([15.7, 17.5, 17, 6.8, 13.7])

linear_8c_amp = np.array([31.4, 33.9, 30.8, 34.0, 30])
linear_8c_angle = np.array([19.4, 15.2, 16.1, 16.8, 19.8])
circular_8c_amp = np.array([32.8, 35.5, 30.7, 35.4, 33.3])
circular_8c_angle = np.array([16.3, 14.8, 9.5, 17.4, 15])

linear_9c_amp = np.array([28.6, 26.7, 25.8, 29.6, 23.6])
linear_9c_angle = np.array([20.6, 19, 15.3, 26.2, 19.8])
circular_9c_amp = np.array([26.8, 29.3, 28.4, 32.1, 26])
circular_9c_angle = np.array([17, 24.9, 17.3, 22.1, 21.5])

linear_10c_amp = np.array([28.6, 23, 25.7, 25.2, 25.8])
linear_10c_angle = np.array([23.6, 17.1, 20.8, 15.2, 19.7])
circular_10c_amp = np.array([30.6, 25, 29.3, 26.7, 28.2])
circular_10c_angle = np.array([19.6, 17.5, 20.9, 13.4, 17.6])

linear_mean_amp = np.mean(np.array([linear_0c_amp, linear_1c_amp, linear_2c_amp, linear_3c_amp, linear_4c_amp, linear_5c_amp, linear_6c_amp, linear_7c_amp, linear_8c_amp, linear_9c_amp, linear_10c_amp]), axis=1)
linear_std_amp = np.std(np.array([linear_0c_amp, linear_1c_amp, linear_2c_amp, linear_3c_amp, linear_4c_amp, linear_5c_amp, linear_6c_amp, linear_7c_amp, linear_8c_amp, linear_9c_amp, linear_10c_amp]), axis=1)

linear_mean_angle = np.mean(np.array([linear_0c_angle, linear_1c_angle, linear_2c_angle, linear_3c_angle, linear_4c_angle, linear_5c_angle, linear_6c_angle, linear_7c_angle, linear_8c_angle, linear_9c_angle, linear_10c_angle]), axis=1)
linear_std_angle = np.std(np.array([linear_0c_angle, linear_1c_angle, linear_2c_angle, linear_3c_angle, linear_4c_angle, linear_5c_angle, linear_6c_angle, linear_7c_angle, linear_8c_angle, linear_9c_angle, linear_10c_angle]), axis=1)

circular_mean_amp = np.mean(np.array([circular_0c_amp, circular_1c_amp, circular_2c_amp, circular_3c_amp, circular_4c_amp, circular_5c_amp, circular_6c_amp, circular_7c_amp, circular_8c_amp, circular_9c_amp, circular_10c_amp]), axis=1)
circular_std_amp = np.std(np.array([circular_0c_amp, circular_1c_amp, circular_2c_amp, circular_3c_amp, circular_4c_amp, circular_5c_amp, circular_6c_amp, circular_7c_amp, circular_8c_amp, circular_9c_amp, circular_10c_amp]), axis=1)

circular_mean_angle = np.mean(np.array([circular_0c_angle, circular_1c_angle, circular_2c_angle, circular_3c_angle, circular_4c_angle, circular_5c_angle, circular_6c_angle, circular_7c_angle, circular_8c_angle, circular_9c_angle, circular_10c_angle]), axis=1)
circular_std_angle = np.std(np.array([circular_0c_angle, circular_1c_angle, circular_2c_angle, circular_3c_angle, circular_4c_angle, circular_5c_angle, circular_6c_angle, circular_7c_angle, circular_8c_angle, circular_9c_angle, circular_10c_angle]), axis=1)


input_directories = [r'C:\Users\lab\OneDrive - University of Dundee\Documents\lab\code\python\optics\projects\confocal_interference_contrast\polarization_memory_effect\results\06-09-2022']
paths = fd.askopenfilenames(initialdir=input_directories[0])
number_path = len(paths)
keys = ['linear_mean_amp', 'linear_std_amp', 'circular_mean_amp', 'circular_std_amp']
results = np.zeros((number_path+1, 4, len(np.load(paths[0])[keys[0]])))
curves_fit = np.zeros((number_path+1, 4, 100))
results[0, 0, :] = linear_mean_amp
results[0, 1, :] = linear_std_amp
results[0, 2, :] = circular_mean_amp
results[0, 3, :] = circular_std_amp
titles = ['Monday']
for data in range(number_path):
    ref_name = paths[data].rsplit('_')[-1].split('.')[0]
    titles.append(ref_name)
    for _ in range(4):
        results[data+1, _, :] = np.load(paths[data])[keys[_]]

for _ in range (number_path+1):
    # results[_,0] = np.log(results[_,0])
    # results[_,2] = np.log(results[_,2])
    pl = np.polyfit(x, np.log(results[_,0]),1)
    pc = np.polyfit(x, np.log(results[_,2]),1)
    curves_fit[_, 0] = np.exp(pl[1]) * np.exp(pl[0]*X)
    curves_fit[_, 2] = np.exp(pc[1]) * np.exp(pc[0]*X)



#display
#titles = ['5626', '5706', '5751', 'Monday']
colors = ['r', 'k', 'g', 'b', 'c']
fig = plt.figure()
count = 0
for _ in range(1, number_path+1):
    plt.plot(x, results[_, 0],  '^' + colors[count], label='Linear ' + titles[_])
    plt.plot(x, results[_, 2], 'o' + colors[count], label='Circular ' + titles[_])
    plt.plot(X, curves_fit[_, 0],  '--' + colors[count], label='Fit Linear ' + titles[_])
    plt.plot(X, curves_fit[_, 2], ':' + colors[count], label='Fit Circular ' + titles[_])
    count += 1

plt.legend()
plt.show()
