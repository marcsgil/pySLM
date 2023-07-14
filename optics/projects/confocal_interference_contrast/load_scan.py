import numpy as np
import matplotlib.pyplot as plt

data_path = r'C:\Users\braia\IdeaProjects\lab\results\scans\DPC_scan_30x30_2023-01-23_17-45-10.npz'
data = np.load(data_path)
camera_data = data['camera_data']
complex_amplitude = data['complex_amplitude']
plt.imshow(np.angle(complex_amplitude))
plt.colorbar(label="Phase (rad)", orientation="vertical")
plt.clim(-np.pi,np.pi)
plt.show()

