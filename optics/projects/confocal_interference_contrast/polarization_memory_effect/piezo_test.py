from optics.instruments.stage.nanostage import NanostageLT3
import time
#axis: 0 -> z
import numpy as np


nano = NanostageLT3(port='COM3')
nano.center_all()
print(nano.stage_range[0,0], nano.stage_range[0,1])

n=11
dx=1000 # in nm
position_x= np.arange(0, 0+dx*n, dx, dtype=int)
print(position_x)


control = ''
while control != 'exit':
    control = input('1-> relative move')
    if control == '1':
        #axis_selection = int(input('Axis selection (0,1,2)'))
        axis_selection = 1
        displacement = input('Displacement value in nm')
        if displacement[0] == '-':
            displacement = -1*int(displacement[1:])
        else:
            displacement = int(displacement)
        nano.move_rel(axis=axis_selection, value=displacement)
    if control == '2':
        for _ in range(100):
            displacement = 50e3 + _*2e3
            print(displacement)
            nano.move(axis=0, value=displacement)

            time.sleep(0.5)
