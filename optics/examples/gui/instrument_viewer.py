import matplotlib.pyplot as plt
import time
import numpy as np
import msgpack_numpy as m
m.patch()

import Pyro5.api
import Pyro5.errors
# Pyro5.config.COMPRESSIrON = False
# Pyro5.config.SERIALIZER = 'msgpack'

from optics.experimental.parallel_instruments import log

from optics.experimental.parallel_instruments.nameserver_proxy_provider import NameServerProxyProvider

from optics.instruments.cam.web_cam import WebCam

if __name__ == '__main__':
    fps_time_constant = 0.1   # FTP averaging time constant

    with NameServerProxyProvider() as nspp:
        with Pyro5.api.Proxy('PYRONAME:experimental.camera') as cam:
            ax_img = None
            avg_time = 1/5
            prev_time = time.perf_counter()
            for _ in range(10 * 10):
                img_data = cam.acquire()
                if ax_img is None:
                    fig, ax = plt.subplots(1, 1)
                    ax_img = ax.imshow(img_data)
                else:
                    ax_img.set_data(img_data)
                plt.show(block=False)
                plt.pause(1e-6)

                # Display FPS
                current_time = time.perf_counter()
                time_diff = current_time - prev_time
                prev_time = current_time
                avg_time = (1 - fps_time_constant) * avg_time + fps_time_constant * time_diff
                if _ % 30 == 0:
                    log.info(f'{1 / avg_time:0.1f} frames per second.')
            log.info(f'{1 / avg_time:0.1f} frames per second.')

