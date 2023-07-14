import tkinter as tk

from optics import log
from optics.gui.container import Window
from optics.gui import KeyDescriptor


if __name__ == '__main__':
    win1 = Window(title='Window 1')
    win2 = Window(title='Window 2')
    def key_handler(key_desc: KeyDescriptor):
        if key_desc.control and key_desc == 'O':
            win1.maximized = not win1.maximized
    win1.on_key += key_handler
    win1.on_change += lambda _: log.info(_)

    # The menu bar
    menu_bar = tk.Menu(win1.tk_container)
    # #
    menu_file = tk.Menu(menu_bar, tearoff=False)
    menu_cam = tk.Menu(menu_bar)

    for cam_name in ['webcam1', 'idscam']:
        menu_cam.add_command(label=cam_name, command=lambda: log.info(cam_name))

    # The final File menus
    menu_file.add_separator()
    menu_file.add_command(label="Close", command=win1.close, accelerator='Alt-F4')
    menu_file.add_command(label="Quit", command=Window.close_all, accelerator='Ctrl-Q')

    menu_bar.add_cascade(label='File', underline=0, menu=menu_file)
    menu_bar.add_cascade(label='Cam', underline=0, menu=menu_cam)
    win1.menu = menu_bar

    Window.run()