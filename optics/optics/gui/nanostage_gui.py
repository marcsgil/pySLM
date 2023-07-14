import tkinter as tk
from tkinter import messagebox
import re

from optics.instruments.stage.nanostage import NanostageLT3


class ScanSettingsWindow(tk.Toplevel):
    """
    Triggers a Popup window, where scan settings can be selected.
    """
    def __init__(self, root, *args, **kwargs):
        tk.Toplevel.__init__(self, root, *args, **kwargs)
        self.root = root
        self.scan_type = self.root.selected_scan
        self.title('Scan Settings')
        self.container = tk.Frame(self, relief=tk.SUNKEN)

        self.__proceed_buttons()
        self.__scan_settings_panel()
        self.container.pack()

    def __scan_settings_panel(self):
        """
        :return:
        """
        # TODO: Create a Database to keep hold of the previous entry values.
        scan_settings_panel = tk.Frame(self.container)

        # TODO: Make the nb_scan_points and scan step into a combo menu
        # Initialising Labels and Entries for the panel
        self.elements = {
            tk.Label(scan_settings_panel, text='Number of scan points per axis: '):
                tk.Entry(scan_settings_panel),
            tk.Label(scan_settings_panel, text='Scan step: '):
                tk.Entry(scan_settings_panel),
            tk.Label(scan_settings_panel, text='Scan axes: '):
                tk.Entry(scan_settings_panel),
            tk.Label(scan_settings_panel, text='Sleep time between scan points: '):
                tk.Entry(scan_settings_panel)
        }

        # Putting the Labels and Entries to the grid
        for idx, element in enumerate(self.elements):
            element.grid(row=idx, column=0, sticky='nse')
            self.elements[element].grid(row=idx, column=1)

        scan_settings_panel.grid(row=0, column=0, columnspan=2)

    def __proceed_buttons(self):
        """
        This is where Scan-Cancel buttons are created, that tells the app how to proceed.
        :return: None
        """
        proceed_buttons = [
            tk.Button(self.container, text='Scan', command=self.__initiate_scan),
            tk.Button(self.container, text='Cancel', command=self.destroy)
        ]

        for idx, button in enumerate(proceed_buttons):
            button.grid(row=10, column=idx, padx=5, pady=5)

        return None

    def __initiate_scan(self):
        """
        Initiates the selected scan type on the nanostage after clicking
        :return:
        """
        # TODO: fix class hierarchy to reference nanostage easier
        # TODO: finish issuing the command
        if isinstance(self.root.root.root.nanostage, NanostageLT3):
            pass


class ButtonPanel(tk.Frame):
    """
    Button panel to store function buttons. Ideally placed to the left of the control panel
    """
    def __init__(self, root, *args, **kwargs):
        tk.Frame.__init__(self, root, *args, **kwargs)
        self.root = root

        self.container = tk.LabelFrame(self, text='Functions')
        self.__panel_formatting()

        self.buttons = {
                        'Boustrophedon Scan':
                        tk.Button(self.container, text='Boustrophedon Scan', command=self.__boustrophedon),
                        'Raster Scan':
                        tk.Button(self.container, text='Raster Scan', command=self.__raster),
                        'Exact Position':
                        tk.Button(self.container, text='Exact Position', command=self.__position),
                        'Center':
                        tk.Button(self.container, text='Center', command=self.__center)
                        }
        # Putting the buttons on the grid
        self.__load_buttons()
        self.container.pack(padx=5)

    def __load_buttons(self):
        for idx, text in enumerate(self.buttons):
            self.buttons[text].grid(row=idx, column=0, sticky='nsew', padx=5, pady=1)

    def __panel_formatting(self):
        for idx in range(5):
            tk.Label(self.container, text='').grid(row=idx, column=0, sticky='nsew')

    def __boustrophedon(self):
        """
        Creates a popup window, from which it is possible to start a boustrophedon scan
        :return:
        """
        self.selected_scan = 'Boustrophedon Scan'
        window = ScanSettingsWindow(self)

    def __raster(self):
        """
        Creates a popup window, from which it is possible to start a raster scan
        :return:
        """
        self.selected_scan = 'Raster Scan'
        window = ScanSettingsWindow(self)

    def __position(self):
        """
        Displays the position of the nanostage (as given by the controller) in a popup window.
        :return:
        """
        # Extracting the position information from the controller
        if isinstance(self.root.root.nanostage, NanostageLT3):
            exact_pos = self.root.root.nanostage.true_pos()
        else:
            exact_pos = {'Error': 'The nanostage is not connected!'}

        # Formatting the information for display
        popup_text = 'Exact nanostage position given by the controller:\n\n'
        for axis in exact_pos:
            popup_text += f'{axis}: {exact_pos[axis]}\n'

        messagebox.showinfo('Piezoconcept LT3 Position Information', popup_text)

    def __center(self):
        """
        Centers the stage with respect to all axes.
        :return:
        """
        if isinstance(self.root.root.nanostage, NanostageLT3):
            self.root.root.nanostage.center_all()
            self.root.adj_position_display()


class ControlPanelSlide(tk.Frame):
    """
    Control Panel, which allows the user to control the stage position with slides.
    Mainly useful for configuring the stage.
    """
    def __init__(self, root, *args, **kwargs):
        tk.Frame.__init__(self, root, *args, **kwargs)
        self.root = root

        # Creating the control sliders and labels
        self.sliders = {}
        self.s_labels = {}
        for idx in range(3):
            self.s_labels[idx] = tk.Label(self, text='{} position is :'.format('XYZ'[idx]))
            self.sliders[idx] = tk.Scale(self, from_=0, to=300, orient=tk.HORIZONTAL,
                                         command=(lambda arg: (lambda val: self.on_scroll(arg)))(idx))  # This nested lambda is required due to looping

        # Putting the slider and labels on the grid
        for idx in self.sliders:
            self.s_labels[idx].grid(row=idx, column=0, sticky='ne')
            self.sliders[idx].grid(row=idx, column=1, sticky='nsew')

        # Additional function buttons
        self.button_panel = ButtonPanel(self)
        self.button_panel.grid(row=0, column=2, rowspan=10, sticky='nsew')

    def on_scroll(self, idx):
        """
        Command, which is given when the slides are moved in the control panel.
        :param idx: axis of movement, given as an integer in the order of XYZ
        :return:
        """
        if isinstance(self.root.nanostage, NanostageLT3):
            self.root.nanostage.move(axis=idx, value=self.sliders[idx].get(), unit='um')


    def adj_position_display(self):
        """
        Adjusts slider position to agree with the ones given by NanostageLT3 class.
        :return:
        """
        if isinstance(self.root.nanostage, NanostageLT3):
            for axis in range(3):
                self.sliders[axis].set(int(self.root.nanostage.position[axis]) * 1E-3)


class ControlPanelText(tk.Frame):
    """
    Control Panel, which allows the user to control the stage via text input.
    Useful, when there is a desire to move the stage to some specific position.
    """
    def __init__(self, root, *args, **kwargs):
        tk.Frame.__init__(self, root, *args, **kwargs)
        self.root = root

        # Creating Labels and Text boxes
        self.labels = {}
        self.text_boxes = {}
        for idx in range(3):
            self.labels[idx] = tk.Label(self, text='{} position is: '.format('XYZ'[idx]))
            self.text_boxes[idx] = tk.Entry(self)

            self.labels[idx].grid(row=idx, column=0, pady=5)
            self.text_boxes[idx].grid(row=idx, column=1, pady=5)

        # Submit Button for new coordinates
        self.submit_btn = tk.Button(self, text='Submit', command=self.on_click_submit)
        self.submit_btn.grid(row=len(self.text_boxes) + 1, column=0,
                             columnspan=2, sticky='nsew')

        self.button_panel = ButtonPanel(self)
        self.button_panel.grid(row=0, column=2, rowspan=10, sticky='nsew')

    def on_click_submit(self):
        """
        Commands the nanostage to move to the position specified in the text input of this panel.
        This function should be used with self.submit_btn button.
        :return:
        """
        if isinstance(self.root.nanostage, NanostageLT3):
            # Using RegEx to separate numbers and units from the string
            pattern_numbers = re.compile(r'^[0-9e.]{1,7}')
            pattern_units = re.compile(r'[nu]m?')
            for axis in self.text_boxes:
                text_input = self.text_boxes[axis].get()
                numbers = float(pattern_numbers.search(text_input).group())
                units = pattern_units.search(text_input).group()
                self.root.nanostage.move(axis=axis, value=numbers, unit=units)

            self.adj_position_display()  # TODO: make the app display in the same units that already exist in a text box

    def adj_position_display(self):
        """
        Adjusts position text values to agree with the ones given by NanostageLT3 class.
        :return:
        """
        if isinstance(self.root.nanostage, NanostageLT3):
            for axis in self.text_boxes:
                self.text_boxes[axis].delete(0, 'end')
                self.text_boxes[axis].insert(0, f'{self.root.nanostage.position[axis] / 1E3} um')


class NanostageGUI(tk.Frame):
    """
    Root class for the nanostage LT3 GUI app.
    """
    def __init__(self, root, *args, **kwargs):
        tk.Frame.__init__(self, root, *args, **kwargs)
        self.root = root
        self.root.title('Piezoconcept Nanostage LT3')
        try:
            self.nanostage = NanostageLT3()
        except Exception:
            # print('The nanostage is not connected...')
            messagebox.showwarning('Warning', 'The nanostage is not connected to the device.')
            self.nanostage = None

        self.__init_menu()
        self.__init_database()

        # Adding pre-made widgets
        self.frames = {ControlPanelText: ControlPanelText(self),
                       ControlPanelSlide: ControlPanelSlide(self)}

        # Initialising pre-made widgets
        for frame in self.frames:
            self.frames[frame].grid(row=0, column=0, sticky='nsew')

        self.show_frame(ControlPanelSlide)

    def show_frame(self, content):
        """
        Allows the user to switch between display configurations
        :param content: frame name, as defined in self.frames
        :return:
        """
        frame = self.frames[content]
        self.frames[content].adj_position_display()
        frame.tkraise()

    def __init_menu(self):
        """
        Formats the top menu bar for the nanostage app.
        :return:
        """

        menu_bar = tk.Menu(self.root)
        self.root.config(menu=menu_bar)

        file = tk.Menu(menu_bar, tearoff=0)
        file.add_separator()
        file.add_command(label='Exit', command=self.quit)
        menu_bar.add_cascade(label='File', menu=file)

        settings = tk.Menu(menu_bar, tearoff=0)
        ctr_mode = tk.Menu(settings, tearoff=0)
        ctr_mode.add_command(label='Slide Control', command=lambda: self.show_frame(ControlPanelSlide))
        ctr_mode.add_command(label='Text Control', command=lambda: self.show_frame(ControlPanelText))
        settings.add_cascade(label='Control Mode', menu=ctr_mode)
        menu_bar.add_cascade(label='Settings', menu=settings)

    def __init_database(self):
        # TODO: create a database for settings memory if it doesn't exist
        pass


def main():
    root = tk.Tk()
    # root.geometry('400x400')
    NanostageGUI(root).pack(side="top", fill="both", expand=True, padx=10, pady=10)
    root.mainloop()


if __name__ == '__main__':
    main()
