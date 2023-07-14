import tkinter as tk
from tkinter import ttk


__all__ = ['Element', 'SwitchButton', 'Combobox', 'OptionMenu', 'Canvas']


class Element:
    """
    Represents a GUI element.
    All custom widgets inherit from this class.
    """
    def __init__(self, widget: tk.Widget):
        self.__tk_widget: tk.Widget = widget
        # super().__init__(self.__tk_widget)

    def pack(self, *args, **kwargs):
        """
        Overrides the same method on tk.Widget, though now returning the widget so it can be daisy chained.
        :param args: arguments to pass.
        :param kwargs: keyword argumetns to pass.
        :return: this tk.Widget
        """
        self.__tk_widget.pack(*args, **kwargs)
        return self

    def grid(self, *args, **kwargs):
        """
        Overrides the same method on tk.Widget, though now returning the widget so it can be daisy chained.
        :param args: arguments to pass.
        :param kwargs: keyword argumetns to pass.
        :return: this tk.Widget:
        """
        self.__tk_widget.grid(*args, **kwargs)
        return self

    @property
    def title(self) -> str:
        return self.__tk_widget.winfo_toplevel().title()

    @title.setter
    def title(self, new_title: str):
        self.__tk_widget.winfo_toplevel().title(new_title)


class SwitchButton(Element):
    def __init__(self, master, text=('OFF', 'ON'), command=lambda s: None, **kwargs):
        s = ttk.Style()
        s.configure('Latched.TButton', relief='sunken', foreground='#ff0000')
        if not isinstance(text, tuple):
            text = (text, text)
        self.__text = text
        self.__command = command
        self.__ttk_button = ttk.Button(master=master.tk_container, text=self.__text[0], command=self.__toggle, **kwargs)
        super().__init__(self.__ttk_button)
        self.__switch_state = False

    @property
    def nb_states(self):
        return len(self.__text)

    @property
    def switch_state(self) -> int:
        return self.__switch_state

    @switch_state.setter
    def switch_state(self, value: int):
        self.__switch_state = value
        self.__ttk_button['text'] = self.__text[self.switch_state]
        if self.switch_state == 0:
            self.__ttk_button['style'] = 'TButton'
        else:
            self.__ttk_button['style'] = 'Latched.TButton'
        # Callback
        self.__command(self.switch_state)

    def __toggle(self):
        self.__iadd__(1)

    def __iadd__(self, number: int):
        self.switch_state = (self.switch_state + number) % self.nb_states

    def __isub(self, number: int):
        self.__iadd__(-number)


class Combobox(Element):
    def __init__(self, master, value=None, values=None, command=None, keep_history=True, **kwargs):
        if value is not None:
            if values is None:
                values = []
            values.insert(0, value)
        self.__ttk_combobox = ttk.Combobox(master.tk_container, values=values, **kwargs)
        self.__ttk_combobox.set(value)  # Do not execute the callback during the initialization!
        super().__init__(self.__ttk_combobox)

        self.__command = None
        self.__keep_history = None

        self.command = command
        self.keep_history = keep_history

    @property
    def value(self):
        return self.__ttk_combobox.get()

    @value.setter
    def value(self, new_value):
        self.__ttk_combobox.set(new_value)
        self.command(self)

    @property
    def command(self):
        return self.__command

    @command.setter
    def command(self, new_command=lambda w: True):
        self.__command = new_command
        events = ['<<ComboboxSelected>>', '<Return>', '<FocusOut>']
        if self.command is not None:
            for event in events:
                self.__ttk_combobox.bind(event, lambda e: self.__callback())
        else:
            for event in events:
                self.__ttk_combobox.unbind(event)

    @property
    def values(self) -> list:
        result = self.__ttk_combobox['values']
        if not isinstance(result, tuple):
            result = (result, )
        return list(result)

    @property
    def keep_history(self) -> bool:
        return self.__keep_history

    @keep_history.setter
    def keep_history(self, new_value: bool):
        self.__keep_history = new_value

    def __callback(self):
        value = self.value
        success = self.command(self) or self.command(self) is None
        if self.keep_history and success:
            values = self.values
            if len(values) > 10:
                values.pop()
            if value not in values:
                values.insert(0, value)

            self.__ttk_combobox['values'] = tuple(values)


class OptionMenu(Element):
    def __init__(self, master, value=None, values=tuple(), command=lambda w: True, width=None, **kwargs):
        self.__callback = command

        def __command(e):
            return self.__callback(self)

        self.__var = tk.StringVar()

        self.__tk_option_menu = ttk.OptionMenu(master.tk_container, self.__var, *values, command=__command, **kwargs)
        self.__tk_option_menu.configure(width=width)
        super().__init__(self.__tk_option_menu)

        self.__var.set(value)  # Do not execute the callback during the initialization!

    @property
    def value(self):
        return self.__var.get()

    @value.setter
    def value(self, new_value):
        self.__var.set(new_value)
        self.command(self)

    @property
    def command(self):
        return self.__callback

    @command.setter
    def command(self, new_command=lambda w: True):
        self.__callback = new_command


class Canvas(Element):
    def __init__(self, master, height: int, width: int):
        self.__canvas = tk.Canvas(master.tk_container, height=height, width=width,
                                  highlightbackground="#ff0000", highlightthickness=0, cursor='none')
        super().__init__(self.__canvas)

    def create_image(self, *args, **kwargs):
        self.__canvas.create_image(*args, **kwargs)
