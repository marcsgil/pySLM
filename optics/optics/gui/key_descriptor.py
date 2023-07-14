import logging

import numpy as np

log = logging.getLogger(__name__)


class KeyDescriptor:
    def __init__(self, event):
        self.__key_code = event.keycode
        self.__key_name = event.keysym
        self.__modifier = event.state
        self.__mouse_position = np.array((event.x, event.y))
        log.debug(event)

    @property
    def key_code(self):
        return self.__key_code

    @property
    def key_name(self):
        return self.__key_name

    @property
    def shift(self):
        return 0x0001 & self.__modifier

    @property
    def caps_lock(self):
        return 0x0002 & self.__modifier

    @property
    def control(self):
        return 0x0004 & self.__modifier

    @property
    def left_alt(self):
        return 0x0008 & self.__modifier

    @property
    def num_lock(self):
        return 0x0010 & self.__modifier

    @property
    def right_alt(self):
        return 0x0080 & self.__modifier

    @property
    def alt(self):
        return self.left_alt or self.right_alt

    @property
    def mouse_button_1(self):
        return 0x00100 & self.__modifier

    @property
    def mouse_button_2(self):
        return 0x00200 & self.__modifier

    @property
    def mouse_button_3(self):
        return 0x00400 & self.__modifier

    @property
    def mouse_button(self):
        return self.mouse_button_1 or self.mouse_button_2 or self.mouse_button_3

    @property
    def mouse_position(self):
        return self.__mouse_position

    def __str__(self):
        desc = ""
        if self.control:
            desc += "CTRL "
        if self.left_alt:
            desc += "LEFT-ALT "
        if self.right_alt:
            desc += "RIGHT-ALT "
        if self.shift:
            desc += "SHIFT "
        desc += self.key_name.upper()
        if self.caps_lock:
            desc += " CAPS-LOCK"
        if self.num_lock:
            desc += " NUM-LOCK"

        return desc

    def __eq__(self, other):
        """
        If other is another KeyDescriptor, an in-depth comparison will be done.
        If other is a string, a case-sensitive comparison will be done on the key_name.
        :param other: another KeyDescriptor or a string such as 'a', 'Enter', ...
        :return: True if the same.
        """
        if isinstance(other, KeyDescriptor):
            return self.key_code == other.key_code \
                   and self.control == other.control \
                   and self.left_alt == other.left_alt \
                   and self.right_alt == other.right_alt \
                   and self.shift == other.shift \
                   and self.caps_lock == other.caps_lock \
                   and self.num_lock == other.num_lock \
                   and self.mouse_button_1 == other.mouse_button_1 \
                   and self.mouse_button_2 == other.mouse_button_2 \
                   and self.mouse_button_3 == other.mouse_button_3
        else:
            return self.key_name.upper() == other.upper()

    def __neq__(self, other):
        return not self.__eq__(other)

