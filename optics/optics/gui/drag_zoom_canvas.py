import tkinter as tk
from typing import Union, Callable
import logging

import numpy as np
from PIL import Image, ImageTk

from optics.utils import Roi
from optics.utils import round125
from optics.utils import unit_registry
from optics.utils.ft import Grid
from optics.gui.container import Container
from optics.gui.io import Element

log = logging.getLogger(__name__)


class DragZoomCanvas(Element, Container):
    def __init__(self, container,
                 get_roi_callback: Union[Callable[[], Roi], None] = None,
                 set_roi_callback: Union[Callable[[Roi], None]] = None,
                 get_pixel_unit_callback: Callable[[], tuple] = None):
        """
        A GUI element that displays an interactive, zoomable, image.

        :param container: A gui.Container element.
        :param get_roi_callback: A callback that returns the current Roi of the viewport.
        :param set_roi_callback: A callback to set the new Roi of the viewport.
        :param get_pixel_unit_callback: A callback that returns the physical size represented by a pixel as a tuple
        (unit_value: float, unit_description: str). The latter being a description of the unit of the physical size.
        """
        if not isinstance(container, Container):  # for backwards compatibility
            container = Container(container)
        tk_canvas = tk.Canvas(container.tk_container, bg='#000000', bd=0, highlightthickness=0, relief='ridge', cursor='tcross')
        Element.__init__(self, tk_canvas)
        Container.__init__(self, tk_canvas)
        # cursors: dot, dotbox, tcross, cross, cross_reverse, none
        # self.tk_container.pack(fill=tk.BOTH, side=tk.TOP, expand=True)

        self.__raw_image_shape_full = (1, 1)
        self.__raw_image_shape_in_roi = None

        # The default callbacks:
        self.__roi: Roi = None
        if set_roi_callback is None:
            def set_roi_callback(new_roi: Roi):
                if new_roi is not None:  # else full roi
                    self.__roi = new_roi.astype(np.int)
                else:
                    self.__roi = Roi(top_left=(0, 0), shape=self.__raw_image_shape_full)
        if get_roi_callback is None:
            def get_roi_callback() -> Roi:
                if self.__roi is not None:
                    return self.__roi
                else:
                    return Roi(top_left=(0, 0), shape=self.__raw_image_shape_full)

        self.__get_roi_callback = get_roi_callback
        self.__set_roi_callback = set_roi_callback
        self.__get_pixel_unit_callback = get_pixel_unit_callback

        self.__raw_image_shape_in_roi = None
        self.__scalebar = None
        self.__scalebar_text = None
        self.__position_text = None
        self.__current_position_px = None
        self.__canvas_tk_image_index = None
        self.__zoom_rectangle = None
        
        self.__canvas_pil_image = None
        self.__canvas_tk_image = None

        self.__zoom = None
        self.__drag_start_pos = None

        # Bind events to Canvas object so we can zoom and drag it
        self.tk_container.bind('<Motion>', self.report_position)
        self.tk_container.bind('<Leave>', lambda x: self.report_position(None))
        self.tk_container.bind('<Double-1>', self.reset_drag_and_zoom)
        self.tk_container.bind('<ButtonPress-1>', self.__on_start_drag)
        self.tk_container.bind('<B1-Motion>', self.__on_drag)
        self.tk_container.bind('<ButtonRelease-1>', self.__on_stop_drag)
        self.tk_container.bind('<Double-3>', self.reset_drag_and_zoom)
        self.tk_container.bind('<ButtonPress-3>', self.__on_start_zoom)
        self.tk_container.bind('<B3-Motion>', self.__on_zoom)
        self.tk_container.bind('<ButtonRelease-3>', self.__on_stop_zoom)
        self.tk_container.bind("<MouseWheel>", self.__on_mousewheel)
        self.tk_container.bind("<Button-4>", self.__on_mousewheel)
        self.tk_container.bind("<Button-5>", self.__on_mousewheel)

        self.fit_to_window()

    @property
    def shape(self):
        # This requires self.tk_container.update() to be called first
        return np.array([self.tk_container.winfo_height(), self.tk_container.winfo_width()])

    @property
    def image(self) -> np.ndarray:
        """
        Returns the image as displayed on the canvas, this may be a different size from what was set.
        """
        return np.array(self.__canvas_pil_image)

    @image.setter
    def image(self, image_array: np.ndarray):
        self.__raw_image_shape_full = np.array(image_array.shape[:2])
        if self.__roi is not None:
            # crop and extent
            ranges = Grid(first=self.__roi.top_left, shape=self.__roi.shape)
            ranges = [np.clip(r, 0, s - 1) for r, s in zip(ranges, image_array.shape)]
            image_array = image_array[ranges[0], ranges[1]]
        if image_array.dtype == np.float:
            image_array = np.asarray(np.clip(image_array, 0, 1) * 255 + 0.5, dtype=np.uint8)

        self.__raw_image_shape_in_roi = np.array(image_array.shape[:2])

        # if image_array.ndim < 3:
        #     image_array = np.repeat(image_array[:, :, np.newaxis], repeats=3, axis=2)
        self.__canvas_pil_image = Image.fromarray(image_array)  # Update image

        canvas_image_current_shape = np.array([self.__canvas_tk_image.height(), self.__canvas_tk_image.width()],
                                              dtype=np.int)

        if np.any(self.__raw_image_shape_in_roi == 0):
            self.__raw_image_shape_in_roi = self.shape.astype(np.float)
        canvas_image_shape = self.__raw_image_shape_in_roi * np.min(self.shape.astype(np.float) / self.__raw_image_shape_in_roi)
        canvas_image_shape = canvas_image_shape.astype(np.int)
        if np.all(canvas_image_shape > 0):
            self.__canvas_pil_image = self.__canvas_pil_image.resize(size=canvas_image_shape[::-1],
                                                                     resample=Image.NEAREST)  # scale to fit display area
        if np.all(canvas_image_shape == canvas_image_current_shape):
            self.__canvas_tk_image.paste(self.__canvas_pil_image)
        else:
            # Create new Tk PhotoImage for the new resolution
            self.__canvas_tk_image = ImageTk.PhotoImage(self.__canvas_pil_image)
            self.tk_container.delete(self.__canvas_tk_image_index)
            self.__canvas_tk_image_index = self.tk_container.create_image(
                self.tk_container.winfo_width() / 2.0, self.tk_container.winfo_height() / 2.0, image=self.__canvas_tk_image,
                anchor=tk.CENTER, state=tk.NORMAL)

        self.update_legend()

    def update_legend(self):
        if self.__raw_image_shape_in_roi is not None:
            canvas_image_shape = self.__raw_image_shape_in_roi * np.min(self.shape.astype(np.float) / self.__raw_image_shape_in_roi)
            # Determine what scale the scalebar length should be
            display_scale = canvas_image_shape / self.__raw_image_shape_in_roi
            if self.__get_pixel_unit_callback is not None:
                pixel_unit_value, pixel_unit_description, origin_pixel = self.__get_pixel_unit_callback()
                representative_width = round125(self.__raw_image_shape_in_roi[1] * pixel_unit_value / 8)  # approximately 1/8 of the image but rounded
                scalebar_shape = (representative_width / pixel_unit_value) * np.array((1.0, 1 / 5)) * display_scale  # horizontal, vertical
                margin_width = 5
                # Display the scale bar
                if self.__scalebar is None:
                    self.__scalebar = self.tk_container.create_rectangle(0, 0, 1, 1,
                                                                     fill='#ffffff', outline='#000000', tags='legend')
                    self.__scalebar_text = self.tk_container.create_text(0, 0, text='',
                                                                     font=('Helvetica', '24', 'bold'),
                                                                     anchor=tk.SW, justify=tk.LEFT,
                                                                     fill='#ffffff', tags='legend')
                    self.__position_text = self.tk_container.create_text(0, 0, text='',
                                                                     font=('Helvetica', '12'),
                                                                     anchor=tk.SE, justify=tk.CENTER,
                                                                     fill='#ffff00', tags='legend')
                scalebar_pos = (margin_width, self.tk_container.winfo_height() - scalebar_shape[1] - margin_width)
                self.tk_container.coords(self.__scalebar, scalebar_pos[0], scalebar_pos[1], scalebar_pos[0] + scalebar_shape[0], scalebar_pos[1] + scalebar_shape[1])
                representative_width = unit_registry.Quantity(representative_width, pixel_unit_description)
                self.tk_container.itemconfigure(self.__scalebar_text,
                                            text=f"{representative_width.to_compact():~0.0f}")
                self.tk_container.coords(self.__scalebar_text, scalebar_pos[0], scalebar_pos[1] - margin_width)

                if self.__current_position_px is not None:
                    current_position_physical = (self.__current_position_px - origin_pixel) * pixel_unit_value
                    current_position_with_units = unit_registry.Quantity(current_position_physical, pixel_unit_description)
                    self.tk_container.itemconfigure(
                        self.__position_text,
                        text=f'{self.__current_position_px[0] - 0.5:0.1f} px, {self.__current_position_px[1] - 0.5:0.1f} px\n' +
                             f'{current_position_with_units[0].to_compact():~0.1f}, {current_position_with_units[1].to_compact():~0.1f}\n' +
                             'vertical, horizontal')
                    position_pos = (self.tk_container.winfo_width() - margin_width, self.tk_container.winfo_height() - margin_width)
                    self.tk_container.coords(self.__position_text, position_pos[0], position_pos[1] - margin_width)
                else:
                    self.tk_container.itemconfigure(self.__position_text, text='')  # hide
            self.tk_container.tag_raise('legend')

    def fit_to_window(self):
        canvas_shape = self.shape
        # Use a blank image for now
        image_array = np.zeros((1, 1, 3), dtype=np.uint8)
        scale = np.min(canvas_shape / image_array.shape[:2])
        canvas_image_shape = np.round(np.array(image_array.shape[:2]) * scale).astype(np.int)
        self.__canvas_pil_image = Image.fromarray(image_array)  # Update image
        self.__canvas_pil_image = self.__canvas_pil_image.resize(size=(canvas_image_shape[::-1]), resample=Image.NEAREST)  # scale to fit display area
        if self.__canvas_tk_image_index is not None:
            self.tk_container.delete(self.__canvas_tk_image_index)
        self.__canvas_tk_image = ImageTk.PhotoImage(self.__canvas_pil_image)
        self.__canvas_tk_image_index = self.tk_container.create_image(
            self.tk_container.winfo_width() / 2.0, self.tk_container.winfo_height() / 2.0, image=self.__canvas_tk_image,
            anchor=tk.CENTER, state=tk.NORMAL)

    def __on_start_drag(self, event):
        start_pos = self.__get_event_position_in_image(event)
        self.__drag_start_pos = start_pos
        self.__drag_roi = self.__get_roi_callback()
        log.debug(f'Started drag at {start_pos}')

    def __on_drag(self, event):
        if self.__drag_start_pos is not None:
            new_pos = self.__get_event_position_in_image(event)
            log.debug(f'Dragging from {self.__drag_start_pos} to {new_pos}')
            # self.__drag_start_pos = new_pos
            new_roi = Roi(top_left=self.__drag_roi.top_left - (new_pos - self.__drag_start_pos),
                          shape=self.__drag_roi.shape)
            self.__set_roi_callback(new_roi)
        else:
            self.__on_start_drag(event)

    def __on_stop_drag(self, event):
        log.debug("Dragging stopped.")
        self.__on_drag(event)
        self.__drag_start_pos = None

    def __on_start_zoom(self, event):
        new_pos = (event.y, event.x)
        log.debug(f"Started zoom at {new_pos}")
        self.__zoom = Roi(top_left=new_pos)

        # Start drawing a rectangle
        if self.__zoom_rectangle is not None:
            self.tk_container.delete(self.__zoom_rectangle)
        self.__zoom_rectangle = self.tk_container.create_rectangle(self.__zoom.left, self.__zoom.top, self.__zoom.left + self.__zoom.width, self.__zoom.top + self.__zoom.height, outline='#ffff00')

    def __on_zoom(self, event):
        log.debug(f"Stretching zoom rectangle to {self.zoom}")
        new_pos = (event.y, event.x)
        if self.__zoom is None:
            self.__zoom = Roi(bottom_right=new_pos)
        else:
            self.__zoom.bottom_right = new_pos

        # Draw a rectangle
        if self.__zoom_rectangle is not None:
            self.tk_container.delete(self.__zoom_rectangle)
        self.__zoom_rectangle = self.tk_container.create_rectangle(self.__zoom.left, self.__zoom.top, self.__zoom.left + self.__zoom.width, self.__zoom.top + self.__zoom.height, outline='#ffff00')

    def __on_stop_zoom(self, event):
        # Handle the last bit of movement
        new_pos = (event.y, event.x)
        if self.__zoom is None:
            self.__zoom = Roi(bottom_right=new_pos)
        else:
            self.__zoom.bottom_right = new_pos

        image_rect = Roi(top_left=self.__get_position_in_image(self.__zoom.top_left),
                         bottom_right=self.__get_position_in_image(self.__zoom.bottom_right), dtype=np.int)
        absolute_rect = image_rect.convex * self.__get_roi_callback()  # Determine the absolute ROI
        if all(absolute_rect.shape > 0):
            log.debug(f'Zooming to {absolute_rect}')
            self.__set_roi_callback(absolute_rect)

        self.__zoom = None

        # Delete the rectangle now
        if self.__zoom_rectangle is not None:
            self.tk_container.delete(self.__zoom_rectangle)

    def report_position(self, event=None):
        if event is not None:
            self.__current_position_px = self.__get_event_position_in_image(event) + self.__get_roi_callback().top_left
        else:
            self.__current_position_px = None
        self.update_legend()

    def reset_drag_and_zoom(self, event=None):
        self.__set_roi_callback(None)

    def __on_mousewheel(self, event):
        scroll_up_amount = event.delta / 120.0

        pos_in_image = self.__get_event_position_in_image(event)
        self.zoom(-scroll_up_amount, pos_in_image)

    def __get_event_position_in_image(self, event) -> np.ndarray:
        """
        :param event: The Canvas <Motion> event with attributes x and y.
        :return: The pixel position in the Canvas'es image.
        """
        return self.__get_position_in_image(np.array([event.y, event.x]))

    def __get_position_in_image(self, absolute_position) -> np.ndarray:
        """
        Converts the absolute position in the canvas to a position in the canvas image.
        :param absolute_position: The pixel position on the canvas [vertical, horizontal]
        :return: The pixel position in the image with sub-pixel resolution [vertical, horizontal].
        The top-left is [0, 0].
        """
        canvas_shape = np.array([self.tk_container.winfo_height(), self.tk_container.winfo_width()])
        canvas_image_shape = np.array([self.__canvas_tk_image.height(), self.__canvas_tk_image.width()])
        canvas_image_position = ((canvas_shape - canvas_image_shape) / 2).astype(np.int)

        relative_position = absolute_position - canvas_image_position
        normalized_position = relative_position / canvas_image_shape
        image_shape = self.__get_roi_callback().shape
        image_position = image_shape * normalized_position

        return image_position

    def zoom(self, zoom_in_amount, pos_in_image=None):
        """
        Zoom in (+) and out (-). One unit corresponds to 10%
        :param zoom_in_amount: Scroll amount: Zoom in (+) and out (-). One unit corresponds to 10%
        :param pos_in_image: The position to keep fixed in the region of interest, default: center
        """
        previous_roi = self.__get_roi_callback()
        new_roi = Roi(center=previous_roi.center, shape=previous_roi.shape * (1.10 ** (-zoom_in_amount)))

        self.__set_roi_callback(new_roi)

    def drag(self, drag_vector):
        previous_roi = self.__get_roi_callback()
        new_roi = Roi(top_left=previous_roi.top_left - drag_vector, shape=previous_roi.shape)

        self.__set_roi_callback(new_roi)

