from __future__ import annotations

import numpy as np
from typing import Optional
import logging

log = logging.getLogger(__name__)


class Lens:
    """
    A class to represent lenses.
    """
    def __init__(self, focal_length: float = 1.0, pupil_diameter: Optional[float] = None, refractive_index: float = 1.0,
                 focal_plane_diameter: float = 0.0, numerical_aperture: Optional[float] = None):
        """
        Define a generic lens for imaging.

        :param focal_length: the focal length of the objective, in meters.

        :param pupil_diameter: the diameter of the pupil, iris-aperture, or beam, in meters

        :param refractive_index: The refractive index of the medium this objective is designed for.

        :param focal_plane_diameter: The diameter of the focal-plane area over which objects are considered in focus, in meters.

        :param numerical_aperture: (optional) The numerical aperture can be specified as an alternative to the pupil diameter.

        """
        self.__focal_length = focal_length
        self.__pupil_diameter = pupil_diameter
        self.__refractive_index = refractive_index
        self.__focal_plane_diameter = focal_plane_diameter

        if pupil_diameter is not None:
            self.pupil_diameter = pupil_diameter
            if numerical_aperture is not None:
                log.warning('Both the pupil diameter and numerical aperture are specified. Ignoring the numerical aperture!')
        else:
            if numerical_aperture is not None:
                self.pupil_diameter = 2.0 * focal_length * numerical_aperture / refractive_index
            else:
                log.warning('Neither pupil diameter, nor numerical aperture is specified. Defaulting to 0.0!')
                self.pupil_diameter = 0.0

    @property
    def focal_length(self) -> float:
        """The focal length of this lens, i.e. its distance of the focal point when illuminated using a collimated beam, in meters."""
        return self.__focal_length

    @focal_length.setter
    def focal_length(self, new_f: float):
        self.__focal_length = new_f

    @property
    def pupil_diameter(self) -> float:
        """The diameter of the pupil, iris-aperture, or beam, in meters"""
        return self.__pupil_diameter

    @pupil_diameter.setter
    def pupil_diameter(self, new_diameter: float):
        self.__pupil_diameter = new_diameter

    @property
    def refractive_index(self) -> float:
        """The refractive index of the medium this objective is designed for."""
        return self.__refractive_index

    @refractive_index.setter
    def refractive_index(self, value: float):
        self.__refractive_index = value

    @property
    def focal_plane_diameter(self) -> float:
        """The maximum diameter of the object, in meters."""
        return self.__focal_plane_diameter

    @focal_plane_diameter.setter
    def focal_plane_diameter(self, new_diameter: float):
        self.__focal_plane_diameter = new_diameter

    @property
    def numerical_aperture(self) -> float:
        r"""
        The numerical aperture of the lens :math:`\sin(\alpha)n`, where :math:`\alpha` is the half angle of acceptance
        of the objective and :math:`n` is the refractive index. Compliance with the Abbe Sine condition is assumed.
        """
        return self.__refractive_index * self.__pupil_diameter / 2.0 / self.__focal_length

    @numerical_aperture.setter
    def numerical_aperture(self, new_na: float):
        """Set the numerical aperture."""
        self.__pupil_diameter = new_na * self.__focal_length / self.__refractive_index * 2.0 / self.__pupil_diameter

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.focal_length}, {self.pupil_diameter}, refractive_index={self.refractive_index}, focal_plane_diameter={self.focal_plane_diameter})"

    def __str__(self) -> str:
        return f"{self.__class__.__name__} f={self.focal_length:0.3f}, D={self.pupil_diameter:0.6f}"

    def __eq__(self, other: Objective) -> bool:
        return type(self) == type(other) and repr(self) == repr(other)


class Objective(Lens):
    """
    A class to represent microscope objective lenses.
    """
    def __init__(self, magnification: Optional[float] = None, numerical_aperture: float = 1.0,
                 refractive_index: float = 1.0, tube_lens_focal_length: float = 200e-3,
                 field_number: Optional[float] = None, max_object_diameter: Optional[float] = None):
        """
        Define a generic objective

        :param magnification: the number in front of the 'X' on the objective's barrel.

        :param numerical_aperture: the number in front of the '/...X' on the objective's barrel.

        :param refractive_index: The refractive index of the medium this objective is designed for.

        :param tube_lens_focal_length: The focal length in meters of the tube lens this objective is designed for.
            The most common tube length lens is 200 mm, used for Nikon and Leica. Olympus uses 180 mm, and Zeiss 165 mm.

        :param field_number: The field number in meters in image space is the maximum image diameter that
            the objective is designed to produce with low aberrations. (default: 1m)

        :param max_object_diameter: (optional) The maximum diameter of the object in meters in object space that the
            objective is designed to image. It equals field_number / magnification.
        """
        if max_object_diameter is not None:
            if field_number is None:
                field_number = max_object_diameter / magnification
            elif magnification is None:
                magnification = max_object_diameter / field_number
            else:
                log.warning(f'Ignoring max_object_diameter = {max_object_diameter} because both magnification and field_number are already specified.')
        else:
            if magnification is None:
                magnification = 1.0
                log.warning(f'The magnification is not specified, defaulting to 1.')
            if field_number is None:
                field_number = 0.0
                log.warning(f'The field_number is not specified, defaulting to 0.')

        super().__init__(focal_length=tube_lens_focal_length / abs(magnification),
                         numerical_aperture=numerical_aperture, refractive_index=refractive_index,
                         focal_plane_diameter=field_number)

        self.__tube_lens_focal_length = tube_lens_focal_length

    @property
    def magnification(self) -> float:
        """The magnification of the objective, given as a positive value."""
        return self.tube_lens_focal_length / self.focal_length

    @magnification.setter
    def magnification(self, value: float):
        self.focal_length = self.tube_lens_focal_length / np.abs(value)

    @property
    def tube_lens_focal_length(self) -> float:
        """The tube lens the objective is designed for, given the magnification on the barrel."""
        return self.__tube_lens_focal_length

    @tube_lens_focal_length.setter
    def tube_lens_focal_length(self, value: float):
        self.__tube_lens_focal_length = value

    @property
    def field_number(self) -> float:
        """The field number, the diameter of the image, in meters. This is synonym to the focal_plane_diameter of a lens."""
        return self.focal_plane_diameter

    @field_number.setter
    def field_number(self, new_field_number: float):
        self.focal_plane_diameter = new_field_number

    @property
    def max_object_diameter(self) -> float:
        """The diameter of the focal plane that is considered to be in focus in object space."""
        return self.field_number / self.magnification

    @max_object_diameter.setter
    def max_object_diameter(self, new_max_object_diameter: float):
        """Changing the maximum diameter of the object will scale the field_number proportionally."""
        self.field_number = new_max_object_diameter / self.magnification

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.magnification}, {self.numerical_aperture}, " + \
               f"refractive_index={self.refractive_index}, tube_lens_focal_length={self.tube_lens_focal_length}, " + \
               f"field_number={self.field_number}, max_object_diameter={self.max_object_diameter})"

    def __str__(self) -> str:
        return f"{self.__class__.__name__} {self.magnification:0.0f}X/{self.numerical_aperture:0.2f}"


class InfinityCorrected(Objective):
    """
    A class to represent infinity-corrected microscope objective lenses.
    """
    def __init__(self, magnification: float = 1.0, numerical_aperture: float = 1.0, refractive_index: float = 1.0,
                 tube_lens_focal_length: float = 200e-3, field_number: float = 1.0,
                 max_object_diameter: Optional[float] = None):
        """
        Define an infinity corrected objective.

        :param magnification: the number in front of the 'X' on the objective's barrel.

        :param numerical_aperture: the number in front of the '/...X' on the objective's barrel.

        :param refractive_index: The refractive index of the medium this objective is designed for.

        :param tube_lens_focal_length: The focal length in meters of the tube lens this objective is designed for.
            The most common tube length lens is 200 mm, used for Nikon and Leica. Olympus uses 180 mm, and Zeiss 165 mm.

        :param field_number: The field number in meters in image space is the maximum image diameter that
            the objective is designed to produce with low aberrations. (default: 1m)

        :param max_object_diameter: The maximum diameter of the object in meters in object space that the objective is
            designed to image. It equals field_number / magnification (optional)

        """
        super().__init__(magnification=magnification, numerical_aperture=numerical_aperture,
                         refractive_index=refractive_index, tube_lens_focal_length=tube_lens_focal_length,
                         field_number=field_number, max_object_diameter=max_object_diameter)


class Nikon(InfinityCorrected):
    """
    A class to represent Nikon microscope objectives lenses.
    These objectives have a tube lens with focal length of 200 mm.
    """
    def __init__(self, magnification: float = 1.0, numerical_aperture: float = 1.0, refractive_index: float = 1.0,
                 field_number: float = 1.0, max_object_diameter: Optional[float] = None):
        """
        Define an infinity corrected objective.

        :param magnification: the number in front of the 'X' on the objective's barrel.

        :param numerical_aperture: the number in front of the '/...X' on the objective's barrel.

        :param refractive_index: The refractive index of the medium this objective is designed for.

        :param field_number: The field number in meters in image space is the maximum image diameter that
            the objective is designed to produce with low aberrations. (default: 1m)

        :param max_object_diameter: The maximum diameter of the object in meters in object space that the objective is
            designed to image. It equals field_number / magnification (optional).

        """
        super().__init__(magnification=magnification, numerical_aperture=numerical_aperture,
                         refractive_index=refractive_index, tube_lens_focal_length=200e-3,
                         field_number=field_number, max_object_diameter=max_object_diameter)


class Mitutoyo(Nikon):
    pass


class EdmundOptics(Nikon):
    pass


class Leica(Nikon):
    """
    A class to represent Leica microscope objectives lenses.
    These objectives have a tube lens with focal length of 200 mm.
    """
    def __init__(self, *args, **kwargs):
        """
        Define a Leica objective.

        :param magnification: the number in front of the 'X' on the objective's barrel.

        :param numerical_aperture: the number in front of the '/...X' on the objective's barrel.

        :param refractive_index: The refractive index of the medium this objective is designed for.

        """
        super().__init__(*args, **kwargs)


class Olympus(InfinityCorrected):
    """
    A class to represent Olympus microscope objectives lenses.
    These objectives have a tube lens with focal length of 200 mm.
    """
    def __init__(self, magnification: float = 1.0, numerical_aperture: float = 1.0, refractive_index: float = 1.0,
                 field_number: float = 1.0, max_object_diameter: Optional[float] = None):
        """
        Define an infinity corrected objective.

        :param magnification: the number in front of the 'X' on the objective's barrel.

        :param numerical_aperture: the number in front of the '/...X' on the objective's barrel.

        :param refractive_index: The refractive index of the medium this objective is designed for.

        :param field_number: The field number in meters in image space is the maximum image diameter that
            the objective is designed to produce with low aberrations. (default: 1m)

        :param max_object_diameter: The maximum diameter of the object in meters in object space that the objective is
            designed to image. It equals field_number / magnification (optional).

        """
        super().__init__(magnification=magnification, numerical_aperture=numerical_aperture,
                         refractive_index=refractive_index, tube_lens_focal_length=180e-3,
                         field_number=field_number, max_object_diameter=max_object_diameter)


class Zeiss(InfinityCorrected):
    """
    A class to represent Zeiss microscope objectives lenses.
    These objectives have a tube lens with focal length of 200 mm.
    """
    def __init__(self, magnification: float = 1.0, numerical_aperture: float = 1.0, refractive_index: float = 1.0,
                 field_number: float = 1.0, max_object_diameter: Optional[float] = None):
        """
        Define an infinity corrected objective.

        :param magnification: the number in front of the 'X' on the objective's barrel.

        :param numerical_aperture: the number in front of the '/...X' on the objective's barrel.

        :param refractive_index: The refractive index of the medium this objective is designed for.

        :param field_number: The field number in meters in image space is the maximum image diameter that
            the objective is designed to produce with low aberrations. (default: 1m)

        :param max_object_diameter: The maximum diamater of the object in meters in object space that the objective is
            designed to image. It equals field_number / magnification (optional).

        """
        super().__init__(magnification=magnification, numerical_aperture=numerical_aperture,
                         refractive_index=refractive_index, tube_lens_focal_length=165e-3,
                         field_number=field_number, max_object_diameter=max_object_diameter)


if __name__ == '__main__':
    def show(o: InfinityCorrected):
        print(f"{o.numerical_aperture:0.2f}/{o.magnification:0.0f}X "
              f"has a back aperture diameter of {o.pupil_diameter*1e3:0.0f} mm and "
              f"a focal length of {o.focal_length*1e3:0.1f} mm.")


    obj = Nikon(40, 0.80)
    show(obj)

    obj.pupil_diameter = 8e-3
    show(obj)

    print(obj)
