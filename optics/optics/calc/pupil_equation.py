import numpy as np
import re
import numexpr as ne

from . import log


class ParseError(Exception):
    """A parse error is thrown when a specified text string cannot be parsed."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def parse(pupil_equation_string: str):
    """
    Parses a text string that describes a pupil function with Cartesian (vector) arguments.

    All operations are point-by-point operations on complex matrices. Boolean values are converted to 0 and 1.
    The following symbols are accepted:
        - u and U are converted to x
        - v and V are converted to y
        - r, rho, R, Rho, and RHO are converted to :code:`numpy.sqrt(x**2 + y**2)`
        - p, phi, P, Phi, and PHI are converted to :code:`numpy.arctan2(y, x)`
        - t, time, T, Time, and TIME are converted to t
        - w, wavelength, lambda, l, Wavelength, W, L, Lambda, and LAMBDA are converted to w
        - a(cos|sin|tan)(h)(2) are interpreted as arc(cos|sin|tan)(h)(2)
        - pi is interpreted as the constant :code:`numpy.pi`
        - i is converted to j and j is converted to 1j
    in addition to the operators and functions listed here:
    https://numexpr.readthedocs.io/projects/NumExpr3/en/latest/user_guide.html#supported-operators

    :param pupil_equation_string: The equation to parse.

    :return: A function in the arguments (x, y, t=0, w=1).

        Input to returned function:
            - `x` : the x-coordinate in pixels
            - `y` : the y-coordinate in pixels
            - `r|rho` : the radial coordinate in pixels: :math:`\sqrt(x^2 + y^2)`
            - `p|ph|phi` : the angle with the x-axis in the direction of the positive y-axis: :code:`numpy.arctan2(y, x)`
            - `t|time` : the time in seconds
            - `w|wavelength` : the wavelength in meters

        The center pixel (0, 0) is generally assumed to be the one past the center in case of an even number of pixels.

    """
    # Put some extra spaces around things to facilitate multiple matches
    pupil_equation_string = re.sub(r"(^|\W)(\w+)(\W|$)", r"\1 \2 \3", pupil_equation_string, flags=re.IGNORECASE)

    pupil_equation_string = re.sub(r"(^|\W)time(\W|$)", r"\1 t \2", pupil_equation_string, flags=re.IGNORECASE)
    pupil_equation_string = re.sub(r"(^|\W)th(eta)?(\W|$)", r"\1 p \3", pupil_equation_string, flags=re.IGNORECASE)
    pupil_equation_string = re.sub(r"(^|\W)(ph|f)i?(\W|$)", r"\1 p \3", pupil_equation_string, flags=re.IGNORECASE)
    pupil_equation_string = re.sub(r"(^|\W)p(\W|$)", r"\1 arctan2(x,y)\2", pupil_equation_string, flags=re.IGNORECASE)
    pupil_equation_string = re.sub(r"(^|\W)r(ho)?(\W|$)", r"\1 sqrt(y**2+x**2)\3", pupil_equation_string, flags=re.IGNORECASE)
    pupil_equation_string = re.sub(r"(^|\W)l(ambda)?(\W|$)", r"\1 w \3", pupil_equation_string, flags=re.IGNORECASE)
    pupil_equation_string = re.sub(r"(^|\W)wave(length)?(\W|$)", r"\1 w \3", pupil_equation_string, flags=re.IGNORECASE)

    # Convert complex unit: i -> j -> 1j
    pupil_equation_string = re.sub(r"(^|\W)(\d*)i(\W|$)", r"\1 \2j \3", pupil_equation_string, flags=re.IGNORECASE)
    pupil_equation_string = re.sub(r"(^|\W)j(\W|$)", r"\g<1> 1j \2", pupil_equation_string, flags=re.IGNORECASE)

    # Convert pi to
    pupil_equation_string = re.sub(r"(^|\W)pi(\W|$)", f"\\g<1> {np.pi:0.12f} \\2", pupil_equation_string, flags=re.IGNORECASE)

    # Convert power to python exponent
    pupil_equation_string = re.sub(r"\^", r"**", pupil_equation_string)

    # Convert a(cos|sin|tan)(h)(2) to arc(cos|sin|tan)(h)(2)
    pupil_equation_string = re.sub(r"(^|\W)a(cos|sin|tan)(h?)(2?)(\W|$)", r"\1 arc\2\3\4 \5",
                                   pupil_equation_string, flags=re.IGNORECASE)

    # Convert uppercase X and Y to lowercase x and y, respectively
    pupil_equation_string = re.sub(r"(^|\W)X(\W|$)", r"\1 x \2", pupil_equation_string)
    pupil_equation_string = re.sub(r"(^|\W)Y(\W|$)", r"\1 y \2", pupil_equation_string)

    # Convert u and v to x and y, respectively
    pupil_equation_string = re.sub(r"(^|\W)u(\W|$)", r"\1 x \2", pupil_equation_string, flags=re.IGNORECASE)
    pupil_equation_string = re.sub(r"(^|\W)v(\W|$)", r"\1 y \2", pupil_equation_string, flags=re.IGNORECASE)

    # if radius != 1:
    #     # Convert u and v to x and y, respectively
    #     pupil_equation_string = re.sub(r"(^|\W)x(\W|$)", r"\1 x / radius \2", pupil_equation_string, flags=re.IGNORECASE)
    #     pupil_equation_string = re.sub(r"(^|\W)y(\W|$)", r"\1 y / radius \2", pupil_equation_string, flags=re.IGNORECASE)

    # Check what input arguments are used
    x_dependent = re.compile(r".*(^|\W)x(\W|$).*").match(pupil_equation_string) is not None
    y_dependent = re.compile(r".*(^|\W)y(\W|$).*").match(pupil_equation_string) is not None
    time_dependent = re.compile(r".*(^|\W)t(\W|$).*").match(pupil_equation_string) is not None
    wavelength_dependent = re.compile(r".*(^|\W)w(\W|$).*").match(pupil_equation_string) is not None
    # radius_dependent = re.compile(r".*(^|\W)radius(\W|$).*").match(pupil_equation_string) is not None

    # Strip white space and line breaks
    pupil_equation_string = re.sub(r"\s+", r"", pupil_equation_string, flags=re.IGNORECASE)

    # Prepare a description
    arg_strings = []
    if x_dependent:
        arg_strings.append('x')
    if y_dependent:
        arg_strings.append('y')
    if time_dependent:
        arg_strings.append('t')
    if wavelength_dependent:
        arg_strings.append('w')
    # if radius_dependent:
    #     arg_strings.append('radius')
    args_string = ', '.join(arg_strings)
    description = f"p({args_string}) = {pupil_equation_string}"

    log.debug(f"Parsed equation as: {description}")

    def pupil_function(x, y, t=0.0, w=1.0):
        try:
            args = dict(x=x, y=y, t=t, w=w)  # The following makes sure that it are ndarrays and converts it to complex values as necessary
            return ne.evaluate(pupil_equation_string, local_dict=args).astype(complex) \
                   + np.zeros_like(x) + np.zeros_like(y)  # Make sure that the result is a matrix of the right size
        except SyntaxError as err:
            log.info(f'Failed to parse pupil function: "{pupil_equation_string}".')
            raise ParseError(err)
        except KeyError as err:
            log.info(f'Pupil function: "{pupil_equation_string}" has an unrecognized variable: ({str(err)}).'
                     + "After parsing, the allowed variables are x, y, t, and w.")
            raise ParseError(err)
        except TypeError as err:
            log.info(f'Pupil function: "{pupil_equation_string}" has an unrecognized type: ({str(err)}).')
            raise ParseError(err)
        except NotImplementedError as err:
            log.info(f'Pupil function: "{pupil_equation_string}" contains functions that are not implemented ({str(err)}).')
            raise ParseError(err)
        except Exception as e:
            log.error(f'Unexpected exception while parsing pupil equation: "{pupil_equation_string}"!')
            raise ParseError(e)

    # Mark what input arguments will be used
    pupil_function.x_dependent = x_dependent
    pupil_function.y_dependent = y_dependent
    pupil_function.time_dependent = time_dependent
    pupil_function.wavelength_dependent = wavelength_dependent
    # pupil_function.radius_dependent = radius_dependent

    pupil_function.__str__ = description

    return pupil_function


if __name__ == '__main__':
    p = parse('(R<1)*exp(i*phi)')
    print(p.__str__)

    x = np.arange(-1, 1, 0.5)[:, np.newaxis]
    y = np.arange(-1, 1, 0.5)[np.newaxis, :]
    print(p(x, y))

