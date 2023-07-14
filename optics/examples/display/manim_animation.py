from manim import *
import numpy as np


config.background_color = WHITE  # for print stills
config.quality = 'low_quality'  # Comment out to get high resolution at 60fps
grid_tile_subdivisions = 4
partial_calc = False  # Calculates (A+B)^{-1}Ax instead of B(A+B)^{-1}Ax
speed_factor = 3  # Times slower than original


class Preconditioner(Scene):
    def construct(self):
        # The coordinate system
        complex_plane = ComplexPlane().set_color(BLUE)
        complex_plane.add(complex_plane.get_coordinate_labels(color=complex_plane.color))   #.add_coordinates().set_color(BLUE)
        # The values of b should be contained in this circle
        circle = Circle(1.0, color=complex_plane.color).move_to([1, 0, 0])

        # Control the complex value of v = 1 - b by amplitude and phase
        amplitude = ValueTracker(1e5)
        phase = ValueTracker(PI)

        # Indicate the position of b with a dot and the label 'B'
        b_dot = Dot(color=BLACK)
        b_label = Text('B', color=str(b_dot.color)).scale(0.5)
        def get_b_pos():
            b = 1.0 - amplitude.get_value() * np.exp(1j * phase.get_value())
            return [b.real, b.imag, 0]
        b_dot.add_updater(lambda _: _.move_to(get_b_pos()))
        b_label.add_updater(lambda _: _.next_to(b_dot, buff=0.1))
        b_dot_with_label = VGroup(b_dot, b_label)
        b_dot_with_label.update()
        # Text display of value of b
        def get_display_b() -> Mobject:
            b = b_dot.get_center()
            b = b[0] + 1j * b[1]
            v = 1 - b
            amp = np.abs(v)
            ph = np.angle(v)
            if amp < 1e4:
                tex_string = r'B & = 1 -' + f'{amp:0.2f}' + r'\mathrm{e}^{' + f'{ph / np.pi:+0.2f}' + r'\pi\mathrm{i}}\\' \
                             + r' & = ~~~~~' + f'{b.real: 0.2f}{b.imag:+0.2f}i'
            else:
                tex_string = r'B & = 1 - \infty\mathrm{e}^{' + f'{ph / np.pi:+0.2f}' + r'\pi\mathrm{i}}\\'
            return MathTex(tex_string, color=b_dot.color).to_corner(UP + LEFT)
        display_b = get_display_b()
        display_b.add_updater(lambda _: _.become(get_display_b()))

        def get_display_title() -> Mobject:
            b = b_dot.get_center()
            b = b[0] + 1j * b[1]
            v = 1 - b
            amp = np.abs(v)
            if amp < 2.5:
                tex_string = r'(A+B)^{-1}Ax' if partial_calc else r'\Gamma^{-1}Ax'
                opacity = 1 - (amp - 1) / (2.5 - 1)
            elif amp > 100:
                tex_string = 'Ax'
                opacity = np.log10(amp) / 4
            else:
                tex_string = ''
                opacity = 0
            opacity = np.clip(opacity, 0, 1)
            return MathTex(tex_string, color=b_dot.color, fill_opacity=opacity, stroke_opacity=opacity).scale(3-partial_calc).next_to(ORIGIN, direction=LEFT)
        display_title = get_display_title()
        display_title.add_updater(lambda _: _.become(get_display_title()))

        # The real half-plane should map to the circle
        grid_radius = 10 * grid_tile_subdivisions
        squares = [Square(fill_opacity=0.5, stroke_width=1, stroke_color=WHITE,
                          fill_color=('#00C000FF' if (1 & (re//grid_tile_subdivisions ^ im//grid_tile_subdivisions)) else '#C0C000FF')
                          ).scale(0.5).align_to([re, im, 0], DOWN + LEFT)
                   for re in np.arange(grid_radius)
                   for im in np.arange(-grid_radius, grid_radius)]
        cartesian_checkerboard = VGroup(*squares).scale(1 / grid_tile_subdivisions, about_point=[0, 0, 0])

        def get_checker():
            inv = lambda _: (1 / _) if _ != 0 else 1e10
            b = 1 - amplitude.get_value() * np.exp(1j * phase.get_value())
            checker = cartesian_checkerboard.copy()
            # checker.apply_complex_function(lambda _: inv(_ + b) * _)
            checker.apply_complex_function(lambda _: inv(inv(_) + inv(b)))
            return checker
        checkerboard = get_checker()
        checkerboard.add_updater(lambda _: _.become(get_checker()))

        #
        # The animation sequence
        #
        # Intro
        self.play(FadeIn(complex_plane))
        self.play(GrowFromCenter(circle), run_time=0.5 * speed_factor)
        self.add(display_b)
        self.play(Create(checkerboard))
        self.add(display_title)
        self.add(b_dot_with_label)
        self.play(FadeIn(display_b))
        # Ax -> prec Ax
        self.play(amplitude.animate.set_value(0.75), rate_func=lambda _: 1 + np.exp(-5 * np.log(10)) - np.exp(-5 * np.log(10) * _), run_time=2 * speed_factor)
        # Rotate B left-right
        self.play(phase.animate.set_value(1.5 * PI), run_time=speed_factor)
        self.play(phase.animate.set_value(-PI), run_time=3 * speed_factor)
        # Rotate B back
        self.play(phase.animate.set_value(1.5 * PI), run_time=3 * speed_factor)
        # Invalid B
        self.play(amplitude.animate.set_value(3.0), run_time=speed_factor)
        # Rotate invalid B
        self.play(phase.animate.set_value(-0.5 * PI), run_time=5 * speed_factor)
        # Back to valid B
        self.play(amplitude.animate.set_value(0.75), run_time=speed_factor)
        self.play(phase.animate.set_value(-1.5 * PI), run_time=5 * speed_factor)
        # prec Ax -> Ax
        self.play(amplitude.animate.set_value(1e5), rate_func=lambda _: - np.exp(-5 * np.log(10)) + np.exp(-5 * np.log(10) * (1 - _)), run_time=2 * speed_factor)
        # Outro
        self.wait(2 * speed_factor)
        self.play(FadeOut(checkerboard, *self.mobjects), run_time=speed_factor)


if __name__ == '__main__':
    scene = Preconditioner()
    scene.render()
