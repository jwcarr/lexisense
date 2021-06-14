'''
Creates PNG images of BACS characters for use in the experiments.

Vidal, C., Content, A., & Chetail, F. (2017). BACS: The Brussels
  Artificial Character Sets for studies in cognitive psychology and
  neuroscience, 49(6), 2093–2112. https://doi.org/10.3758/s13428-016-0844-8
'''

import core
import cairocffi as cairo


BACS1_lower = {'font_face': 'BACS1', 'lower_case': True, 'characters': 'abcdefghijklmnopqrstuvyz'}
BACS1_upper = {'font_face': 'BACS1', 'lower_case': False, 'characters': 'abcdefghijklmnopqrstuvyz'}
BACS2_sans_lower = {'font_face': 'BACS2', 'lower_case': True, 'characters': 'abcdefghijklmnopqrstuvwxyz'}
BACS2_sans_upper = {'font_face': 'BACS2', 'lower_case': False, 'characters': 'abcdefghijklmnopqrstuvwxyz'}
BACS2_serif_lower = {'font_face': 'BACS2', 'lower_case': True, 'characters': 'abcdefghijklmnopqrstuvwxyz'}
BACS2_serif_upper = {'font_face': 'BACS2', 'lower_case': False, 'characters': 'abcdefghijklmnopqrstuvwxyz'}
Courier_upper = {'font_face': 'Courier New', 'lower_case': False, 'characters': 'abcdefghijklmnopqrstuvwxyz'}


def make_image(out_dir, letter, font, color=(0, 0, 0), bg_color=(1, 1, 1)):
	font_face = cairo.ToyFontFace(font['font_face'])
	scaled_font = cairo.ScaledFont(font_face, cairo.Matrix(xx=100, yy=100))
	char_width = scaled_font.text_extents(letter)[4]
	font_size = 100 / char_width * 100
	surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 100, 200)
	context = cairo.Context(surface)
	with context:
		context.set_source_rgb(*bg_color)
		context.paint()
	with context:
		context.set_source_rgb(*color)
		context.set_font_face(font_face)
		context.set_font_size(font_size)
		context.move_to(0, 140)
		if font['lower_case']:
			context.show_text(letter.lower())
		else:
			context.show_text(letter.upper())
	surface.write_to_png(str(out_dir / f'{letter}.png'))

def make_all_images(font, out_dir):
	for letter in font['characters']:
		make_image(out_dir, letter, font)


if __name__ == '__main__':

	# make_all_images(BACS2_sans_upper, core.ROOT / 'experiments' / 'online' / 'client' / 'images' / 'alphabet')
	make_all_images(Courier_upper, core.ROOT / 'experiments' / 'online' / 'client' / 'images' / 'courier')
