'''
Creates PNG images of BACS characters for use in the experiments.

Vidal, C., Content, A., & Chetail, F. (2017). BACS: The Brussels
  Artificial Character Sets for studies in cognitive psychology and
  neuroscience, 49(6), 2093–2112. https://doi.org/10.3758/s13428-016-0844-8
'''

import core
import cairosvg


TEMPLATE = '''<svg width='100px' height='200px' xmlns:rdf='http://www.w3.org/1999/02/22-rdf-syntax-ns#' xmlns:svg='http://www.w3.org/2000/svg' xmlns='http://www.w3.org/2000/svg' version='1.1'><text x='50px' y='150px' style='font-family:"{font}"; font-size:{fontsize}px; text-anchor:middle;' fill='{color}'>{letter}</text></svg>'''

BACS1_lower = 'abcdefghijklmnopqrstuvyz'
BACS1_upper = 'ABCDEFGHIJKLMNOPQRSTUVYZ'
BACS2_sans_lower = 'abcdefghijklmnopqrstuvyzwx'
BACS2_sans_upper = 'ABCDEFGHIJKLMNOPQRSTUVYZWX'
BACS2_serif_lower = 'abcdefghijklmnopqrstuvyzwx'
BACS2_serif_upper = 'ABCDEFGHIJKLMNOPQRSTUVYZWX'


def make_image_file(out_dir, letter, style):
	filename = f'{letter.lower()}.png'
	filepath = str(out_dir / filename)
	svg = TEMPLATE.format(letter=letter, **style)
	with open(filepath, mode='w', encoding='utf-8') as file:
		file.write(svg)
	cairosvg.svg2png(url=filepath, write_to=filepath)


if __name__ == '__main__':

	for letter in BACS2_sans_upper:
		make_image_file(core.ROOT / 'experiments' / 'online' / 'client' / 'images' / 'alphabet' , letter, {'font':'BACS2', 'fontsize':190, 'color':'black'})
