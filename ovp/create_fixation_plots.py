'''
This script extracts the relevant fixation data from the ASC file and merges
it into the participant's JSON file. Run from the command line like so:

	python merge_fixation_data.py exp2_left 01

'''

from pathlib import Path
import eyekit
from core import EXP_DATA


def fixation_color(fixation):
	if fixation.has_tag('initial_fixation'):
		return 'deeppink'
	if fixation.has_tag('subsequent_fixation'):
		return 'seagreen'
	return 'black'

def plot_all_trials(user_data_path, out_dir):
	user_data = eyekit.io.load(user_data_path)
	out_dir = out_dir / user_data['user_id']
	if not out_dir.exists():
		out_dir.mkdir()
	screen_width = user_data['screen_width_px']
	screen_height = user_data['screen_height_px']
	trials = user_data['responses']['free_fixation_test'] + user_data['responses']['controlled_fixation_test']
	word_forms = user_data['word_forms']
	for i, (txt, ia, seq) in enumerate(iter_trials(trials, word_forms)):
		img = eyekit.vis.Image(screen_width, screen_height)
		# draw guidelines
		img.draw_line(
			(0, screen_height//2),
			(screen_width, screen_height//2),
			dashed=True, stroke_width=0.5, color='gray'
		)
		img.draw_line(
			(screen_width//2, 0),
			(screen_width//2, screen_height),
			dashed=True, stroke_width=0.5, color='gray'
		)
		img.draw_circle(
			(screen_width//2, screen_height//2),
			radius=18, dashed=True, stroke_width=0.5, color='gray'
		)
		# draw buttons
		for button in user_data['buttons']:
			img.draw_rectangle(button, dashed=True, stroke_width=0.5, color='gray')
		# draw word and fixations
		img.draw_text_block(txt)
		img.draw_rectangle(txt[0::])
		img.draw_fixation_sequence(seq,
			number_fixations=True,
			color=fixation_color,
			saccade_color='black',
			show_discards=True,
			fixation_radius=8,
			stroke_width=1,
			opacity=0.7,
		)
		img.set_crop_area(user_data['presentation_area'])
		img.save(out_dir / f'{i}.pdf', crop_margin=1)
