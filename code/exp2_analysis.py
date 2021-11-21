from collections import defaultdict
from pathlib import Path
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import numpy as np
import eyekit
from core import Figure

asc_variables = ['trial_type', 'target_item', 'word_position_x', 'word_position_y', 'start_word_presentation', 'end_word_presentation', 'trial_abandoned']

def partition_trials_by_type(trials):
	trials_by_type = defaultdict(list)
	for trial in trials:
		if trial['trial_abandoned']:
			continue
		trials_by_type[trial['trial_type']].append(trial)
	return trials_by_type

def merge_fixations_into_user_data(user_data_path, fixation_data_path):
	'''
	Extract fixations from an ASC file and merge them into the user data JSON
	file.
	'''
	user_data = eyekit.io.load(user_data_path)
	extracted_trials = eyekit.io.import_asc(fixation_data_path, variables=asc_variables)
	extracted_trials_by_type = partition_trials_by_type(extracted_trials)
	for trial_type, extracted_trials in extracted_trials_by_type.items():
		for response, fixation_data in zip(user_data['responses'][trial_type], extracted_trials):
			# check that data logged in JSON matches data logged in ASC
			assert response['target_item'] == int(fixation_data['target_item'])
			assert response['word_position'][0] == int(fixation_data['word_position_x'])
			assert response['word_position'][1] == int(fixation_data['word_position_y'])
			# make fixation sequence start at time = 0
			shift = fixation_data['fixations'].shift_start_time_to_zero()
			response['start_word_presentation'] = fixation_data['start_word_presentation'] - shift
			response['end_word_presentation'] = fixation_data['end_word_presentation'] - shift
			# add fixation data to JSON
			response['fixations'] = fixation_data['fixations']
	new_user_data_path = user_data_path.with_name(f'{user_data["user_id"]}_merged.json')
	eyekit.io.save(user_data, new_user_data_path)

def iter_trials(trials, word_forms):
	for trial in trials:
		x, y = trial['word_position']
		word = word_forms[trial['target_item']]
		txt = eyekit.TextBlock(word,
			position=(x, y+20), # +20 because PsychoPy's anchor point is 20px above the baseline
			font_face='Courier New',
			font_size=60,
			align='center',
			autopad=False
		)
		word_ia = txt[0:0:7]
		word_ia.adjust_padding(bottom=-10) # adjust bounding box so that it is central around the letters
		for fixation in trial['fixations']:
			if fixation in word_ia:
				if fixation.start >= trial['start_word_presentation'] and fixation.start < trial['end_word_presentation']:

					if fixation.end <= trial['end_word_presentation']:
						fixation.add_tag('in_time')
					else:
						fixation.add_tag('over_time')
						# fixation.end = trial['end_word_presentation'] # clip fixation to end of of word presentation

				else:
					fixation.discard()
			else:
				fixation.discard()

		found_first = False
		for fixation in trial['fixations'].iter_without_discards():
			if found_first is False:
				fixation.add_tag('initial_fixation')
				found_first = True
			else:
				fixation.add_tag('subsequent_fixation')

		yield txt, word_ia, trial['fixations']

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
	trials = user_data['responses']['free_fixation_test']
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

def get_landing_positions(trials, word_forms, initial_only=False):
	initial_positions = []
	remaining_positions = []
	for txt, ia, seq in iter_trials(trials, word_forms):
		positions = eyekit.measure.landing_distances(ia, seq)
		if len(positions) > 0:
			initial_positions.append(positions[0])
		if len(positions) > 1:
			remaining_positions.extend(positions[1:])
	x = np.linspace(0, 252, 100)
	initial_distribution = gaussian_kde(initial_positions).pdf(x)
	initial_distribution /= initial_distribution.max()
	if initial_only:
		return x, initial_distribution
	remaining_distribution = gaussian_kde(remaining_positions).pdf(x)
	remaining_distribution /= remaining_distribution.max()
	return x, initial_distribution, remaining_distribution

def setup_figure_gridlines(fig):
	for axis in fig:
		for i in range(1, 7):
			axis.plot([i*36, i*36], [-0.05, 1.05], color='black', linestyle='--', linewidth=0.5)
			axis.set_xlim(0, 252)
			axis.set_ylim(-0.05, 1.05)
			axis.set_xticks([i*36 for i in range(8)])
			axis.set_yticks([])	

def plot_landing_distribution(user_data_path):
	user_data = eyekit.io.load(user_data_path)
	long_trials = user_data['responses']['free_fixation_test'][:64]
	short_trials = user_data['responses']['free_fixation_test'][64:]

	with Figure(f'../visuals/{user_data["task_id"]}/{user_data["user_id"]}.pdf', 1, 2, width='double') as fig:

		setup_figure_gridlines(fig)

		x, distribution = get_landing_positions(short_trials, user_data['word_forms'], initial_only=True)

		fig[0,0].plot(x, distribution, color='deeppink')
		fig[0,0].set_title('Short presentation (50 ms)')
		fig[0,0].set_xlabel('Fixation position (pixels)')
		fig[0,0].set_ylabel('Density')

		x, distribution_1, distribution_2 = get_landing_positions(long_trials, user_data['word_forms'], initial_only=False)

		fig[0,1].plot(x, distribution_1, color='deeppink')
		fig[0,1].plot(x, distribution_2, color='seagreen', linestyle='--')
		fig[0,1].set_title('Long presentation (500 ms)')
		fig[0,1].set_xlabel('Fixation position (pixels)')
		fig[0,1].set_ylabel('Density')


if __name__ == '__main__':

	# merge_fixations_into_user_data(
	# 	Path('../data/experiments/lab/pilot_exp2_left/01.json'),
	# 	Path('../data/experiments/lab/pilot_exp2_left/01.asc')
	# )
	# quit()

	# plot_all_trials(
	# 	Path('../data/experiments/lab/pilot_exp2_left/01_merged.json'),
	# 	Path('../visuals/pilot_exp2_left/')
	# )
	# quit()

	plot_landing_distribution(
		Path('../data/experiments/lab/pilot_exp2_right/01_merged.json')
	)
	quit()
