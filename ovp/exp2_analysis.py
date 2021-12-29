from pathlib import Path
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import numpy as np
import eyekit
from core import Figure


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
		word_ia.adjust_padding(top=10) # adjust bounding box so that it is central around the letters
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
	# long_trials = user_data['responses']['free_fixation_test'][:64]
	# short_trials = user_data['responses']['free_fixation_test'][64:]
	short_trials = user_data['responses']['free_fixation_test']

	with Figure(f'../visuals/{user_data["task_id"]}/{user_data["user_id"]}.pdf', 1, 2, width='double') as fig:

		setup_figure_gridlines(fig)

		x, distribution = get_landing_positions(short_trials, user_data['word_forms'], initial_only=True)

		fig[0,0].plot(x, distribution, color='deeppink')
		fig[0,0].set_title('Short presentation (50 ms)')
		fig[0,0].set_xlabel('Fixation position (pixels)')
		fig[0,0].set_ylabel('Density')

		# x, distribution_1, distribution_2 = get_landing_positions(long_trials, user_data['word_forms'], initial_only=False)

		# fig[0,1].plot(x, distribution_1, color='deeppink')
		# fig[0,1].plot(x, distribution_2, color='seagreen', linestyle='--')
		# fig[0,1].set_title('Long presentation (500 ms)')
		# fig[0,1].set_xlabel('Fixation position (pixels)')
		# fig[0,1].set_ylabel('Density')

def plot_landing_distribution_all(data_path_left, data_path_right):
	with Figure(f'../visuals/pilot_all.pdf', 1, 2, width='double') as fig:

		setup_figure_gridlines(fig)

		for path in [data_path_left, data_path_right]:

			dist = np.zeros(100)
			n_subjects = 0
			for user_id in range(1, 31):
				user_id = str(user_id).zfill(2)
				user_data_path = path / f'{user_id}.json'
				if not user_data_path.exists():
					continue
				user_data = eyekit.io.load(user_data_path)
				short_trials = user_data['responses']['free_fixation_test'][64:]
				try:
					x, distribution = get_landing_positions(short_trials, user_data['word_forms'], initial_only=True)
				except ValueError:
					continue
				dist += distribution
				n_subjects += 1
			dist /= n_subjects

			fig[0,0].plot(x, dist, color='cadetblue')
			fig[0,0].set_title('Short presentation (50 ms)')
			fig[0,0].set_xlabel('Fixation position (pixels)')
			fig[0,0].set_ylabel('Density')


if __name__ == '__main__':

	plot_all_trials(
		Path('../data/experiments/pilot3_right/05.json'),
		Path('../visuals/pilot3_right/')
	)
	quit()

	# plot_landing_distribution(
	# 	Path('../data/experiments/pilot3_right/05.json')
	# )
	# quit()

	plot_landing_distribution_all(
		Path('../data/experiments/pilot3_left/'),
		Path('../data/experiments/pilot3_right/')
	)
	quit()
