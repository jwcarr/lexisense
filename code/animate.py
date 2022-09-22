import io
from pathlib import Path
import numpy as np
import imageio.v3 as iio
import eyekit


def mode(values):
	values = list(values)
	return max(set(values), key=values.count)


def create_frames_from_samples(samples, start_word_presentation, end_word_presentation, sample_rate=1000, fps=25):
	samples = np.array(samples, dtype=int)
	s = start_word_presentation - samples[0, 0]
	e = end_word_presentation - samples[0, 0]
	samples[ :s, 0] = 1 # presentation of fixation point
	samples[s:e, 0] = 2 # presentation of word
	samples[e: , 0] = 3 # presentation of response array
	ms_per_frame = sample_rate // fps
	n_frames = len(samples) // ms_per_frame
	n_samples = n_frames * ms_per_frame
	frames = np.zeros((n_frames, 3), dtype=int)
	for frame_i, sample_i in enumerate(range(0, n_samples, ms_per_frame)):
		frame = samples[sample_i : sample_i + ms_per_frame]
		frames[frame_i, 0] = mode(frame[:, 0])
		frames[frame_i, 1] = np.mean(frame[:, 1])
		frames[frame_i, 2] = np.mean(frame[:, 2])
	return frames


def build_animation(frames, word, bg_image, presentation_area, output_path, show_trail=False, fps=25):
	images = []
	trail = []
	for i, (screen, x, y) in enumerate(frames):
		frame = eyekit.vis.Image(1920, 1080)
		frame.set_crop_area(presentation_area)
		if screen == 1: # fixation point
			frame.draw_circle((1920//2, 1080//2), 6, stroke_width=2)
		elif screen == 2: # word
			frame.draw_text_block(word)
		elif screen == 3: # response array
			frame.set_background_image(bg_image)
		if show_trail:
			for x, y in trail:
				frame.draw_circle((x, y), radius=10, fill_color='crimson', stroke_width=0)
		if screen > 0: # if screen == 0, this frame is during a blink
			frame.draw_circle((x, y), radius=10, fill_color='crimson', stroke_width=0)
			trail.append((x, y))
		images.append(iio.imread(frame.render_frame()))
	iio.imwrite(output_path, images,
		fps=fps,
		# quality=9.
		loop=0,
		palettesize=128,
		subrectangles=True,
	)


def make_trial_animation(condition_id, participant_id, trial_id, show_trail=False):
	data_path = Path(f'../data/experiments/{condition_id}/{participant_id}')
	word = eyekit.io.load(data_path.with_suffix('.json'))['responses']['free_fixation_test'][trial_id]['word']
	dataset = eyekit.io.import_asc(data_path.with_suffix('.asc'), import_samples=True,
		variables=['start_word_presentation', 'end_word_presentation', 'trial_abandoned']
	)
	dataset = [trial for trial in dataset if not trial['trial_abandoned']]
	trial = dataset[trial_id]
	frames = create_frames_from_samples(
		trial['samples'],
		trial['start_word_presentation'],
		trial['end_word_presentation'],
	)
	build_animation(
		frames=frames,
		word=word,
		bg_image=Path('../private/ani_bg_image.png'),
		presentation_area=[480, 270, 960, 540],
		output_path=f'../results/exp2/animations/{condition_id}_{participant_id}_{trial_id}.gif',
		show_trail=show_trail,
	)
