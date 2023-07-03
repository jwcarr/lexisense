from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy import stats
import arviz as az
import eyekit

# try:
# 	import mplcairo
# 	import matplotlib
# 	matplotlib.use("module://mplcairo.macosx")
# except:
# 	pass

import matplotlib.pyplot as plt
from matplotlib import lines, patches
plt.rcParams.update({'font.sans-serif': 'Helvetica Neue', 'font.size': 7})


# Create easily accessible distribution functions that are named and
# parameterized in the same way as PyMC.
SCIPY_DISTRIBUTION_FUNCS = {
	'normal': stats.norm,
	'beta': stats.beta,
	'gamma': lambda mu, sigma: stats.gamma(mu**2/sigma**2, scale=1/(mu/sigma**2)),
	'exponential': lambda lam: stats.expon(scale=1/lam),
	'uniform': lambda lower, upper: stats.uniform(loc=lower, scale=upper - lower),
}


# Widths of single and double column figures
SINGLE_COLUMN_WIDTH = 3.4
DOUBLE_COLUMN_WIDTH = 7.0


class Figure:

	def __init__(self, file_path=None, n_rows=1, n_cols=1, width='single', height=None, units='mm'):
		if file_path is None:
			self.file_path = None
		else:
			self.file_path = Path(file_path).resolve()
		
		self.n_rows = n_rows
		self.n_cols = n_cols

		if width == 'single':
			self.width = SINGLE_COLUMN_WIDTH
		elif width == 'double':
			self.width = DOUBLE_COLUMN_WIDTH
		else:
			if units == 'mm':
				self.width = mm_to_inches(width)
			else:
				self.width = width

		if height is None:
			self.height = (self.width / self.n_cols) * self.n_rows / (2**0.5)
		else:
			if units == 'mm':
				self.height = mm_to_inches(height)
			else:
				self.height = height
		
		self.auto_deduplicate_axes = True

	def __enter__(self):
		self.fig, self.axes = plt.subplots(self.n_rows, self.n_cols, figsize=(self.width, self.height), squeeze=False)
		self.used = [[False]*self.n_cols for _ in range(self.n_rows)]
		return self

	def __exit__(self, exc_type, exc_value, exc_traceback):
		if self.auto_deduplicate_axes:
			self.deduplicate_axes()
		self.turn_off_unused_axes()
		self.fig.tight_layout(pad=0.5, h_pad=1, w_pad=1)
		if self.file_path is None:
			plt.show()
		else:
			if exc_type:
				file_path = self.file_path.parent / f'{self.file_path.stem}_fail{self.file_path.suffix}'
			else:
				file_path = self.file_path
			if not file_path.parent.exists():
				file_path.parent.mkdir(parents=True)
			self.fig.savefig(file_path)
		plt.close(self.fig)

	def __getitem__(self, index):
		i, j = index
		if i >= self.n_rows or j >= self.n_cols:
			raise IndexError(f'Invalid axis index, Figure has {self.n_rows} rows and {self.n_cols} columns')
		self.used[i][j] = True
		return self.axes[i,j]

	def __iter__(self):
		for i in range(self.n_rows):
			for j in range(self.n_cols):
				yield self[i,j]

	def deduplicate_axes(self):
		for row in self.axes:
			if len(set([cell.get_ylabel() for cell in row])) == 1:
				for i in range(1, self.n_cols):
					row[i].set_ylabel('')
			if len(set([str(cell.get_yticks()) for cell in row])) == 1:
				for i in range(1, self.n_cols):
					row[i].set_yticklabels([])
		for col in self.axes.T:
			if len(set([cell.get_xlabel() for cell in col])) == 1:
				for i in range(self.n_rows-1):
					col[i].set_xlabel('')
			if len(set([str(cell.get_xticks()) for cell in col])) == 1:
				for i in range(self.n_rows-1):
					col[i].set_xticklabels([])

	def turn_off_unused_axes(self):
		for i, row in enumerate(self.axes):
			for j, cell in enumerate(row):
				if not self.used[i][j]:
					cell.axis('off')

	def unpack(self):
		return (axis for axis in self)

	def unpack_row(self, row_i):
		return (self[row_i, j] for j in range(self.n_cols))

	def unpack_column(self, col_j):
		return (self[i, col_j] for i in range(self.n_rows))


def ensure_axis(axis):
	if isinstance(axis, Figure):
		return axis[0, 0]
	if isinstance(axis, plt.Axes):
		return axis
	raise ValueError(f'Expected Figure or plt.Axes, got {axis.__class__.__name__}')


def _plot_guidelines(axis, fixation_positions, mean_uncertainty, color):
	middle = len(fixation_positions) // 2 + 1
	first_x = fixation_positions[:middle]
	first_y = mean_uncertainty[:middle]
	last_x = fixation_positions[middle-1:][::-1]
	last_y = mean_uncertainty[middle-1:][::-1]
	for i in range(len(last_x)-(len(fixation_positions)%2)):
		axis.plot([first_x[i], last_x[i]], [first_y[i], last_y[i]], c=color, linestyle=':', linewidth=.5)

def _plot_min_uncertainty(axis, uncertainty_by_position, color):
	ovp = np.argmin(uncertainty_by_position) + 1
	marker_width = (len(uncertainty_by_position) - 1) / 40
	marker_height = 0.4
	triangle = [(ovp, -0.07), (ovp-marker_width, -marker_height), (ovp+marker_width, -marker_height)]
	axis.add_patch(patches.Polygon(triangle, color=color, clip_on=False, closed=True, zorder=10))

def _plot_mean_diff(axis, fixation_positions, mean_uncertainty, color):
	middle = len(fixation_positions) // 2 + 1
	left_uncertainties = mean_uncertainty[:middle-1]
	right_uncertainties = list(reversed(mean_uncertainty[middle-1:]))[:len(left_uncertainties)]
	uncertainty_reduction = np.mean(right_uncertainties) - np.mean(left_uncertainties)
	if color == 'black':
		axis.text(1, 0.5, str(round(uncertainty_reduction, 2)), color=color, ha='left', fontsize=6)
	else:
		axis.text(len(fixation_positions), 0.5, str(round(uncertainty_reduction, 2)), color=color, ha='right', fontsize=6)

def plot_uncertainty(axis, uncertainty_by_position, color=None, show_guidelines=True, show_min=False, show_mean_diff=False, linewidth=1, label=None):
	axis = ensure_axis(axis)
	word_length = len(uncertainty_by_position)
	positions = list(range(1, word_length+1))
	if show_guidelines:
		_plot_guidelines(axis, positions, uncertainty_by_position, color)
	if show_min:
		_plot_min_uncertainty(axis, uncertainty_by_position, color)
	if show_mean_diff:
		_plot_mean_diff(axis, positions, uncertainty_by_position, color)
	axis.plot(positions, uncertainty_by_position, color=color, linewidth=linewidth, label=label)
	xpad = (word_length - 1) * 0.05
	axis.set_xlim(1-xpad, word_length+xpad)
	axis.set_xticks(range(1, word_length+1))
	axis.set_xticklabels(range(1, word_length+1))
	axis.set_xlabel('Fixation position')
	axis.set_ylabel('Uncertainty (bits)')


def plot_learning_scores(axis, experiment):
	axis = ensure_axis(axis)
	learning_scores_left = defaultdict(int)
	learning_scores_right = defaultdict(int)
	for user in experiment.left.iter_with_excludes():
		score = user.learning_score(experiment.left.n_last_trials)
		learning_scores_left[score] += 1
	for user in experiment.right.iter_with_excludes():
		score = user.learning_score(experiment.right.n_last_trials)
		learning_scores_right[score] += 1
	left_x, left_y = zip(*learning_scores_left.items())
	right_x, right_y = zip(*learning_scores_right.items())
	axis.bar(left_x, left_y, label='Left-heavy', color=experiment.left.color)
	axis.bar(right_x, right_y, label='Right-heavy', color=experiment.right.color,
		bottom=[learning_scores_left[x] for x in right_x])
	axis.set_xlim(-0.5, experiment.left.n_last_trials+0.5)
	axis.set_xticks(range(experiment.left.n_last_trials+1))
	axis.set_ylabel('Number of participants')
	axis.set_xlabel(f'Number of correct responses during final {experiment.left.n_last_trials} training rounds')
	axis.legend(frameon=False)
	exclusions = [learning_scores_left[k] + learning_scores_right[k] for k in range(experiment.left.min_learning_score)]
	if sum(exclusions) > 0:
		draw_brace(axis, (0, experiment.left.min_learning_score-1), max(exclusions), 'Excluded')


def plot_learning_curve(axis, experiment, n_previous_trials=8):
	axis = ensure_axis(axis)
	for condition in experiment.unpack():
		learning_curves = []
		for participant in condition:
			learning_curves.append(participant.learning_curve(n_previous_trials))
		mean_learning_curve = sum(learning_curves) / len(learning_curves)
		x_vals = range(n_previous_trials, len(mean_learning_curve) + n_previous_trials)
		try:
			color = condition.color
		except AttributeError:
			color = 'black'
		axis.plot(x_vals, mean_learning_curve, color=color)
	padding = (64 - n_previous_trials) * 0.05
	axis.set_xlim(n_previous_trials - padding, 64 + padding)
	axis.set_ylim(-0.05, 1.05)
	axis.set_xticks([1, 8, 16, 24, 32, 40, 48, 56, 64])
	axis.set_xlabel('Mini-test trial')
	axis.set_ylabel('Probability of correct response')
	draw_chance_line(axis, 1 / 8)

def jitter(array):
	mn = array.max()
	mx = array.min()
	jitter_scale = (mx - mn) * 0.05
	jitter = (np.random.random(len(array)) - 0.5) * jitter_scale
	return array + jitter

def plot_test_curve(axis, experiment, show_individuals=True, add_jitter=False):
	axis = ensure_axis(axis)
	for condition in experiment.unpack():
		try:
			word_length = condition['n_letters']
		except TypeError:
			word_length = 7
		positions = range(1, word_length+1)
		test_curves = []
		for participant in condition:
			test_curve = participant.ovp_curve()
			if show_individuals:
				if add_jitter:
					plotted_test_curve = jitter(test_curve)
				else:
					plotted_test_curve = test_curve
				try:
					color = condition.light_color
				except AttributeError:
					color = 'black'
				axis.plot(positions, plotted_test_curve, color=color, linewidth=0.5)
			test_curves.append(test_curve)
		mean_test_curve = sum(test_curves) / len(test_curves)
		try:
			color = condition.color
		except AttributeError:
			color = 'black'
		axis.plot(positions, mean_test_curve, color=color, linewidth=2)
	padding = (word_length - 1) * 0.05
	axis.set_xlim(1-padding, word_length+padding)
	axis.set_ylim(-0.05, 1.05)
	axis.set_xticks(positions)
	axis.set_xlabel('Fixation position')
	axis.set_ylabel('Probability of correct response')
	draw_chance_line(axis, 1 / 8)


def plot_word_inferences(axis, experiment, target_item, fixation_position):
	axis = ensure_axis(axis)
	for condition in experiment.unpack():
		n_correct = np.zeros(8, dtype=int)
		# n_trials = np.zeros(8, dtype=int)
		for participant in condition:
			for trial in participant.iter_controlled_fixation_trials():
				if trial['fixation_position'] == fixation_position and trial['target_item'] == target_item:
					n_correct[ trial['selected_item'] ] += 1
				# n_trials[ trial['selected_item'] ] += 1
		print(n_correct)
		# print(n_trials)
		axis.plot(n_correct / n_correct.sum(), condition.color)
	axis.set_ylim(-0.05, 1.05)


def plot_landing_curve(axis, experiment, show_average=False, show_individuals=True, letter_width=36, n_letters=7):
	axis = ensure_axis(axis)
	word_width = letter_width * n_letters
	x = np.linspace(0, word_width, 1000)
	for condition in experiment.unpack():
		landing_x_all = []
		for participant in condition:
			landing_x = participant.landing_positions()
			landing_x_all.extend(landing_x)
			if show_individuals:
				y = density(landing_x, x)
				try:
					color = condition.light_color
				except AttributeError:
					color = 'black'
				axis.plot(x, y, color=color, linewidth=0.5)
		if show_average:
			y = density(landing_x_all, x)
			axis.plot(x, y, color=condition.color, linewidth=2)
	axis.set_yticks([])
	axis.set_xlabel('Landing position (pixels)')
	draw_letter_grid(axis, letter_width, n_letters)


def plot_landing_curve_fits(axis, experiment, show_individuals=True, letter_width=36, n_letters=7):
	axis = ensure_axis(axis)
	trace = experiment.get_posterior()
	x = np.linspace(0, letter_width * n_letters, 1000)
	for k, condition in enumerate(experiment.unpack()):
		condition_index = condition.ID.split('_')[1][0]
		for j in range(condition['n_participants']):
			try:
				μ_samples = trace.posterior[f'μ_{condition_index}'][:, :, j]
				σ_samples = trace.posterior[f'σ_{condition_index}'][:, :, j]
			except IndexError:
				break
			μ, σ = float(μ_samples.mean()), float(σ_samples.mean())
			y = SCIPY_DISTRIBUTION_FUNCS['normal'](μ, σ).pdf(x)
			axis.plot(x, y, color=condition.light_color, linewidth=0.5)
		τ = float(trace.posterior.τ[:, :, k].mean())
		δ = float(trace.posterior.δ[:, :, k].mean())
		y = SCIPY_DISTRIBUTION_FUNCS['normal'](τ, δ).pdf(x)
		axis.plot(x, y, color=condition.color, linewidth=2, zorder=10)
	axis.set_yticks([])
	axis.set_xlabel('Landing position (pixels)')
	draw_letter_grid(axis, letter_width=36, n_letters=7)


def plot_prior(axis, experiment, param, transform_to_param_bounds=False, label=None):
	axis = ensure_axis(axis)
	if param in experiment.priors:
		dist_type, dist_params = experiment.priors[param]
		x = np.linspace(*experiment.params[param], 1000)
		x_ = np.linspace(0, 1, 1000) if transform_to_param_bounds else x
		y = SCIPY_DISTRIBUTION_FUNCS[dist_type](*dist_params).pdf(x_)
		axis.plot(x, y / y.sum(), color=experiment.light_color, linestyle='--', linewidth=1, label=label)
	else:
		for condition in experiment.unpack():
			dist_type, dist_params = condition.priors[param]
			x = np.linspace(*condition.params[param], 1000)
			x_ = np.linspace(0, 1, 1000) if transform_to_param_bounds else x
			y = SCIPY_DISTRIBUTION_FUNCS[dist_type](*dist_params).pdf(x_)
			axis.plot(x, y / y.sum(), color=condition.light_color, linestyle='--', linewidth=1, label=label)
	axis.set_xlabel(f'${param}$')
	axis.set_xlim(*map(round, experiment.params[param]))
	axis.set_yticks([])


def plot_posterior(axis, experiment, param, linestyle='-', posterior_file=None, label=None, hdi=None):
	axis = ensure_axis(axis)
	trace = experiment.get_posterior(posterior_file)
	x = np.linspace(*experiment.params[param], 1000)
	if len(trace.posterior[param].shape) == 3:
		for i, condition in enumerate(experiment.unpack()):
			y = stats.gaussian_kde(trace.posterior[param][:,:,i].to_numpy().flatten()).pdf(x)
			axis.plot(x, y / y.sum(), color=condition.color, linewidth=1, linestyle=linestyle, label=label)
	else:
		samples = trace.posterior[param].to_numpy().flatten()
		y = stats.gaussian_kde(samples).pdf(x)
		axis.plot(x, y / y.sum(), color=experiment.color, linewidth=1, linestyle=linestyle, label=label)
		if hdi:
			az_hdi = az.hdi(trace.posterior, hdi_prob=hdi)
			lower, upper = float(az_hdi[param][0]), float(az_hdi[param][1])
			draw_hdi(axis, lower, upper, hdi)
			# if show_mean_and_ci:
			# 	mean = float(samples.mean())
			# 	draw_mean_and_ci(axis, mean, lower, upper)

	axis.set_xlabel(f'${param}$')
	axis.set_xlim(*map(round, experiment.params[param]))
	axis.set_yticks([])



def plot_posterior_difference(axis, experiment, param, hdi=None, rope=None, show_mean_and_ci=True, xlim=None, linestyle='-', posterior_file=None):
	axis = ensure_axis(axis)
	trace = experiment.get_posterior(posterior_file)
	diff_param = f'Δ({param})'
	diff_samples = trace.posterior[diff_param]
	if xlim:
		mn, mx = xlim
	else:
		if rope:
			mn = min(diff_samples.min(), rope[0])
		else:
			mn = diff_samples.min()
		mx = diff_samples.max()
	x = np.linspace(mn, mx, 1000)
	y = stats.gaussian_kde(diff_samples.to_numpy().flatten()).pdf(x)
	axis.plot(x, y / y.sum(), color='black', linewidth=1, linestyle=linestyle)
	axis.set_xlabel(f'Δ(${param}$)')
	axis.set_xlim(mn, mx)
	axis.set_yticks([])
	if hdi:
		az_hdi = az.hdi(diff_samples, hdi_prob=hdi)
		lower, upper = float(az_hdi[diff_param][0]), float(az_hdi[diff_param][1])
		draw_hdi(axis, lower, upper, hdi)
		if show_mean_and_ci:
			mean = float(diff_samples.mean())
			draw_mean_and_ci(axis, mean, lower, upper)
	if rope:
		draw_rope(axis, max(mn, rope[0]), rope[1])


def draw_params(posterior_samples, n_draws=100):
	n_samples = len(posterior_samples[0])
	drawn_params = []
	for draw_i in np.random.randint(0, n_samples, n_draws):
		drawn_params.append([samples[draw_i] for samples in posterior_samples])
	return drawn_params


def plot_landing_posterior_predictive(axis, experiment, n_samples=100):
	axis = ensure_axis(axis)
	trace = experiment.get_posterior()
	x = np.linspace(0, 252, 1000)
	for i, condition in enumerate(experiment.unpack()):
		τ_samples = trace.posterior.τ[:, :, i+2].to_numpy().flatten()
		δ_samples = trace.posterior.δ[:, :, i+2].to_numpy().flatten()
		for τ, δ in draw_params([τ_samples, δ_samples]):
			y = stats.norm(τ, δ).pdf(x)
			axis.plot(x, y, color=condition.color, alpha=0.1)
	axis.set_xlabel('Landing position (pixels)')
	axis.set_yticks([])
	draw_letter_grid(axis, letter_width=36, n_letters=7)


def plot_posterior_predictive(axis, datasets, condition, lexicon_index=0, show_mean=True, show_veridical=True, show_legend=False):
	axis = ensure_axis(axis)
	word_length = condition['n_letters']
	positions = list(range(1, word_length + 1))
	test_curves = []
	for dataset in datasets:
		test_curve = convert_dataset_to_ovp_curves(dataset, lexicon_index, word_length)
		test_curves.append(test_curve)
		axis.plot(positions, test_curve, color=condition.light_color, linewidth=0.5)
	if show_mean:
		mean_test_curve = sum(test_curves) / len(test_curves)
		axis.plot(positions, mean_test_curve, color=condition.color, linewidth=1, linestyle='--')
	if show_veridical:
		plot_test_curve(axis, condition, show_individuals=False)
	padding = (word_length - 1) * 0.05
	axis.set_xlim(1-padding, word_length+padding)
	axis.set_ylim(0.5, 1)
	axis.set_xticks(positions)
	axis.set_xlabel('Fixation position')
	axis.set_ylabel('Probability of correct response')
	draw_chance_line(axis, 1 / 8)
	if show_legend:
		handles = [
			lines.Line2D([0], [0], color='gray', linestyle='-', linewidth=0.5, label='Simulated runs of the experiment')
		]
		if show_mean:
			handles.append(
				lines.Line2D([0], [0], color='black', linestyle='--', linewidth=1, label='Mean of simulated runs')
			)
		if show_veridical:
			handles.append(
				lines.Line2D([0], [0], color='black', linestyle='-', linewidth=2, label='Actual experimental results')
			)
		labels = [h.get_label() for h in handles]
		axis.legend(handles=handles, labels=labels, frameon=False)

def convert_dataset_to_ovp_curves(dataset, lexicon_index, word_length):
	correct = np.zeros(word_length)
	n_trials = np.zeros(word_length)
	for l, t, j, w in dataset:
		if l == lexicon_index:
			correct[j] += t == w
			n_trials[j] += 1
	return correct / n_trials


def make_trial_image(participant, trial):
	screen_width = participant['screen_width_px']
	screen_height = participant['screen_height_px']
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
	selected_button_i = participant['object_array'].index(trial['selected_item'])
	correct_button_i = participant['object_array'].index(trial['target_item'])
	for i, button in enumerate(participant['buttons']):
		if i == correct_button_i:
			img.draw_rectangle(button, dashed=True, stroke_width=1, color='green')
		if i == selected_button_i and selected_button_i != correct_button_i:
			img.draw_rectangle(button, dashed=True, stroke_width=1, color='red')
		else:
			img.draw_rectangle(button, dashed=True, stroke_width=0.5, color='gray')
	# draw word and fixations
	word_ia = trial['word'][0:0:7]
	img.draw_text_block(trial['word'])
	img.draw_rectangle(word_ia)

	def fixation_color_func(fixation):
		if fixation in word_ia:
			if fixation.start >= trial['start_word_presentation'] and fixation.start < trial['end_word_presentation']:
				return 'blue'
		return 'black'
	
	img.draw_fixation_sequence(trial['fixations'],
		number_fixations=True,
		color=fixation_color_func,
		saccade_color='black',
		show_discards=True,
		fixation_radius=8,
		stroke_width=1,
		opacity=0.7,
	)
	img.set_crop_area(participant['presentation_area'])
	return img


def landing_position_image(experiment, file_path):
	image = eyekit.vis.Image(1920, 1080)
	for condition, word_y in zip(experiment.unpack(), [500, 600]):
		positions = []
		for participant in condition:
			for trial in participant.iter_free_fixation_trials():
				seq = trial['fixations']
				word_ia = trial['word'][0:0:7]
				for fixation in seq:
					if fixation.start > trial['start_word_presentation'] and fixation.start < trial['end_word_presentation']:
						if fixation in word_ia:
							x = int(fixation.x - word_ia.x_tl)
							y = int(fixation.y - word_ia.y_tl)
							positions.append((x, y))
							break
		centered_word_txt = eyekit.TextBlock(
			'SXXXXXS',
			position=(1000, word_y),
			font_face='Courier New',
			font_size=60,
			align='center',
			autopad=False,
		)
		centered_word = centered_word_txt[0:0:7]
		centered_word.adjust_padding(top=10)
		image.draw_text_block(centered_word_txt)
		for x, y in positions:
			x_ = x + centered_word.x_tl
			y_ = y + centered_word.y_tl
			image.draw_circle((x_, y_), 2, fill_color=condition.color, opacity=0.3)
	image.save(file_path, crop_margin=10)


def landing_saccade_image(experiment, file_path):
	image = eyekit.vis.Image(1920, 1080)
	for condition, word_y in zip(experiment.unpack(), [500, 600]):
		positions = []
		for participant in condition:
			for trial in participant.iter_free_fixation_trials():
				seq = trial['fixations']
				word_ia = trial['word'][0:0:7]
				for prev_fixation, fixation in seq.iter_pairs():
					if fixation.start > trial['start_word_presentation'] and fixation.start < trial['end_word_presentation']:
						if fixation in word_ia:
							x = int(fixation.x - word_ia.x_tl)
							y = int(fixation.y - word_ia.y_tl)
							x_ = int(prev_fixation.x - word_ia.x_tl)
							y_ = int(prev_fixation.y - word_ia.y_tl)
							positions.append((x, y, x_, y_))
							break
		centered_word_txt = eyekit.TextBlock(
			'SXXXXXS',
			position=(1000, word_y),
			font_face='Courier New',
			font_size=60,
			align='center',
			autopad=False,
		)
		centered_word = centered_word_txt[0:0:7]
		centered_word.adjust_padding(top=10)
		image.draw_text_block(centered_word_txt)
		for x, y, prev_x, prev_y in positions:
			x_ = x + centered_word.x_tl
			y_ = y + centered_word.y_tl
			prev_x_ = prev_x + centered_word.x_tl
			prev_y_ = prev_y + centered_word.y_tl
			image.draw_line((x_, y_), (prev_x_, prev_y_), color=condition.color, opacity=0.5, stroke_width=0.1)
	image.save(file_path)


def draw_chance_line(axis, chance):
	start, end = axis.get_xlim()
	axis.autoscale(False)
	axis.plot((start, end), (chance, chance), color='gray', linestyle=':', linewidth=1, zorder=0)


def draw_brace(axis, xspan, yy, text):
	xmin, xmax = xspan
	xspan = xmax - xmin
	ax_xmin, ax_xmax = axis.get_xlim()
	xax_span = ax_xmax - ax_xmin

	ymin, ymax = axis.get_ylim()
	yspan = ymax - ymin
	resolution = int(xspan/xax_span*100)*2+1 # guaranteed uneven
	beta = 300./xax_span # the higher this is, the smaller the radius

	x = np.linspace(xmin, xmax, resolution)
	x_half = x[:int(resolution/2)+1]
	y_half_brace = (1/(1.+np.exp(-beta*(x_half-x_half[0])))
					+ 1/(1.+np.exp(-beta*(x_half-x_half[-1]))))
	y = np.concatenate((y_half_brace, y_half_brace[-2::-1]))
	y = yy + (.05*y - .01)*yspan # adjust vertical position

	axis.autoscale(False)
	axis.plot(x, y, color='black', lw=1)

	axis.text((xmax+xmin)/2., yy+.07*yspan, text, ha='center', va='bottom')


def draw_letter_grid(axis, letter_width, n_letters):
	word_width = letter_width * n_letters
	letter_grid = list(range(0, word_width + 1, letter_width))
	axis.set_xlim(0, word_width)
	axis.set_xticks(letter_grid)
	axis.grid(axis='x')


def draw_hdi(axis, lower, upper, hdi_prob):
	mn_y, mx_y  = axis.get_ylim()
	padding = (mx_y - mn_y) * 0.1
	axis.plot((lower, upper), (0, 0), color='MediumSeaGreen')
	hdi_text = f'{int(hdi_prob*100)}% HDI'
	hdi_width = round(upper - lower, 2)
	# hdi_text = f'← {hdi_width} →'
	# axis.text((lower + upper)/2, mn_y + padding, hdi_text, ha='center', color='MediumSeaGreen', font='Arial')


def draw_mean_and_ci(axis, mean, lower, upper):
	mn_y, mx_y  = axis.get_ylim()
	y_pos = (mx_y - mn_y) * 0.2
	axis.text(lower, y_pos, str(round(lower, 1)), ha='right', color='MediumSeaGreen')
	axis.text(mean, y_pos, str(round(mean, 1)), ha='center', color='MediumSeaGreen')
	axis.text(upper, y_pos, str(round(upper, 1)), ha='left', color='MediumSeaGreen')


def draw_rope(axis, lower, upper):
	axis.axvspan(lower, upper, color='#DDDDDD', lw=0)
	mn_y, mx_y  = axis.get_xlim()
	h_padding = (mx_y - mn_y) * 0.05
	x = lower + h_padding
	mn_y, mx_y  = axis.get_ylim()
	v_padding = (mx_y - mn_y) * 0.05
	y = mx_y - 2 * v_padding
	# axis.text(0, y, 'ROPE', ha='center')


def mm_to_inches(mm):
	return mm / 25.4


def density(samples, x, normalize=True):
	y = stats.gaussian_kde(samples).pdf(x)
	if normalize:
		return y / y.sum()
	return y

def unify_scales(axes):
	global_mn = np.inf
	global_mx = -np.inf
	for axis in axes:
		mn, mx = axis.get_ylim()
		if mn < global_mn:
			global_mn = mn
		if mx > global_mx:
			global_mx = mx
	for axis in axes:
		axis.set_ylim(global_mn, global_mx)
