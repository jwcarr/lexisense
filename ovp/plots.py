from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy import stats

try:
	import mplcairo
	import matplotlib
	matplotlib.use("module://mplcairo.macosx")
except:
	pass

import matplotlib.pyplot as plt
from matplotlib import lines, patches
plt.rcParams.update({'font.sans-serif':'Helvetica Neue', 'font.size': 7})


SCIPY_DISTRIBUTION_FUNCS = {'normal': stats.norm, 'beta':stats.beta}

# Widths of single and double column figures
SINGLE_COLUMN_WIDTH = 3.46 # 88mm
DOUBLE_COLUMN_WIDTH = 7.09 # 180mm


class Figure:

	def __init__(self, file_path, n_rows=1, n_cols=1, width='single', height=None):
		self.file_path = Path(file_path).resolve()
		
		self.n_rows = n_rows
		self.n_cols = n_cols

		if width == 'single':
			self.width = SINGLE_COLUMN_WIDTH
		elif width == 'double':
			self.width = DOUBLE_COLUMN_WIDTH
		else:
			self.width = mm_to_inches(width)

		if height is None:
			self.height = (self.width / self.n_cols) * self.n_rows / (2**0.5)
		else:
			self.height = mm_to_inches(height)
		
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
		self.fig.savefig(self.file_path)
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

def plot_uncertainty(axis, uncertainty_by_position, color=None, show_guidelines=True, show_min=True, show_mean_diff=True):
	axis = ensure_axis(axis)
	word_length = len(uncertainty_by_position)
	positions = list(range(1, word_length+1))
	if show_guidelines:
		_plot_guidelines(axis, positions, uncertainty_by_position, color)
	if show_min:
		_plot_min_uncertainty(axis, uncertainty_by_position, color)
	if show_mean_diff:
		_plot_mean_diff(axis, positions, uncertainty_by_position, color)
	axis.plot(positions, uncertainty_by_position, color=color)
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
	axis.set_ylim(0, 1)
	axis.set_xticks([1, 8, 16, 24, 32, 40, 48, 56, 64])
	axis.set_xlabel('Mini-test trial')
	axis.set_ylabel('Probability of correct response')
	draw_chance_line(axis, 1 / 8)


def plot_test_curve(axis, experiment):
	axis = ensure_axis(axis)
	for condition in experiment.unpack():
		test_curves = []
		for participant in condition:
			test_curves.append(participant.ovp_curve())
		mean_test_curve = sum(test_curves) / len(test_curves)
		word_length = len(mean_test_curve)
		positions = range(1, word_length+1)
		try:
			color = condition.color
		except AttributeError:
			color = 'black'
		axis.plot(positions, mean_test_curve, color=color, linewidth=2)
	padding = (word_length - 1) * 0.05
	axis.set_xlim(1-padding, word_length+padding)
	axis.set_ylim(0, 1)
	axis.set_xticks(positions)
	axis.set_xlabel('Fixation position')
	axis.set_ylabel('Probability of correct response')
	draw_chance_line(axis, 1 / 8)


def plot_prior(axis, experiment, param, unnormalize=False):
	axis = ensure_axis(axis)
	x = np.linspace(*experiment.params[param], 1000)
	x_ = np.linspace(0, 1, 1000) if unnormalize else x
	dist_type, dist_params = experiment.priors[param]
	y = SCIPY_DISTRIBUTION_FUNCS[dist_type](*dist_params).pdf(x_)
	axis.plot(x, y, color='gray', linestyle='--', linewidth=1)
	axis.set_xlabel(f'${param}$')
	axis.set_xlim(*map(round, experiment.params[param]))
	axis.set_yticks([])


def plot_posterior(axis, experiment, param, unnormalize=False):
	axis = ensure_axis(axis)
	x = np.linspace(*experiment.params[param], 1000)
	x_ = np.linspace(0, 1, 1000) if unnormalize else x
	trace = experiment.get_posterior()
	y = stats.gaussian_kde(trace.posterior[param].to_numpy().flatten()).pdf(x_)
	axis.plot(x, y, color=experiment.color, linewidth=1)
	axis.set_xlabel(f'${param}$')
	axis.set_xlim(*map(round, experiment.params[param]))
	axis.set_yticks([])


def plot_posterior_predictive(axis, datasets, condition, lexicon_index=0, show_mean=True, show_veridical=True, show_legend=False):
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
		plot_test_curve(axis, condition)
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




def draw_chance_line(axis, chance):
	start, end = axis.get_xlim()
	axis.autoscale(False)
	axis.plot((start, end), (chance, chance), color='black', linestyle='--', linewidth=1, zorder=0)


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


def mm_to_inches(mm):
	return mm / 25.4
