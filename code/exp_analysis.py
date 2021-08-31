from collections import defaultdict
import pickle
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.stats import beta, gaussian_kde
from statsmodels.stats import proportion
import core


DATA_DIR = core.EXP_DATA


class Experiment:

	def __init__(self, experiment_id):
		self.experiment_id = experiment_id
		self.left = Task(experiment_id + '_left', 'cadetblue', '#AFD0D0')
		self.right = Task(experiment_id + '_right', 'crimson', '#F58CA1')
		self.color = 'black'
		self.light_color = 'gray'

	def __iter__(self):
		for task in [self.left, self.right]:
			yield task

	@property
	def id(self):
		return self.experiment_id

	def set_exclusion_threshold(self, min_learning_score, n_last_trials):
		for task in self:
			task.set_exclusion_threshold(min_learning_score, n_last_trials)

	def get_fittable_dataset(self):
		'''
		Convert entire experimental results to dataset format for fitting to
		the model.
		'''
		dataset = []
		lexicons = []
		for l, task in enumerate(self):
			lexicons.append(task.lexicon)
			for user in task:
				for trial in user.iter_test_trials():
					t = trial['object']
					j = trial['fixation_position']
					w = trial['selected_object']
					dataset.append((l, t, j, w))
		return dataset, lexicons


class Task:

	def __init__(self, task_id, color='black', light_color='gray'):
		self.task_id = task_id
		self.color = color
		self.light_color = light_color
		self.task_data = core.json_read(DATA_DIR / f'{self.task_id}.json')
		self._users = []
		for user_id in range(1, self['n_participants'] + 1):
			user_id = str(user_id).zfill(2)
			try:
				user = User(self.task_id, user_id)
			except FileNotFoundError:
				print(f'Missing participant data file: {self.task_id}, {user_id}')
				continue
			self._users.append(user)
		self.min_learning_score = 0
		self.n_last_trials = self['n_items']

	def __getitem__(self, key):
		return self.task_data[key]

	def __iter__(self):
		for user in self._users:
			if user.learning_score(self.n_last_trials) >= self.min_learning_score:
				yield user

	@property
	def id(self):
		return self.task_id

	@property
	def label(self):
		return self['task_name']

	@property
	def n_participants(self):
		return len(self._users)

	@property
	def n_retained_participants(self):
		return len([user for user in self])

	@property
	def lexicon(self):
		return list(map(tuple, self['words']))

	def set_exclusion_threshold(self, min_learning_score, n_last_trials):
		self.min_learning_score = min_learning_score
		self.n_last_trials = n_last_trials

	def iter_with_excludes(self):
		for user in self._users:
			yield user

	def get_fittable_dataset(self):
		'''
		Convert individual task results to dataset format for fitting to the model.
		'''
		dataset = []
		for user in self:
			for trial in user.iter_test_trials():
				t = trial['object']
				j = trial['fixation_position']
				w = trial['selected_object']
				dataset.append((0, t, j, w))
		return dataset, [self.lexicon]


class User:

	def __init__(self, task_id, user_id):
		self.task_id = task_id
		self.user_id = user_id
		self.user_data = core.json_read(DATA_DIR / self.task_id / f'{self.user_id}.json')
		self.trials = {'mini_test':[], 'ovp_test':[]}
		for response in self['responses']:
			response['correct'] = response['object'] == response['selected_object']
			self.trials[response['test_type']].append(response)
		self.excluded = False

	def __getitem__(self, key):
		return self.user_data[key]

	def exclude(self):
		self.excluded = True

	def iter_training_trials(self):
		for trial in self.trials['mini_test']:
			yield trial

	def iter_test_trials(self):
		for trial in self.trials['ovp_test']:
			yield trial

	def learning_score(self, n_last_trials=8):
		return sum([trial['correct'] for trial in self.trials['mini_test'][-n_last_trials:]])

	def ovp_score(self):
		return sum([trial['correct'] for trial in self.trials['ovp_test']])

	def learning_curve(self, n_previous_trials=8):
		correct = [trial['correct'] for trial in self.iter_training_trials()]
		return np.array([
			sum(correct[i-(n_previous_trials-1) : i+1]) for i in range(n_previous_trials-1, len(correct))
		]) / n_previous_trials

	def ovp_curve(self, normalize=True):
		n_successes_by_position = defaultdict(int)
		n_trials_by_position = defaultdict(int)
		for trial in self.iter_test_trials():
			position = trial['fixation_position']
			n_successes_by_position[position] += trial['correct']
			n_trials_by_position[position] += 1
		n_successes_by_position = np.array([
			n_successes_by_position[i] for i in range(len(n_successes_by_position))
		])
		if normalize:
			n_trials_by_position = np.array([
				n_trials_by_position[i] for i in range(len(n_trials_by_position))
			])
			return n_successes_by_position / n_trials_by_position
		return n_successes_by_position




def print_comments(experiment):
	'''

	Print all the comments from a selection of tasks.

	'''
	for task in experiment:
		print(task.task_id)
		for user in task.iter_with_excludes():
			comments = user['comments'].strip().replace('\n', ' ')
			print(f"{user['user_id']} {comments}\n")


def calculate_median_completion_time(experiment):
	'''

	Calculate median completion time. Because participants often do not start the
	task immediately after accepting it, we will define this as the time between
	the first trial response and the final submission time, plus 60 seconds to
	account for initial instruction reading etc.

	'''
	times = []
	for task in experiment:
		for user in task.iter_with_excludes():
			times.append(user['modified_time'] - user['responses'][0]['time'] + 60)
	base_rate = task['basic_pay'] / 100
	time = round(np.median(times) / 60)
	rate = round(60 / time * base_rate, 2)
	print(f'Median completion time of {time} minutes, resulting in an hourly rate of £{rate}')


def calculate_median_bonus(experiment):
	'''

	Calculate median bonus amount.

	'''
	bonuses = []
	for task in experiment:
		for user in task.iter_with_excludes():
			bonuses.append(user['total_bonus'])
	median_bonus = round(np.median(bonuses))
	print(f'Median bonus: {median_bonus}')


def print_linguistic_backgrounds(experiment, min_learning_score=0):
	langs_1st = defaultdict(int)
	langs_2nd = defaultdict(int)
	for task in experiment:
		for user in task.iter_with_excludes():
			if user.learning_score() < min_learning_score:
				continue
			langs_1st[user['first_language']] += 1
			for lang in user['other_languages']:
				langs_2nd[lang] += 1

	sum_langs_1st = sum(langs_1st.values())
	for lang, count in langs_1st.items():
		langs_1st[lang] = count / sum_langs_1st * 100

	sum_langs_2nd = sum(langs_2nd.values())
	for lang, count in langs_2nd.items():
		langs_2nd[lang] = count / sum_langs_2nd * 100

	print('First languages')
	for percentage, language in sorted([(n, lang) for lang, n in langs_1st.items()], reverse=True):
		print('-', round(percentage, 1), language)
	print('Other languages')
	for percentage, language in sorted([(n, lang) for lang, n in langs_2nd.items()], reverse=True):
		print('-', round(percentage, 1), language)


def check_size_selections(experiment):
	size_selections = {i:0 for i in range(50)}
	for task in experiment:
		for user in task.iter_with_excludes():
			size_selections[user['size_selection']] += 1
	plt.bar(*zip(*size_selections.items()))
	plt.show()





def plot_learning_scores(axis, experiment):
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

def plot_learning_curve(axis, learning_curve, n_previous_trials, color='black'):
	padding = (64 - n_previous_trials) * 0.05
	x_vals = range(n_previous_trials, len(learning_curve) + n_previous_trials)
	axis.plot([-4, 68], [1/8, 1/8], color='black', linestyle='--', linewidth=1)
	axis.plot(x_vals, learning_curve, color=color)
	axis.set_xlim(n_previous_trials-padding, 64+padding)
	axis.set_ylim(0, 1)
	axis.set_xticks([1, 8, 16, 24, 32, 40, 48, 56, 64])
	axis.set_xlabel('Mini-test trial')
	axis.set_ylabel('Probability of correct response')

def plot_learning_curves(axis, experiment, n_previous_trials=8):
	for task in experiment:
		learning_curves = []
		for user in task:
			learning_curves.append(user.learning_curve(n_previous_trials))
		mean_learning_curve = sum(learning_curves) / len(learning_curves)
		plot_learning_curve(axis, mean_learning_curve, n_previous_trials, color=task.color)

def plot_ovp_curve(axis, task, show_confidence_interval=True):
	ovp_curves = []
	n_participants = 0
	for user in task:
		ovp_curves.append(user.ovp_curve(normalize=False))
		n_participants += 1
	mean_ovp_curve = sum(ovp_curves) / (8*n_participants)

	word_length = len(mean_ovp_curve)
	positions = range(1, word_length + 1)
	
	axis.plot(np.full(word_length + 2, 1/8), color='black', linestyle='--', linewidth=1)
	if show_confidence_interval:
		lower, upper = proportion.proportion_confint(sum(ovp_curves), 8*n_participants, method='jeffreys')
		axis.fill_between(range(1, word_length+1), lower, upper, facecolor='#eeeeee')
	axis.plot(positions, mean_ovp_curve, color=task.color, linewidth=2)
	
	padding = (word_length - 1) * 0.05
	axis.set_xlim(1-padding, word_length+padding)
	axis.set_ylim(0, 1)
	axis.set_xticks(positions)
	axis.set_xlabel('Fixation position')
	axis.set_ylabel('Probability of correct response')


def make_results_figure(experiment, fig_file):
	with core.Figure(fig_file, 4, 2, width='double', height=4) as fig:
		plot_learning_scores(fig[0, 0], experiment)
		plot_learning_curves(fig[0, 1], experiment, n_previous_trials=1)
		plot_ovp_curve(fig[1, 0], experiment.left)
		draw_brace(fig[1, 0], (2,4), 0.2, 'High\ninformation\ncontent')
		draw_brace(fig[1, 0], (6,7), 0.2, 'Low\ninformation\ncontent')
		plot_ovp_curve(fig[1, 1], experiment.right)
		draw_brace(fig[1, 1], (4,6), 0.2, 'High\ninformation\ncontent')
		draw_brace(fig[1, 1], (1,2), 0.2, 'Low\ninformation\ncontent')
		fig.auto_deduplicate_axes = False


def print_posterior_summary(experiment):
	from arviz import summary
	for task in [experiment.left, experiment.right, experiment]:
		trace, parameters = load_posterior_trace(task)
		print(summary(trace))


def make_posterior_summary_table(experiment):
	from arviz import hdi
	latex_table = [[r'$\alpha$  '], [r'$\beta$   '], [r'$\gamma$  '], [r'$\epsilon$']]
	for task in [experiment.left, experiment.right, experiment]:
		trace, parameters = load_posterior_trace(task)
		intervals = hdi(trace, hdi_prob=0.95)
		for i, param in enumerate(parameters):
			x = np.linspace(*param['bounds'], 1000)
			mcmc_draws = trace.posterior.data_vars[param['name']].to_numpy().flatten()
			posterior = gaussian_kde(mcmc_draws).pdf(x)
			mode = x[posterior.argmax()]
			low, high = intervals[param['name']].to_numpy()
			latex_table[i].extend([str(round(mode, 2)), f'{round(low, 2)}---{round(high, 2)}'])
	for row in latex_table:
		print(' & '.join(row) + r' \\')


def load_posterior_trace(task):
	posterior_trace_file = core.MODEL_FIT / f'{task.id}_posterior.pkl'
	with open(posterior_trace_file, 'rb') as file:
		trace, parameters = pickle.load(file)
	return trace, parameters


def make_posterior_projections_figure(experiment, fig_file, max_normalize=True):
	if isinstance(experiment, Experiment):
		tasks = [experiment.left, experiment.right, experiment]
	else:
		tasks = [experiment]
	traces = [load_posterior_trace(task) for task in tasks]
	with core.Figure(fig_file, 4, width='double', height=1.5) as fig:
		label_added = False
		max_ys = []
		for axis, param in zip(fig, traces[0][1]):
			x = np.linspace(*param['bounds'], 1000)
			prior = beta.pdf(x, *param['prior'], loc=param['bounds'][0], scale=param['bounds'][1] - param['bounds'][0])
			if max_normalize:
				prior /= prior.max()
			max_ys.append(prior.max())
			if label_added:
				axis.plot(x, prior, color='gray', linestyle='--', linewidth=0.5)
			else:
				axis.plot(x, prior, color='gray', linestyle='--', linewidth=0.5, label='Pr($\\theta$)')
			axis.set_xlabel(f'${param["name"]}$')
			axis.set_xlim(*map(round, param['bounds']))
			axis.set_yticks([])
			axis.spines['top'].set_visible(False)
			axis.spines['right'].set_visible(False)
			axis.spines['left'].set_visible(False)
			label_added = True
		for task, (trace, parameters) in zip(tasks, traces):
			labels_added = False
			for axis, param in zip(fig, parameters):
				x = np.linspace(*param['bounds'], 1000)
				mcmc_draws = trace.posterior.data_vars[param['name']].to_numpy().flatten()
				posterior = gaussian_kde(mcmc_draws).pdf(x)
				if max_normalize:
					posterior /= posterior.max()
				max_ys.append(posterior.max())
				if labels_added:
					axis.plot(x, posterior, color=task.color, linewidth=0.5)
				else:
					data_subscript = '_' + task.label[0].upper() if isinstance(task, Task) else ''
					label = f'Pr($\\theta$|$D{data_subscript}$)'
					axis.plot(x, posterior, color=task.color, linewidth=0.5, label=label)
					labels_added = True
		max_y = max(max_ys)
		padding = max_y * 0.02
		min_y = -padding
		max_y += padding
		for axis in fig:
			axis.set_ylim(min_y, max_y)
		
		legend = fig.fig.legend(bbox_to_anchor=(0.45, 1), loc="upper center", frameon=False)
		for line in legend.legendHandles:
			line.set_linewidth(1.0)


def calculate_uncertainty_for_params(lexicon, params, n_simulations=1000):
	reader = model_fit.model.Reader(lexicon, *params)
	return [reader.uncertainty(j, 'fast', n_simulations) for j in range(reader.word_length)]

def half_violin(body, point_right=False):
	m = np.mean(body.get_paths()[0].vertices[:, 0])
	if point_right:
		body.get_paths()[0].vertices[:, 0] = np.clip(body.get_paths()[0].vertices[:, 0], m, np.inf)
	else:
		body.get_paths()[0].vertices[:, 0] = np.clip(body.get_paths()[0].vertices[:, 0], -np.inf, m)
	return body

def plot_uncertainty_prediction(experiment, model_fit_path, output_path, mcmc_draws=1000, mcmc_samples=10000, burn_in=300, n_uncertainty_sims=1000):
	samples = model_fit.sample_posterior(model_fit_path, mcmc_samples, burn_in, mcmc_draws)
	with core.Figure(output_path, 1, width='single', height=3) as fig:
		for i, task in enumerate(experiment):
			uncertainty_curves = []
			for *params, p in samples:
				uncertainty_curve = calculate_uncertainty_for_params(task.lexicon, params, n_uncertainty_sims)
				uncertainty_curves.append(uncertainty_curve)
			uncertainty_curves = np.array(uncertainty_curves)
			violin_plot = fig[0,0].violinplot(uncertainty_curves, showextrema=False)
			for body in violin_plot['bodies']:
				body.set_color(task.light_color)
				body.set_alpha(1)
				body.set_linewidth(0)
				half_violin(body, i)
			map_uncertainty = calculate_uncertainty_for_params(task.lexicon, [0.86, 0.09, 0.38, 0.07], n_uncertainty_sims)
			fig[0,0].plot(range(1, len(map_uncertainty)+1), map_uncertainty, color=task.color, label=task['task_name'])
			fig[0,0].set_ylim(0, 1.2)
			fig[0,0].set_ylabel('Uncertainty (bits)')
			fig[0,0].legend(frameon=False, loc='upper left')


def convert_dataset_to_ovp_curves(dataset):
	correct = np.zeros((2, 7))
	n_trials = np.zeros((2, 7))
	for l, t, j, w in dataset:
		correct[l, j] += t == w
		n_trials[l, j] += 1
	return correct / n_trials

def plot_posterior_predictive_checks(experiment, output_path, n_simulations=100):
	from model import simulate_experimental_dataset
	lexicons = [experiment.left.lexicon, experiment.right.lexicon]
	n_participants = [experiment.left.n_retained_participants, experiment.right.n_retained_participants]
	trace, parameters = load_posterior_trace(experiment)
	mcmc_draws = np.column_stack([trace.posterior.data_vars[param].to_numpy().flatten() for param in trace.posterior])
	random_draw_indices = np.random.choice(len(mcmc_draws), n_simulations, replace=False)
	posterior_predictive_draws = mcmc_draws[random_draw_indices]

	with core.Figure(output_path, 2, width='double', height=2) as fig:
		all_ovp_curves = []
		for param_values in posterior_predictive_draws:
			dataset = simulate_experimental_dataset(lexicons, param_values, n_participants)
			ovp_curves = convert_dataset_to_ovp_curves(dataset)
			all_ovp_curves.append(ovp_curves)
			fig[0,0].plot(range(1, 8), ovp_curves[0], color=experiment.left.light_color, linewidth=0.5)
			fig[0,1].plot(range(1, 8), ovp_curves[1], color=experiment.right.light_color, linewidth=0.5)

		mean_ovp_curves = sum(all_ovp_curves) / len(all_ovp_curves)
		fig[0,0].plot(range(1, 8), mean_ovp_curves[0], color=experiment.left.color, linewidth=1, linestyle='--')
		fig[0,1].plot(range(1, 8), mean_ovp_curves[1], color=experiment.right.color, linewidth=1, linestyle='--')

		plot_ovp_curve(fig[0,0], experiment.left, show_confidence_interval=False)
		plot_ovp_curve(fig[0,1], experiment.right, show_confidence_interval=False)
		fig[0,0].set_ylim(0.5, 1)
		fig[0,1].set_ylim(0.5, 1)

		a = Line2D([0], [0], color='gray', linestyle='-', linewidth=0.5, label='Simulated runs of the experiment')
		b = Line2D([0], [0], color='black', linestyle='--', linewidth=1, label='Mean of simulated runs')
		c = Line2D([0], [0], color='black', linestyle='-', linewidth=2, label='Actual experimental results')
		handles = [a, b, c]
		labels = [h.get_label() for h in handles]
		fig[0,0].legend(handles=handles, labels=labels, frameon=False)


def draw_brace(ax, xspan, yy, text):
	"""Draws an annotated brace on the axes."""
	xmin, xmax = xspan
	xspan = xmax - xmin
	ax_xmin, ax_xmax = ax.get_xlim()
	xax_span = ax_xmax - ax_xmin

	ymin, ymax = ax.get_ylim()
	yspan = ymax - ymin
	resolution = int(xspan/xax_span*100)*2+1 # guaranteed uneven
	beta = 300./xax_span # the higher this is, the smaller the radius

	x = np.linspace(xmin, xmax, resolution)
	x_half = x[:int(resolution/2)+1]
	y_half_brace = (1/(1.+np.exp(-beta*(x_half-x_half[0])))
					+ 1/(1.+np.exp(-beta*(x_half-x_half[-1]))))
	y = np.concatenate((y_half_brace, y_half_brace[-2::-1]))
	y = yy + (.05*y - .01)*yspan # adjust vertical position

	ax.autoscale(False)
	ax.plot(x, y, color='black', lw=1)

	ax.text((xmax+xmin)/2., yy+.07*yspan, text, ha='center', va='bottom')
