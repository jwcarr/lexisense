import numpy as np
import core
import matplotlib.pyplot as plt
import model_fit
from collections import defaultdict


CONDITION_NAMES = ['Left-heavy', 'Right-heavy']


class Task:

	def __init__(self, task_id):
		self.task_id = task_id
		self.task_data = core.json_read(core.EXP_DATA / f'{self.task_id}.json')
		self._users = []
		for user_id in range(1, self['n_participants'] + 1):
			user_id = str(user_id).zfill(2)
			try:
				user = User(self.task_id, user_id)
			except FileNotFoundError:
				print(f'Missing participant data file: {self.task_id}, {user_id}')
				continue
			self._users.append(user)

	def __getitem__(self, key):
		return self.task_data[key]

	def __iter__(self):
		for user in self._users:
			yield user

	@property
	def n_participants(self):
		return len(self._users)

	@property
	def lexicon(self):
		return list(map(tuple, self['words']))


class User:

	def __init__(self, task_id, user_id):
		self.task_id = task_id
		self.user_id = user_id
		self.user_data = core.json_read(core.EXP_DATA / self.task_id / f'{self.user_id}.json')
		self.trials = {'mini_test':[], 'ovp_test':[]}
		for response in self['responses']:
			response['correct'] = response['object'] == response['selected_object']
			self.trials[response['test_type']].append(response)

	def __getitem__(self, key):
		return self.user_data[key]

	def iter_training_trials(self):
		for trial in self.trials['mini_test']:
			yield trial

	def iter_test_trials(self):
		for trial in self.trials['ovp_test']:
			yield trial

	def learning_score(self, n_last_trials=8):
		return sum([trial['correct'] for trial in self.trials['mini_test'][-n_last_trials:]])

	def learning_curve(self, n_previous_trials=8):
		correct = [trial['correct'] for trial in self.iter_training_trials()]
		return np.array([
			sum(correct[i-(n_previous_trials-1) : i+1]) for i in range(n_previous_trials-1, len(correct))
		]) / n_previous_trials

	def ovp_curve(self):
		n_successes_by_position = defaultdict(int)
		n_trials_by_position = defaultdict(int)
		for trial in self.iter_test_trials():
			position = trial['fixation_position']
			n_successes_by_position[position] += trial['correct']
			n_trials_by_position[position] += 1
		n_successes_by_position = np.array([
			n_successes_by_position[i] for i in range(len(n_successes_by_position))
		])
		n_trials_by_position = np.array([
			n_trials_by_position[i] for i in range(len(n_trials_by_position))
		])
		return n_successes_by_position / n_trials_by_position


def print_comments(tasks):
	'''

	Print all the comments from a selection of tasks.

	'''
	for task in tasks:
		print(task.task_id)
		for user in task:
			comments = user['comments'].strip().replace('\n', ' ')
			print(f"{user['user_id']} {comments}\n")


def calculate_median_completion_time(tasks):
	'''

	Calcualte median completion time. Because participants often do not start the
	task immediately after accepting it, we will define this as the time between
	the first trial response and the final submission time, plus 60 seconds to
	account for initial instruction reading etc.

	'''
	times = []
	for task in tasks:
		for user in task:
			times.append(user['modified_time'] - user['responses'][0]['time'] + 60)
	base_rate = task['basic_pay'] / 100
	time = round(np.median(times) / 60)
	rate = round(60 / time * base_rate, 2)
	print(f'Median completion time of {time} minutes, resulting in an hourly rate of £{rate}')


def check_size_selections(tasks):
	size_selections = {i:0 for i in range(50)}
	for task in tasks:
		for user in task:
			size_selections[user['size_selection']] += 1
	plt.bar(*zip(*size_selections.items()))
	plt.show()


def plot_learning_curve(axis, learning_curve, n_previous_trials):
	padding = (64 - n_previous_trials) * 0.05
	x_vals = range(n_previous_trials, len(learning_curve) + n_previous_trials)
	axis.plot(np.full(68, 1/8), color='black', linestyle='--', linewidth=1)
	axis.plot(x_vals, learning_curve, color='black')
	axis.set_xlim(n_previous_trials-padding, 64+padding)
	axis.set_ylim(-0.05, 1.05)
	axis.set_xticks([8, 16, 24, 32, 40, 48, 56, 64])
	axis.set_xlabel('Mini-test number')


def plot_ovp_curve(axis, ovp_curve, color='black', conf_low=None, conf_high=None):
	word_length = len(ovp_curve)
	positions = range(1, word_length + 1)
	padding = (word_length - 1) * 0.05
	axis.plot(np.full(word_length + 2, 1/8), color='black', linestyle='--', linewidth=1)
	# if conf_low is not None and conf_high is not None:
	# 	axis.fill_between(range(1, 6), conf_low, conf_high, facecolor='#eeeeee')
	axis.plot(positions, ovp_curve, color=color, linewidth=2)
	axis.set_xlim(1-padding, word_length+padding)
	axis.set_ylim(-0.05, 1.05)
	axis.set_xticks(positions)
	axis.set_xlabel('Fixation position')


def plot_individual_results(out_dir, tasks, n_previous_trials=8):
	'''

	Create a plot for each individual participant, showing their learning
	curve and their OVP curve.

	'''
	out_dir = out_dir / 'individual_results'
	if not out_dir.exists():
		out_dir.mkdir(parents=True)
	for task in tasks:
		for user in task:
			fig_file = out_dir / f'{task.task_id}_{user.user_id}.pdf'
			with core.Figure(fig_file, 2, width='double') as fig:
				plot_learning_curve(fig[0,0], user.learning_curve(n_previous_trials), n_previous_trials)
				plot_ovp_curve(fig[0,1], user.ovp_curve())
				fig[0,0].set_ylabel('Probability of correct response')


def plot_learning_scores(out_dir, tasks, n_last_trials=8):
	if not out_dir.exists():
		out_dir.mkdir(parents=True)
	learning_scores = defaultdict(int)
	for task in tasks:
		for user in task:
			score = user.learning_score(n_last_trials)
			learning_scores[score] += 1
	fig_file = out_dir / 'scores.pdf'
	with core.Figure(fig_file, 1, width='single', height=1.7) as fig:
		fig[0,0].bar(*zip(*learning_scores.items()))
		fig[0,0].set_xticks(range(n_last_trials+1))
		fig[0,0].set_ylabel('Number of participants')
		fig[0,0].set_xlabel(f'Number of correct responses during final {n_last_trials} mini-tests')


def plot_learning_curves(out_dir, tasks, n_previous_trials=8):
	if not out_dir.exists():
		out_dir.mkdir(parents=True)
	fig_file = out_dir / 'training.pdf'
	with core.Figure(fig_file, len(tasks), width='double') as fig:
		for axis, task in zip(fig, tasks):
			learning_curves = []
			for user in task:
				learning_curves.append(user.learning_curve(n_previous_trials))
			mean_learning_curve = sum(learning_curves) / len(learning_curves)
			plot_learning_curve(axis, mean_learning_curve, n_previous_trials)
		fig[0,0].set_ylabel('Probability of correct response')
		fig[0,0].set_title('Left-heavy language')
		fig[0,1].set_title('Right-heavy language')


def plot_ovp_curves(out_dir, tasks, min_learning_score=7, show_individual_curves=True):
	if not out_dir.exists():
		out_dir.mkdir(parents=True)
	fig_file = out_dir / 'test.pdf'
	with core.Figure(fig_file, len(tasks), width='double') as fig:
		for axis, task in zip(fig, tasks):
			ovp_curves = []
			for user in task:
				if user.learning_score() < min_learning_score:
					continue
				ovp_curve = user.ovp_curve()
				if show_individual_curves:
					axis.plot(range(1, len(ovp_curve)+1), ovp_curve, color='gray', alpha=0.3)
				ovp_curves.append(ovp_curve)
			mean_ovp_curve = sum(ovp_curves) / len(ovp_curves)
			plot_ovp_curve(axis, mean_ovp_curve)
		fig[0,0].set_ylabel('Probability of correct response')
		fig[0,0].set_title('Left-heavy language')
		fig[0,1].set_title('Right-heavy language')


def plot_training_inferences(out_dir, tasks, min_learning_score=0):
	fig_file = out_dir / 'training_inferences.pdf'
	with core.Figure(fig_file, len(tasks), len(tasks), width='double') as fig:
		for i, task in enumerate(tasks):
			n_words = task['n_items']
			inferences = np.zeros((n_words, n_words))
			for user in task:
				if user.learning_score() < min_learning_score:
					continue
				for trial in user.iter_training_trials():
					inferences[trial['object'], trial['selected_object']] += 1
			inferences /= inferences.sum(axis=1)
			fig[0,i].pcolor(inferences, vmin=0, vmax=1, cmap='Greys')
			fig[0,i].invert_yaxis()
			fig[0,i].set_xticks(np.arange(n_words)+0.5)
			fig[0,i].set_yticks(np.arange(n_words)+0.5)
			fig[0,i].set_xticklabels(np.arange(1, n_words+1))
			fig[0,i].set_yticklabels(np.arange(1, n_words+1))
			fig[0,i].set_ylabel('Target')
			fig[0,i].set_xlabel('Selection')
			fig[0,i].set_title(CONDITION_NAMES[i])


def plot_test_inferences(out_dir, tasks, min_learning_score=0):
	fig_file = out_dir / 'test_inferences.pdf'
	n_words = tasks[0]['n_items']
	word_length = tasks[0]['n_letters']
	with core.Figure(fig_file, word_length*len(tasks), word_length, width='double') as fig:
		for i, task in enumerate(tasks):
			inferences = np.zeros((n_words, word_length, n_words))
			for user in task:
				if user.learning_score() < min_learning_score:
					continue
				for trial in user.iter_test_trials():
					inferences[trial['object'], trial['fixation_position'], trial['selected_object']] += 1
			for j in range(word_length):
				inferences_this_pos = inferences[:, j, :]
				inferences_this_pos /= inferences_this_pos.sum(axis=1)
				fig[i,j].pcolor(inferences_this_pos, vmin=0, vmax=1, cmap='Greys')
				fig[i,j].invert_yaxis()
				fig[i,j].set_xticks(np.arange(n_words)+0.5)
				fig[i,j].set_yticks(np.arange(n_words)+0.5)
				fig[i,j].set_xticklabels(np.arange(1, n_words+1))
				fig[i,j].set_yticklabels(np.arange(1, n_words+1))
				fig[i,j].set_ylabel('Target')
				fig[i,j].set_xlabel('Selection')
				if j == word_length // 2:
					fig[i,j].set_title(CONDITION_NAMES[i])














# def compute_ovp_curve_from_dataset(dataset):
# 	successes_by_position = (np.zeros(5), np.zeros(5))
# 	counts_by_position = (np.zeros(5), np.zeros(5))
# 	for l, p, t, j, w in dataset:
# 		successes_by_position[l][j] += t == w
# 		counts_by_position[l][j] += 1
# 	ovp_curve_left = successes_by_position[0] / counts_by_position[0]
# 	ovp_curve_right = successes_by_position[1] / counts_by_position[1]
# 	return ovp_curve_left, ovp_curve_right

# def compute_word_inferences(dataset):
# 	inferences = (np.zeros((8, 5, 8), dtype=float), np.zeros((8, 5, 8), dtype=float))
# 	for l, p, t, j, w in dataset:
# 		inferences[l][t, j, w] += 1
# 	return inferences

# def plot_word_inferences(out_dir, conditions, min_learning_score=7, model_comparison=None, model_color=None):
# 	if model_comparison:
# 		simulated_dataset = core.json_read(core.DATA / 'model_fit_datasets' / f'{model_comparison}.json')
# 		model_inferences = compute_word_inferences(simulated_dataset)

# 	for i, condition in enumerate(conditions):
# 		inferences = np.zeros((8, 5, 8), dtype=float)
# 		for usr in iter_participant_data(condition, n_participants):
# 			results = get_test_results(usr)
# 			learning_score = sum([trial['correct'] for trial in results['mini_test'][-8:]])
# 			if learning_score < min_learning_score:
# 				continue
# 			for trial in results['ovp_test']:
# 				t = trial['object']
# 				j = trial['fixation_position']
# 				w = trial['selected_object']
# 				inferences[t, j, w] += 1
# 		if model_comparison:
# 			fig_file = out_dir / f'{condition}_word_inferences_{model_comparison}.pdf'
# 		else:
# 			fig_file = out_dir / f'{condition}_word_inferences.pdf'
# 		with core.Figure(fig_file, 40, 5, width=9, height=6) as fig:
# 			for t in range(8):
# 				for j in range(5):
# 					inferences[t, j] /= inferences[t, j].sum()
# 					fig[t,j].bar(range(1, 9), inferences[t, j], color='black')
# 					if model_comparison:
# 						model_inferences[i][t, j] /= model_inferences[i][t, j].sum()
# 						fig[t,j].bar(range(1, 9), model_inferences[i][t, j], width=0.3, color=model_color)
# 					fig[t,j].set_ylim(0, 1)
# 					fig[t,j].set_xticks(range(1, 9))
# 					fig[t,j].set_xticklabels(range(1, 9))
# 					fig[t,j].set_xlabel(f'Fixation position {j+1}')
# 					fig[t,j].set_ylabel(f'Word {t+1}')
