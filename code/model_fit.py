import numpy as np
import skopt
import model
import core


N_CALLS = 300
N_RANDOM_STARTS = 100
N_SIMS = 100000
SPACE = [
	skopt.space.Real( 0.5000, 0.9999, name='α'),
	skopt.space.Real( 0.1000, 0.9999, name='β'),
	skopt.space.Real( 0.5000, 2.5000, name='γ'),
	skopt.space.Real(-0.5000, 0.5000, name='δ'),
	skopt.space.Real( 0.0001, 0.2000, name='ε'),
]


def retrieve_lexicons(conditions):
	'''

	Retreive the lexicons from the relevant experiment setup files.

	'''
	lexicons = []
	for condition in conditions:
		lexicon = list(map(tuple, core.json_read(core.EXP_DATA / f'{condition}.json')['words']))
		lexicons.append(lexicon)
	return lexicons


def simulate_experimental_dataset(lexicons, n_participants, theta):
	'''

	Simulate an experimental dataset with a certain number of participants per
	condition/lexicon.

	'''
	experimental_dataset = []
	for l, lexicon in enumerate(lexicons):
		for p in range(n_participants):
			reader = model.Reader(lexicon, *theta)
			participant_dataset = [(l, t, j, w) for t, j, w in reader.test()]
			experimental_dataset.extend(participant_dataset)
	return experimental_dataset


def print_iteration(result, final=False):
	'''

	Skopt callback function for printing result of current iteration.

	'''
	if final:
		parameter_vals = ', '.join([f'{param.name} = {round(value, 4)}' for param, value in zip(SPACE, result.x)])
		log_likelihood = f'log likelihood = {-result.fun}'
		print(parameter_vals, log_likelihood)
	else:
		parameter_vals = ', '.join([f'{param.name} = {round(value, 4)}' for param, value in zip(SPACE, result.x_iters[-1])])
		log_likelihood = f'log likelihood = {-result.func_vals[-1]}'
		print(f'{len(result.x_iters)}: ' + parameter_vals, log_likelihood)


def compute_word_inferences(lexicons, theta):
	'''

	Given a bunch of lexicons and candidate parameter values, compute an 8x5x8
	matrix for each lexicon, which gives the probability that the model reader
	would infer word w given t,j.
	
	'''
	word_inferences = []
	for lexicon in lexicons:
		reader = model.Reader(lexicon, *theta)
		p_w_given_tj = np.zeros((8, 5, 8), dtype=np.float64)
		for t in range(8):
			for j in range(5):
				p_w_given_tj[t,j] = reader.p_word_given_target(t, j, method='fast', n_sims=N_SIMS)
		word_inferences.append(p_w_given_tj)
	return word_inferences


def fit_model_to_dataset(dataset, lexicons, file_path):
	'''
	
	Use the skopt optimizer to the find parameter values that minimize the
	negative log likelihood of the model generating an observed experimental
	dataset. The resulting OptimizeResult object is writen to a file.

	'''
	def neg_log_likelihood_dataset(theta):
		word_inferences = compute_word_inferences(lexicons, theta)
		epsilon = theta[-1]
		p_stick_with_w = (1 - epsilon)
		p_switch_to_w = (epsilon / 7)
		log_likelihood = 0.0
		for l, t, j, w in dataset:
			log_likelihood += np.log2(
				# probability of inferring w and sticking to it, plus probability of
				# inferring some other w' but switching to w by mistake
				word_inferences[l][t, j, w] * p_stick_with_w + sum([
					word_inferences[l][t, j, w_prime] * p_switch_to_w for w_prime in range(8) if w_prime != w
				])
			)
		return -log_likelihood

	result = skopt.gp_minimize(
		neg_log_likelihood_dataset,
		dimensions=SPACE,
		n_calls=N_CALLS,
		n_random_starts=N_RANDOM_STARTS,
		model_queue_size=1,
		callback=print_iteration,
	)
	del result.specs['args']['func']
	skopt.utils.dump(result, file_path)
	return result


def get_model_fit(file_path):
	result = skopt.utils.load(file_path)
	return result.x


def view_model_fit(file_path):
	'''

	Plot model fit results from the OptimizeResult object previously written to a
	file.

	'''
	from skopt.plots import plot_convergence, plot_objective
	import matplotlib.pyplot as plt
	result = skopt.utils.load(file_path)
	print_iteration(result, final=True)
	plot_objective(result, size=2, levels=32)
	plt.show()


def convert_tasks_to_fitable_dataset(tasks, min_learning_score=0):
	'''

	Convert experimental task results to dataset format for fitting to the
	model.

	'''
	lexicons = []
	dataset = []
	for l, task in enumerate(tasks):
		lexicons.append(task.lexicon)
		for user in task:
			if user.learning_score() < min_learning_score:
				continue
			for trial in user.iter_test_trials():
				t = trial['object']
				j = trial['fixation_position']
				w = trial['selected_object']
				dataset.append((l, t, j, w))
	return lexicons, dataset


def fit_model(tasks, model_name, min_learning_score=0):
	'''

	Fit model to experimental results.

	'''
	lexicons, dataset = convert_tasks_to_fitable_dataset(tasks, min_learning_score)
	file_path = core.DATA / 'model_fit' / f'{model_name}.pkl'
	fit_model_to_dataset(dataset, lexicons, file_path)


def generate_synthetic_dataset_from_model_fit(model_name, conditions, theta):
	'''
	
	Generate a synthetic dataset from a model fit and write the dataset to a
	file.

	'''
	lexicons = retrieve_lexicons(conditions)
	dataset = model_fit.simulate_experimental_dataset(lexicons, 1000, theta)
	core.json_write(dataset, core.DATA / 'model_fit_datasets' / f'{model_name}.json', compress=True)


if __name__ == '__main__':

	# Test the model fit procedure by generating a synthetic dataset and
	# attempting to recover the underlying parameters.

	lexicons = retrieve_lexicons(['exp1_left', 'exp1_right'])
	simulated_dataset = simulate_experimental_dataset(lexicons, 30, 0.8, 0.2, 1.0, 0.3, 0.05)
	model_fit = fit_model_to_dataset(simulated_dataset, lexicons)
