'''

This module handles fitting experimental data to the model. To fit the model,
pass an Experiment or Task object to fit_model_to_dataset():

>>> fit_model_to_dataset(experiment, 'path/to/store/results.pkl')

And to print the MAP estimates with 95% credible regions, use:

>>> print_map_estimates('path/to/store/results.pkl')

'''


import numpy as np
import skopt
import model


PARAMETER_SPACE = [
	skopt.space.Real( 0.0625, 0.9999, name='α'),
	skopt.space.Real( 0.0001, 0.9999, name='β'),
	skopt.space.Real(-0.9999, 0.9999, name='γ'),
	skopt.space.Real( 0.0001, 0.9999, name='ε'),
]


def print_iteration(result, final=False):
	'''
	Skopt callback function for printing result of current iteration.
	'''
	if final:
		parameter_vals = ', '.join([f'{param.name} = {round(value, 4)}' for param, value in zip(PARAMETER_SPACE, result.x)])
		log_likelihood = f'log likelihood = {-result.fun}'
		print(parameter_vals, log_likelihood)
	else:
		parameter_vals = ', '.join([f'{param.name} = {round(value, 4)}' for param, value in zip(PARAMETER_SPACE, result.x_iters[-1])])
		log_likelihood = f'log likelihood = {-result.func_vals[-1]}'
		if len(result.func_vals) > 1 and result.func_vals[-1] < min(result.func_vals[:-1]):
			log_likelihood += ' *'
		print(f'{len(result.x_iters)}: ' + parameter_vals, log_likelihood)


def compute_word_inferences(lexicons, theta, n_simulations):
	'''
	Given a bunch of lexicons and candidate parameter values, compute an 8x7x8
	matrix for each lexicon, which gives the probability that the model reader
	would infer word w given t,j.
	'''
	word_length = len(lexicons[0][0])
	word_inference_matrix_for_each_lexicon = []
	for lexicon in lexicons:
		reader = model.Reader(lexicon, *theta)
		p_w_given_tj = np.zeros((8, word_length, 8), dtype=np.float64)
		for t in range(8):
			for j in range(word_length):
				p_w_given_tj[t,j] = reader.p_word_given_target(t, j, method='fast', n_sims=n_simulations)
		word_inference_matrix_for_each_lexicon.append(p_w_given_tj)
	return word_inference_matrix_for_each_lexicon


def fit_model_to_dataset(experiment, output_path, n_evaluations=300, n_random_evaluations=100, n_simulations=100000, tip=None):
	'''
	Use the skopt optimizer to the find parameter values that minimize the
	negative log likelihood of the model generating an observed experimental
	dataset. The resulting OptimizeResult object is writen to a file.
	'''
	if isinstance(experiment, tuple) and len(experiment) == 2:
		dataset, lexicons = experiment
	else:
		dataset, lexicons = experiment.get_fittable_dataset()

	def neg_log_likelihood_dataset(theta):
		word_inferences = compute_word_inferences(lexicons, theta, n_simulations)
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
		dimensions=PARAMETER_SPACE,
		n_calls=n_evaluations,
		n_random_starts=n_random_evaluations,
		x0=tip,
		model_queue_size=1,
		callback=print_iteration,
	)
	del result.specs['args']['func']
	del result.specs['args']['callback']
	skopt.utils.dump(result, output_path)
	return result


def view_model_fit(file_path):
	'''
	Plot model fit results from the OptimizeResult object previously written to a
	file.
	'''
	from skopt.plots import plot_convergence, plot_objective
	import matplotlib.pyplot as plt
	result = skopt.utils.load(file_path)
	print_iteration(result, final=True)
	plot_objective(result, size=2, levels=32, show_points=False)
	plt.show()


def metropolis(evaluation_func, n_params, mcmc_samples, burn_in, sd=0.01):
	'''
	Draw samples from a log2 probability function using the Metropolis
	algorithm.
	'''
	samples = np.empty((mcmc_samples, n_params + 1))
	params = [0.5] * n_params
	log_p = evaluation_func(params)
	for i in range(mcmc_samples + burn_in):
		cand_params = [np.clip(np.random.normal(param, sd), 0, 1) for param in params]
		cand_log_p = evaluation_func(cand_params)
		alpha = cand_log_p - log_p
		if (alpha >= 0.0) or (np.log2(np.random.random()) < alpha):
			params = cand_params
			log_p = cand_log_p
		if i >= burn_in:
			samples[i - burn_in] = [*params, log_p]
	return samples


def logsumexp2(array):
	'''
	Sum an array in the log2 domain.
	'''
	array_max = array.max()
	return np.log2(np.sum(np.exp2(array - array_max))) + array_max


def sample_model(result, mcmc_samples, burn_in):
	'''
	Draw samples from the final model in a skopt optimization result object.
	'''
	n_params = len(result.space.dimensions)
	GP_model = result.models[-1]
	evaluation_func = lambda params: -GP_model.predict([params])[0]
	samples = metropolis(evaluation_func, n_params, mcmc_samples, burn_in)
	for i, dim in enumerate(result.space.dimensions):
		samples[:, i] = dim.inverse_transform(samples[:, i])
	samples[:, -1] -= logsumexp2(samples[:, -1])
	return samples


def maximum_a_posteriori(samples):
	'''
	Calculate the MAP estimates from some MCMC samples.
	'''
	return samples[np.argmax(samples[:, -1]), :-1]


def highest_posterior_density(samples, probability=0.95):
	'''
	Calculate HPD credible regions from some MCMC samples for a given
	probability.
	'''
	probability = np.log2(probability)
	sum_p = None
	params_in_hdi = []
	# Iterate over samples from most probable to least probable
	for *params, p in sorted(samples, key=lambda r: r[-1], reverse=True):
		sum_p = p if sum_p is None else np.logaddexp2(sum_p, p)
		params_in_hdi.append(params)
		if sum_p >= probability:
			break # stop once x% of the probability mass has been accounted for
	return list(zip(np.min(params_in_hdi, axis=0), np.max(params_in_hdi, axis=0)))


def print_map_estimates(result_file, credible_regions=0.95, round_to=2, mcmc_samples=10000, burn_in=300):
	result = skopt.utils.load(result_file)
	samples = sample_model(result, mcmc_samples, burn_in)
	map_estimates = maximum_a_posteriori(samples)
	credible_regions = highest_posterior_density(samples, credible_regions)
	for dim, estimate, (lower, upper) in zip(result.space.dimensions, map_estimates, credible_regions):
		estimate = round(estimate, round_to)
		lower = round(lower, round_to)
		upper = round(upper, round_to)
		print(f'{dim.name} = {estimate} ({lower}---{upper})')


def extract_slice_from_gaussian_process(result, target_param, granularity=1000, return_log=False):
	'''
	Extract the posterior over parameter values for a particular target
	parameter from a skopt optimization result. This effectively takes a
	slice through the Gaussian process landscape along one dimension,
	while holding all other dimensions constant at the ML estimate.
	'''
	param_bounds = result.space.dimensions[target_param].bounds
	param_space = np.linspace(*param_bounds, granularity)
	
	# Holding all other parameters constant at their ML estimate, extract the shape
	# of the Gaussian process along the target parameter
	GP_input = np.empty((granularity, len(result.x)))
	for param, param_value in enumerate(result.x):
		dimension = result.space.dimensions[param]
		if param == target_param:
			GP_input[:, param] = dimension.transform(param_space)
		else:
			GP_input[:, param] = dimension.transform(param_value)
	GP_prediction = result.models[-1].predict(GP_input)
	
	# Convert the Gaussian process predictions into posterior estimates
	log_posterior = -GP_prediction # un-negative the predictions
	log_posterior = log_posterior - logsumexp2(log_posterior) # normalize
	if return_log:
		return param_space, log_posterior
	return param_space, np.exp2(log_posterior)


def simulate_experimental_dataset(lexicons, n_participants, params):
	'''
	Simulate an experimental dataset with a certain number of participants per
	condition/lexicon.
	'''
	simulated_dataset = []
	for l, lexicon in enumerate(lexicons):
		for p in range(n_participants):
			reader = model.Reader(lexicon, *params)
			participant_dataset = [(l, t, j, w) for t, j, w in reader.test()]
			simulated_dataset.extend(participant_dataset)
	return simulated_dataset, lexicons


if __name__ == '__main__':

	# Test the model fit procedure by generating a synthetic dataset and
	# attempting to recover the underlying parameters.

	lexicons = [
		[(0, 6, 12, 2, 10, 1, 0), (0, 7, 13, 2, 10, 1, 0), (0, 8, 14, 3, 10, 1, 0), (0, 9, 15, 3, 10, 1, 0), (0, 8, 12, 4, 11, 1, 0), (0, 6, 13, 4, 11, 1, 0), (0, 9, 14, 5, 11, 1, 0), (0, 7, 15, 5, 11, 1, 0)],
		[(0, 1, 10, 2, 12, 6, 0), (0, 1, 10, 2, 13, 7, 0), (0, 1, 10, 3, 14, 8, 0), (0, 1, 10, 3, 15, 9, 0), (0, 1, 11, 4, 12, 8, 0), (0, 1, 11, 4, 13, 6, 0), (0, 1, 11, 5, 14, 9, 0), (0, 1, 11, 5, 15, 7, 0)]
	]
	
	dataset = simulate_experimental_dataset(
		lexicons,
		n_participants=30,
		params=[0.8, 0.1, 0.3, 0.05]
	)
	
	fit_model_to_dataset(
		dataset,
		output_path='test.pkl',
		n_evaluations=100,
		n_random_evaluations=30,
		n_simulations=100000,
		tip=[0.8, 0.15, 0.4, 0.1]
	)

	print_map_estimates('test.pkl')
