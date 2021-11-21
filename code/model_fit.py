'''
This code performs the model fit for Experiment 1, which is done in two steps.
First you create the surrogate likelihood from the experimental data
(represented by an Experiment/Task object), e.g.:

	create_surrogate_likelihood(experiment, 'likelihood.pkl')

Second, you create the posterior trace using the surrogate likelihood and
priors:

	create_posterior_trace('likelihood.pkl', 'posterior.pkl')

The posterior trace can then be inspected using the functions provided in
exp_analysis.py
'''

import pickle
import numpy as np
import skopt
import pymc3
import model

# Each model parameter is rescaled in [0, 1] under the hood, thus the priors
# are specified as beta distributions over the closed interval [0, 1]. The
# maximum_a_priori values are specified in the true parameter bounds and are
# used as one of the initial random points in the formation of the Gaussian
# process surrogate of the likelihood.

PARAMETERS = [
	{'name':'α', 'bounds':( 0.0625, 0.9999), 'prior':(8, 2), 'maximum_a_priori':0.883},
	{'name':'β', 'bounds':( 0.0001, 1.0000), 'prior':(2, 8), 'maximum_a_priori':0.125},
	{'name':'γ', 'bounds':(-0.9999, 0.9999), 'prior':(4, 2), 'maximum_a_priori':0.500},
	{'name':'ε', 'bounds':( 0.0001, 0.9999), 'prior':(2, 8), 'maximum_a_priori':0.125},
]


def print_iteration(result, final=False):
	'''
	Skopt callback function for printing result of current iteration.
	'''
	if final:
		parameter_vals = ', '.join([f'{param.name} = {round(value, 4)}'
			for param, value in zip(result.space.dimensions, result.x)
		])
		log_likelihood = f'log likelihood = {-result.fun}'
		print(parameter_vals, log_likelihood)
	else:
		parameter_vals = ', '.join([f'{param.name} = {round(value, 4)}'
			for param, value in zip(result.space.dimensions, result.x_iters[-1])
		])
		log_likelihood = f'log likelihood = {-result.func_vals[-1]}'
		if len(result.func_vals) > 1 and result.func_vals[-1] < min(result.func_vals[:-1]):
			log_likelihood += ' *'
		print(f'{len(result.x_iters)}: ' + parameter_vals, log_likelihood)


def compute_p_word_given_target(lexicons, theta, n_words, word_length, n_simulations):
	'''
	Given a bunch of lexicons and candidate parameter values, compute an 8x7x8
	matrix for each lexicon, which gives the probability that the model reader
	would infer word w given t,j.
	'''
	word_inference_matrix_for_each_lexicon = []
	for lexicon in lexicons:
		reader = model.Reader(lexicon, *theta)
		p_w_given_tj = np.zeros((n_words, word_length, n_words), dtype=np.float64)
		for t in range(n_words):
			for j in range(word_length):
				p_w_given_tj[t,j] = reader.p_word_given_target(t, j, method='fast', n_sims=n_simulations)
		word_inference_matrix_for_each_lexicon.append(p_w_given_tj)
	return word_inference_matrix_for_each_lexicon


def create_surrogate_likelihood(experiment, surrogate_likelihood_file, n_evaluations=500, n_random_evaluations=199, n_simulations=100000):
	'''
	Use Skopt to the find parameter values that minimize the negative log
	likelihood of the model generating an observed experimental dataset.
	The resulting OptimizeResult object is writen to a file. The final
	Gaussian Process model contained in this object is an approximation of
	the true likelihood function.
	'''
	if isinstance(experiment, tuple) and len(experiment) == 2:
		dataset, lexicons = experiment
	else:
		dataset, lexicons = experiment.get_fittable_dataset()
	n_words = len(lexicons[0])
	word_length = len(lexicons[0][0])

	def neg_log_likelihood_dataset(theta):
		# Precompute Pr(w'|t,j) for each lexicon
		p_word_given_target = compute_p_word_given_target(lexicons, theta, n_words, word_length, n_simulations)
		epsilon = theta[-1]
		p_stick_with_w = (1 - epsilon)
		p_switch_to_w = epsilon / (n_words - 1)
		log_likelihood = 0.0
		for l, t, j, w in dataset:
			log_likelihood += np.log(
				# probability of inferring w and sticking to it, plus probability of
				# inferring some other w' but switching to w by mistake
				p_word_given_target[l][t, j, w] * p_stick_with_w + sum([
					p_word_given_target[l][t, j, w_prime] * p_switch_to_w for w_prime in range(n_words) if w_prime != w
				])
			)
		return -log_likelihood

	result = skopt.gp_minimize(
		neg_log_likelihood_dataset,
		dimensions=[skopt.space.Real(*param['bounds'], name=param['name']) for param in PARAMETERS],
		n_calls=n_evaluations,
		n_random_starts=n_random_evaluations,
		x0=[param['maximum_a_priori'] for param in PARAMETERS],
		model_queue_size=1,
		callback=print_iteration,
	)
	del result.specs['args']['func']
	del result.specs['args']['callback']
	skopt.utils.dump(result, surrogate_likelihood_file)
	return result


class BlackBoxLikelihood(pymc3.utils.tt.Op):

	itypes = [pymc3.utils.tt.dvector]
	otypes = [pymc3.utils.tt.dscalar]

	def __init__(self, func):
		self.func = func

	def perform(self, node, inputs, outputs):
		outputs[0][0] = pymc3.utils.tt.np.array(self.func(inputs[0]))


def create_posterior_trace(surrogate_likelihood_file, posterior_trace_file, n_samples=30000, n_tuning_samples=500, n_chains=4):
	'''
	Use PyMC3 to draw samples from the posterior. This combines the surrogate
	likelihood function, created by create_surrogate_likelihood(), with
	the prior, defined in PARAMETERS, and pickles the trace.
	'''
	skopt_optimization_result = skopt.utils.load(surrogate_likelihood_file)
	final_GP_model = skopt_optimization_result.models[-1]
	surrogate_likelihood = BlackBoxLikelihood(lambda theta: -final_GP_model.predict([theta])[0])

	with pymc3.Model() as model:
		theta = pymc3.utils.tt.as_tensor_variable([
			pymc3.Beta(param['name'], *param['prior']) for param in PARAMETERS
		])
		pymc3.DensityDist('likelihood', lambda v: surrogate_likelihood(v), observed={'v': theta})
		trace = pymc3.sample(n_samples, tune=n_tuning_samples, chains=n_chains, cores=1,
			return_inferencedata=True, idata_kwargs={'density_dist_obs': False}
		)

	for dim in skopt_optimization_result.space.dimensions:
		trace.posterior.data_vars[dim.name][:] = dim.inverse_transform(trace.posterior.data_vars[dim.name][:])

	with open(posterior_trace_file, 'wb') as file:
		pickle.dump((trace, PARAMETERS), file)
