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
from scipy import stats
from . import model


PYMC3_DISTRIBUTION_FUNCS = {'normal': pymc3.Normal, 'beta':pymc3.Beta}
SCIPY_DISTRIBUTION_FUNCS = {'normal': stats.norm, 'beta':stats.beta}


PARAMETER_BOUNDS = {
	'α': ( 0.0625, 0.9999),
	'β': ( 0.0001, 1.0000),
	'γ': (-0.9999, 0.9999),
	'ε': ( 0.0001, 0.9999),
}


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


def maximum_a_priori(experiment):
	'''
	Return the maximum a-priori parameter estimates for a given experiment,
	which may be used as an initial evaluation point in the creation of
	the surrogate likelihood.
	'''
	max_a_priori = []
	transformed_space = np.linspace(0, 1, 1000)
	for name, (distribution, params) in experiment.priors.items():
		pdf = SCIPY_DISTRIBUTION_FUNCS[distribution](*params).pdf(transformed_space)
		parameter_space = np.linspace(*PARAMETER_BOUNDS[name], 1000)
		max_a_priori.append(parameter_space[pdf.argmax()])
	return max_a_priori


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


def create_surrogate_likelihood(experiment, n_evaluations=500, n_random_evaluations=199, n_simulations=100000):
	'''
	Use Skopt to the find parameter values that minimize the negative log
	likelihood of the model generating an observed experimental dataset.
	The resulting OptimizeResult object is writen to a file. The final
	Gaussian Process model contained in this object is an approximation of
	the true likelihood function.
	'''
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
		dimensions=[skopt.space.Real(*bounds, name=name) for name, bounds in PARAMETER_BOUNDS.items()],
		n_calls=n_evaluations,
		n_random_starts=n_random_evaluations,
		x0=maximum_a_priori(experiment),
		model_queue_size=1,
		callback=print_iteration,
	)
	del result.specs['args']['func']
	del result.specs['args']['callback']
	skopt.utils.dump(result, experiment.likelihood_file)
	return result


class BlackBoxLikelihood(pymc3.utils.tt.Op):

	itypes = [pymc3.utils.tt.dvector]
	otypes = [pymc3.utils.tt.dscalar]

	def __init__(self, func):
		self.func = func

	def perform(self, node, inputs, outputs):
		outputs[0][0] = np.array(self.func(inputs[0]))


def fit_posterior(experiment, n_samples=30000, n_tuning_samples=500, n_chains=8):
	skopt_optimization_result = skopt.utils.load(experiment.likelihood_file)
	final_GP_model = skopt_optimization_result.models[-1]
	surrogate_likelihood = BlackBoxLikelihood(lambda theta: -final_GP_model.predict([theta])[0])
	with pymc3.Model() as model:
		theta = pymc3.utils.tt.as_tensor_variable([
			PYMC3_DISTRIBUTION_FUNCS[distribution](name, *params)
			for name, (distribution, params) in experiment.priors.items()
		])
		pymc3.DensityDist('likelihood', lambda v: surrogate_likelihood(v), observed={'v': theta})
		trace = pymc3.sample(n_samples, tune=n_tuning_samples, chains=n_chains, cores=1,
			return_inferencedata=True, idata_kwargs={'density_dist_obs': False}
		)
	trace.to_netcdf(experiment.posterior_file)


def inverse_transform(values, bounds):
	'''
	Transform parameter values from [0,1] to specified parameter bounds.
	'''
	diff = bounds[1] - bounds[0]
	return values * diff + bounds[0]


def simulate_from_posterior(experiment, n_sims=100):
	'''
	Simulate a dataset based on the lexicon, number of participants, and
	posterior parameter values of a given experimental condition. This
	dataset can be used to perform posterior predictive checks.
	'''
	params = experiment.priors.keys()
	trace = experiment.get_posterior()
	post_pred = np.zeros((n_sims, 4), dtype=float)
	for i, param in enumerate(params):
		draws = trace.posterior[param].to_numpy().flatten()
		post_pred[:, i] = np.random.choice(draws, n_sims)
		post_pred[:, i] = inverse_transform(post_pred[:, i], experiment.params[param])
	datasets = []
	for param_values in post_pred:
		dataset = []
		for l, condition in enumerate(experiment.unpack()):
			D = model.simulate_dataset(condition.lexicon, param_values, condition.n_retained_participants, lexicon_index=l)
			dataset.extend(D)
		datasets.append(dataset)
	return datasets


def uncertainty_curve_from_posterior(experiment, n_sims=1000):
	'''
	Compute the expected uncertainty curve using the posterior parameter
	estimates from an experiment.
	'''
	trace = experiment.get_posterior()
	mean_param_values = [
		inverse_transform(float(value), experiment.params[param])
		for param, value in trace.posterior.mean().items()
	]
	curves = []
	for condition in experiment.unpack():
		reader = model.Reader(condition.lexicon, *mean_param_values)
		curve = [reader.uncertainty(j, 'fast', n_sims) for j in range(reader.word_length)]
		curves.append(curve)
	return curves
