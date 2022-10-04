import numpy as np
from . import model


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
	datasets = []
	for param_values in post_pred:
		dataset = []
		for l, condition in enumerate(experiment.unpack()):
			D = model.simulate_dataset(condition.lexicon, param_values, condition.n_retained_participants, n_test_reps=1, lexicon_index=l, decision_rule='MAP')
			dataset.extend(D)
		datasets.append(dataset)
	return datasets


def uncertainty_curve_from_posterior(experiment, n_sims=1000):
	'''
	Compute the expected uncertainty curve using the posterior parameter
	estimates from an experiment.
	'''
	trace = experiment.get_posterior()
	mean_param_values = [float(trace.posterior[param].mean()) for param in experiment.params]
	curves = []
	for condition in experiment.unpack():
		reader = model.Reader(condition.lexicon, *mean_param_values)
		curve = [reader.uncertainty(j, 'fast', n_sims) for j in range(reader.word_length)]
		curves.append(curve)
	return curves


def uncertainty_curves_from_posterior_draws(experiment, n_draws=100, n_sims=1000):
	'''
	Compute the expected uncertainty curve using the posterior parameter
	estimates from an experiment.
	'''
	trace = experiment.get_posterior()
	vals = [trace.posterior[param].to_numpy().flatten() for param in experiment.params]
	curves_by_condition = []
	for condition in experiment.unpack():
		curves = []
		for sample_i in np.random.randint(0, len(vals[0]), n_draws):
			param_values = [vals[param_i][sample_i] for param_i in range(len(experiment.params))]
			reader = model.Reader(condition.lexicon, *param_values)
			curve = [reader.uncertainty(j, 'fast', n_sims) for j in range(reader.word_length)]
			curves.append(curve)
		curves = np.array(curves)
		curves_by_condition.append(curves)
	return curves_by_condition
