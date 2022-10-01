import numpy as np
import pymc as pm
from experiment import Experiment


N_CORES = 6


def fit_posterior(experiment, n_chains=6, n_samples=2500, n_tuning_samples=500, data_subset=None, independent_ζ=False, independent_ξ=False, uniform_priors=False):

	# Get relevant prior params from Experiment object
	τ_l_mu, τ_l_sigma = experiment.left.priors['τ'][1]
	δ_l_mu, δ_l_sigma = experiment.left.priors['δ'][1]
	τ_r_mu, τ_r_sigma = experiment.right.priors['τ'][1]
	δ_r_mu, δ_r_sigma = experiment.right.priors['δ'][1]
	ζ, = experiment.priors['ζ'][1]
	ξ, = experiment.priors['ξ'][1]

	# Get the datasets from the Experiment object
	landing_x_l, subject_indices_l = experiment.left.get_FFT_dataset(data_subset)
	landing_x_r, subject_indices_r = experiment.right.get_FFT_dataset(data_subset)
	
	coords = {
		'condition': ['left', 'right'],
		'subject_l': sorted(list(set(subject_indices_l))),
		'subject_r': sorted(list(set(subject_indices_r))),
	}
	
	with pm.Model(coords=coords) as model:

		# Priors
		if uniform_priors:
			τ = pm.Uniform('τ', np.array([0, 0]), np.array([252, 252]), dims='condition')
			δ = pm.Uniform('δ', np.array([0, 0]), np.array([60, 60]), dims='condition')
		else:
			τ = pm.Normal('τ', mu=np.array([τ_l_mu, τ_r_mu]), sigma=np.array([τ_l_sigma, τ_r_sigma]), dims='condition')
			δ = pm.Gamma('δ', mu=np.array([δ_l_mu,  δ_r_mu]), sigma=np.array([δ_l_sigma, δ_r_sigma]), dims='condition')
		if independent_ζ:
			ζ = pm.Exponential('ζ', lam=ζ, dims='condition')
			μ_l = pm.Normal('μ_l', mu=τ[0], sigma=ζ[0], dims='subject_l')
			μ_r = pm.Normal('μ_r', mu=τ[1], sigma=ζ[1], dims='subject_r')
		else:
			ζ = pm.Exponential('ζ', lam=ζ)
			μ_l = pm.Normal('μ_l', mu=τ[0], sigma=ζ, dims='subject_l')
			μ_r = pm.Normal('μ_r', mu=τ[1], sigma=ζ, dims='subject_r')
		if independent_ξ:
			ξ = pm.Exponential('ξ', lam=ξ, dims='condition')
			σ_l = pm.Gamma('σ_l', mu=δ[0], sigma=ξ[0], dims='subject_l')
			σ_r = pm.Gamma('σ_r', mu=δ[1], sigma=ξ[1], dims='subject_r')
		else:
			ξ = pm.Exponential('ξ', lam=ξ)
			σ_l = pm.Gamma('σ_l', mu=δ[0], sigma=ξ, dims='subject_l')
			σ_r = pm.Gamma('σ_r', mu=δ[1], sigma=ξ, dims='subject_r')

		# Likelihoods
		x_l = pm.Normal('x_l', mu=μ_l[subject_indices_l], sigma=σ_l[subject_indices_l], observed=landing_x_l)
		x_r = pm.Normal('x_r', mu=μ_r[subject_indices_r], sigma=σ_r[subject_indices_r], observed=landing_x_r)

		# Differences between conditions
		Δτ = pm.Deterministic('Δ(τ)', τ[1] - τ[0])
		Δδ = pm.Deterministic('Δ(δ)', δ[1] - δ[0])
		if independent_ζ:
			Δζ = pm.Deterministic('Δ(ζ)', ζ[1] - ζ[0])
		if independent_ξ:
			Δξ = pm.Deterministic('Δ(ξ)', ξ[1] - ξ[0])

		# Sample from posterior
		trace = pm.sample(n_samples, tune=n_tuning_samples, chains=n_chains, cores=N_CORES,
			return_inferencedata=True, idata_kwargs={'log_likelihood': False}
		)
	return trace


def simulate_dataset(params, n_participants, n_test_trials):
	from scipy import stats
	Normal = stats.norm
	Gamma = lambda mu, sigma: stats.gamma(mu**2/sigma**2, scale=1/(mu/sigma**2))
	τ, δ, ζ, ξ = params
	Μ = Normal(τ, ζ).rvs(n_participants)
	Σ = Gamma(δ, ξ).rvs(n_participants)
	landing_x = []
	subject_indices = []
	for subject_i, (μ, σ) in enumerate(zip(Μ, Σ)):
		X = Normal(μ, σ).rvs(n_test_trials)
		landing_x.extend(X)
		subject_indices.extend([subject_i] * n_test_trials)
	return landing_x, subject_indices


if __name__ == '__main__':

	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument('--n_chains', action='store', type=int, default=6, help='number of chains')
	parser.add_argument('--n_samples', action='store', type=int, default=2500, help='number of MCMC samples')
	parser.add_argument('--n_tuning_samples', action='store', type=int, default=500, help='number of MCMC tuning samples')
	parser.add_argument('--data_subset', action='store', type=str, default=None, help='fit one subset independently (above or below)')
	parser.add_argument('--independent_ζ', action='store_true', help="fit independent ζ's")
	parser.add_argument('--independent_ξ', action='store_true', help="fit independent ξ's")
	parser.add_argument('--uniform_priors', action='store_true', help='use uniform priors')
	parser.add_argument('--output_file', action='store', default=None, help='file to write posterior trace to')
	args = parser.parse_args()

	experiment = Experiment('exp2')
	experiment.set_exclusion_threshold(7, 8)
	experiment.set_params({
		'τ': (0, 252),
		'δ': (0, 60),
		'ζ': (0, 40),
		'ξ': (0, 40),
	})
	experiment.set_priors({
		'ζ': ('exponential', (0.1,)),
		'ξ': ('exponential', (0.1,)),
	})
	experiment.left.set_priors({
		'τ': ('normal', (72., 20.)),
		'δ': ('gamma', (20., 8.)),
	})
	experiment.right.set_priors({
		'τ': ('normal', (144., 20.)),
		'δ': ('gamma', (30., 8.)),
	})

	if args.output_file is None:
		output_file = str(experiment.posterior_file)
	else:
		output_file = args.output_file

	trace = fit_posterior(experiment,
		n_chains=args.n_chains,
		n_samples=args.n_samples,
		n_tuning_samples=args.n_tuning_samples,
		data_subset=args.data_subset,
		independent_ζ=args.independent_ζ,
		independent_ξ=args.independent_ξ,
		uniform_priors=args.uniform_priors,
	)
	trace.to_netcdf(output_file, compress=False)
