import pymc3 as pm
import numpy as np
from scipy import stats


def fit_posterior(experiment, n_samples=1000, n_tuning_samples=1000, n_chains=2):

	# Get relevant prior params from Experiment object
	τ_l_mu, τ_l_sigma = experiment.left.priors['τ'][1]
	δ_l_mu, δ_l_sigma = experiment.left.priors['δ'][1]
	τ_r_mu, τ_r_sigma = experiment.right.priors['τ'][1]
	δ_r_mu, δ_r_sigma = experiment.right.priors['δ'][1]
	ζ, = experiment.priors['ζ'][1]
	ξ, = experiment.priors['ξ'][1]

	# Get the datasets from the Experiment object
	landing_x_l, subject_indices_l = experiment.left.get_FFT_dataset()
	landing_x_r, subject_indices_r = experiment.right.get_FFT_dataset()
	
	coords = {
		'condition': ['left', 'right'],
		'subject_l': sorted(list(set(subject_indices_l))),
		'subject_r': sorted(list(set(subject_indices_r))),
	}
	
	with pm.Model(coords=coords) as model:

		# Hyperpriors
		τ = pm.Normal('τ', mu=np.array([τ_l_mu, τ_r_mu]), sigma=np.array([τ_l_sigma, τ_r_sigma]), dims='condition')
		δ = pm.Gamma('δ', mu=np.array([δ_l_mu,  δ_r_mu]), sigma=np.array([δ_l_sigma, δ_r_sigma]), dims='condition')
		ζ = pm.Exponential('ζ', lam=ζ)
		ξ = pm.Exponential('ξ', lam=ξ)

		# Priors
		μ_l = pm.Normal('μ_l', mu=τ[0], sigma=ζ, dims='subject_l')
		σ_l = pm.Gamma('σ_l', mu=δ[0], sigma=ξ, dims='subject_l')
		μ_r = pm.Normal('μ_r', mu=τ[1], sigma=ζ, dims='subject_r')
		σ_r = pm.Gamma('σ_r', mu=δ[1], sigma=ξ, dims='subject_r')

		# Likelihood
		x_l = pm.Normal('x_l', mu=μ_l[subject_indices_l], sigma=σ_l[subject_indices_l], observed=landing_x_l)
		x_r = pm.Normal('x_r', mu=μ_r[subject_indices_r], sigma=σ_r[subject_indices_r], observed=landing_x_r)

		# Differences between conditions
		Δτ = pm.Deterministic('Δ(τ)', τ[1] - τ[0])
		Δδ = pm.Deterministic('Δ(δ)', δ[1] - δ[0])

		trace = pm.sample(n_samples, tune=n_tuning_samples, chains=n_chains, cores=1, target_accept=0.99,
			trace=[τ, δ, ζ, ξ, Δτ, Δδ], return_inferencedata=True, idata_kwargs={'log_likelihood': False}
		)

	trace.to_netcdf(experiment.posterior_file)
