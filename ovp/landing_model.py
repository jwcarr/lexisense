import pymc3 as pm
import numpy as np


def fit_posterior(experiment, n_samples=1000, n_tuning_samples=1000, n_chains=2, n_cores=2, independent_ζ=False, independent_ξ=False):

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

		# Priors
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
		trace = pm.sample(n_samples, tune=n_tuning_samples, chains=n_chains, cores=n_cores,
			return_inferencedata=True, idata_kwargs={'log_likelihood': False}
		)

	trace.to_netcdf(experiment.posterior_file)
