from pathlib import Path
import pandas as pd
import numpy as np
import pymc as pm
import arviz as az


ROOT = Path(__file__).parent.parent.resolve()
EXP_DATA = ROOT / 'data' / 'experiments'
MODEL_FIT = ROOT / 'data' / 'model_fit'
N_CORES = 6


def get_landing_positions_for_condition(dataset, condition):
	dataset = dataset[ dataset['condition'] == condition ]
	dataset = dataset[ dataset['trial_type'] == 'test' ]
	dataset = dataset[ dataset['excluded'] is False ]
	dataset = dataset.dropna(subset='landing_position')
	subject_indices, _ = dataset.subject.factorize()
	return dataset.landing_position, subject_indices


def fit_basic_model(dataset, n_chains=6, n_samples=10000, n_tuning_samples=1000):
	landing_x_l, subject_indices_l = get_landing_positions_for_condition(dataset, 'left')
	landing_x_r, subject_indices_r = get_landing_positions_for_condition(dataset, 'right')
	coords = {
		'condition': ['left', 'right'],
		'subject_l': sorted(list(set(subject_indices_l))),
		'subject_r': sorted(list(set(subject_indices_r))),
	}
	with pm.Model(coords=coords) as basic_model:

		τ = pm.Normal('τ', mu=np.array([94, 128]), sigma=np.array([30, 30]), dims='condition')
		δ = pm.Gamma( 'δ', mu=23, sigma=10)
		ζ = pm.Exponential('ζ', lam=0.1)
		ξ = pm.Exponential('ξ', lam=0.1)

		μ_l = pm.Normal('μ_l', mu=τ[0], sigma=ζ, dims='subject_l')
		μ_r = pm.Normal('μ_r', mu=τ[1], sigma=ζ, dims='subject_r')
		σ_l = pm.Gamma('σ_l', mu=δ, sigma=ξ, dims='subject_l')
		σ_r = pm.Gamma('σ_r', mu=δ, sigma=ξ, dims='subject_r')
		
		x_l = pm.Normal('x_l', mu=μ_l[subject_indices_l], sigma=σ_l[subject_indices_l], observed=landing_x_l)
		x_r = pm.Normal('x_r', mu=μ_r[subject_indices_r], sigma=σ_r[subject_indices_r], observed=landing_x_r)
		
		Δτ = pm.Deterministic('Δ(τ)', τ[1] - τ[0])
		
		trace = pm.sample(n_samples, tune=n_tuning_samples, chains=n_chains, cores=N_CORES,
			return_inferencedata=True, idata_kwargs={'log_likelihood': False}
		)
	
	return trace


def fit_full_model(dataset, n_chains=6, n_samples=10000, n_tuning_samples=1000):
	landing_x_l, subject_indices_l = get_landing_positions_for_condition(dataset, 'left')
	landing_x_r, subject_indices_r = get_landing_positions_for_condition(dataset, 'right')
	coords = {
		'condition': ['left', 'right'],
		'subject_l': sorted(list(set(subject_indices_l))),
		'subject_r': sorted(list(set(subject_indices_r))),
	}
	with pm.Model(coords=coords) as full_model:

		τ = pm.Normal('τ', mu=np.array([94, 128]), sigma=np.array([30, 30]), dims='condition')
		δ = pm.Gamma( 'δ', mu=np.array([22,  24]), sigma=np.array([10, 10]), dims='condition')
		ζ = pm.Exponential('ζ', lam=np.array([0.1, 0.1]), dims='condition')
		ξ = pm.Exponential('ξ', lam=np.array([0.1, 0.1]), dims='condition')
		
		μ_l = pm.Normal('μ_l', mu=τ[0], sigma=ζ[0], dims='subject_l')
		μ_r = pm.Normal('μ_r', mu=τ[1], sigma=ζ[1], dims='subject_r')
		σ_l = pm.Gamma( 'σ_l', mu=δ[0], sigma=ξ[0], dims='subject_l')
		σ_r = pm.Gamma( 'σ_r', mu=δ[1], sigma=ξ[1], dims='subject_r')
		
		x_l = pm.Normal('x_l', mu=μ_l[subject_indices_l], sigma=σ_l[subject_indices_l], observed=landing_x_l)
		x_r = pm.Normal('x_r', mu=μ_r[subject_indices_r], sigma=σ_r[subject_indices_r], observed=landing_x_r)
		
		Δτ = pm.Deterministic('Δ(τ)', τ[1] - τ[0])
		Δδ = pm.Deterministic('Δ(δ)', δ[1] - δ[0])
		Δζ = pm.Deterministic('Δ(ζ)', ζ[1] - ζ[0])
		Δξ = pm.Deterministic('Δ(ξ)', ξ[1] - ξ[0])
		
		trace = pm.sample(n_samples, tune=n_tuning_samples, chains=n_chains, cores=N_CORES,
			return_inferencedata=True, idata_kwargs={'log_likelihood': False}
		)

	return trace


def fit_uniform_model(dataset, n_chains=6, n_samples=10000, n_tuning_samples=1000):
	landing_x_l, subject_indices_l = get_landing_positions_for_condition(dataset, 'left')
	landing_x_r, subject_indices_r = get_landing_positions_for_condition(dataset, 'right')
	coords = {
		'condition': ['left', 'right'],
		'subject_l': sorted(list(set(subject_indices_l))),
		'subject_r': sorted(list(set(subject_indices_r))),
	}
	with pm.Model(coords=coords) as basic_model:

		τ = pm.Uniform('τ', np.array([0, 0]), np.array([252, 252]), dims='condition')
		δ = pm.Uniform('δ', 0, 60)
		ζ = pm.Exponential('ζ', lam=0.1)
		ξ = pm.Exponential('ξ', lam=0.1)

		μ_l = pm.Normal('μ_l', mu=τ[0], sigma=ζ, dims='subject_l')
		μ_r = pm.Normal('μ_r', mu=τ[1], sigma=ζ, dims='subject_r')
		σ_l = pm.Gamma('σ_l', mu=δ, sigma=ξ, dims='subject_l')
		σ_r = pm.Gamma('σ_r', mu=δ, sigma=ξ, dims='subject_r')
		
		x_l = pm.Normal('x_l', mu=μ_l[subject_indices_l], sigma=σ_l[subject_indices_l], observed=landing_x_l)
		x_r = pm.Normal('x_r', mu=μ_r[subject_indices_r], sigma=σ_r[subject_indices_r], observed=landing_x_r)
		
		Δτ = pm.Deterministic('Δ(τ)', τ[1] - τ[0])
		
		trace = pm.sample(n_samples, tune=n_tuning_samples, chains=n_chains, cores=N_CORES,
			return_inferencedata=True, idata_kwargs={'log_likelihood': False}
		)
	
	return trace


if __name__ == '__main__':

	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument('experiment_id', action='store', type=str, help='experiment ID')
	parser.add_argument('--n_chains', action='store', type=int, default=6, help='number of chains')
	parser.add_argument('--n_samples', action='store', type=int, default=2500, help='number of MCMC samples')
	parser.add_argument('--n_tuning_samples', action='store', type=int, default=500, help='number of MCMC tuning samples')
	parser.add_argument('--fit_full_model', action='store_true', help='fit the full model with all hyperparameters stratified')
	parser.add_argument('--fit_uniform_model', action='store_true', help='fit the full model with all hyperparameters stratified')
	args = parser.parse_args()

	dataset = pd.read_csv(EXP_DATA / f'{args.experiment_id}.csv')

	if args.fit_full_model:
		trace = fit_full_model(dataset, args.n_chains, args.n_samples, args.n_tuning_samples)
		trace.to_netcdf(MODEL_FIT / f'{args.experiment_id}_posterior_full.nc', compress=False)
	elif args.fit_uniform_model:
		trace = fit_uniform_model(dataset, args.n_chains, args.n_samples, args.n_tuning_samples)
		trace.to_netcdf(MODEL_FIT / f'{args.experiment_id}_posterior_uniform.nc', compress=False)
	else:
		trace = fit_basic_model(dataset, args.n_chains, args.n_samples, args.n_tuning_samples)
		trace.to_netcdf(MODEL_FIT / f'{args.experiment_id}_posterior.nc', compress=False)
	
	lower, upper = az.hdi(trace.posterior['Δ(τ)'].to_numpy().flatten(), hdi_prob=0.95)
	print('95% HDI width for Δ(τ):', upper - lower)
