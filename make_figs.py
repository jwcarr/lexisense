'''
This script builds all the figures and saves them to manuscript/figs/
'''


from code import *

try:
	import mplcairo
	import matplotlib
	matplotlib.use("module://mplcairo.macosx")
except:
	pass

# EXPERIMENT 1
##############################################################################
exp1 = Experiment('exp1')
exp1.set_exclusion_threshold(7, 8)
exp1.set_params({
	'α': ( 0.0625, 0.9999),
	'β': ( 0.0001, 1.0000),
	'γ': (-0.9999, 0.9999),
	'ε': ( 0.0001, 0.9999),
})
exp1.set_priors({
	'α': ('beta', (8, 2)),
	'β': ('beta', (2, 8)),
	'γ': ('beta', (4, 2)),
	'ε': ('beta', (2, 16)),
})
##############################################################################


# EXPERIMENT 2
##############################################################################
exp2 = Experiment('exp2')
exp2.set_exclusion_threshold(7, 8)
exp2.set_params({
	'τ': (0, 252),
	'δ': (0, 60),
	'ζ': (0, 60),
	'ξ': (0, 60),
})
exp2.set_priors({
	'ζ': ('exponential', (0.1,)),
	'ξ': ('exponential', (0.1,)),
})
exp2.left.set_priors({
	'τ': ('normal', (72., 20.)),
	'δ': ('gamma', (20., 8.)),
	
})
exp2.right.set_priors({
	'τ': ('normal', (144., 20.)),
	'δ': ('gamma', (30., 8.)),
})
##############################################################################


# FIGURE 2
##############################################################################
languages = {
	'nl': 'Dutch',
	'en': 'English',
	'de': 'German',
	'gr': 'Greek',
	'he': 'Hebrew',
	'it': 'Italian',
	'pl': 'Polish',
	'es': 'Spanish',
	'sw': 'Swahili',
}
file_path = FIGS / 'lang_uncertainty.eps'
with Figure(file_path, n_rows=9, n_cols=5, width='double', height=180) as fig:
	for i, (lang, lang_name) in enumerate(languages.items()):
		uncertainty_symm = json_read(DATA / 'lang_uncertainty' / 'gamma0.0' / f'{lang}.json')
		uncertainty_RVFA = json_read(DATA / 'lang_uncertainty' / 'gamma0.5' / f'{lang}.json')
		if lang == 'he':
			uncertainty_LVFA = json_read(DATA / 'lang_uncertainty' / 'gamma-0.5' / f'{lang}.json')
		for j, length in enumerate(range(5, 10)):
			plots.plot_uncertainty(fig[i,j], uncertainty_RVFA[str(length)], color='MediumSeaGreen', show_min=True)
			if lang == 'he':
				plots.plot_uncertainty(fig[i,j], uncertainty_LVFA[str(length)], color='DeepSkyBlue', show_min=True)
			plots.plot_uncertainty(fig[i,j], uncertainty_symm[str(length)], color='black', show_min=True)
			fig[i,j].set_xlabel(f'{length}-letter words')
			fig[i,j].set_ylabel(lang_name)
			fig[i,j].set_ylim(0, 5)
##############################################################################


# FIGURE 4
##############################################################################
file_path = FIGS / 'exp1_results.eps'
with Figure(file_path, n_rows=1, n_cols=2, width='single', height=60) as fig:
	plots.plot_test_curve(fig[0,0], exp1.left, show_individuals=True)
	plots.plot_test_curve(fig[0,1], exp1.right, show_individuals=True)
	plots.draw_brace(fig[0,0], (2,4), 0.2, 'High\ninfo')
	plots.draw_brace(fig[0,0], (6,7), 0.2, 'Low\ninfo')
	plots.draw_brace(fig[0,1], (4,6), 0.2, 'High\ninfo')
	plots.draw_brace(fig[0,1], (1,2), 0.2, 'Low\ninfo')
##############################################################################


# FIGURE 5a
##############################################################################
file_path = FIGS / 'exp1_posteriors.svg'
with Figure(file_path, n_cols=4, n_rows=1, width='double', height=40) as fig:
	for param, axis in zip(['α', 'β', 'γ', 'ε'], fig):
		plots.plot_prior(axis, exp1, param, transform_to_param_bounds=True)
		plots.plot_posterior(axis, exp1, param)
##############################################################################

# FIGURE 5b
##############################################################################
from code import model_fit
sim_datasets = model_fit.simulate_from_posterior(exp1, n_sims=100)
file_path = FIGS / 'exp1_predictive.svg'
with Figure(file_path, n_cols=2, n_rows=1, width='single', height=40) as fig:
	plots.plot_posterior_predictive(fig[0,0], sim_datasets, exp1.left, lexicon_index=0, show_legend=False)
	plots.plot_posterior_predictive(fig[0,1], sim_datasets, exp1.right, lexicon_index=1)
	fig[0,0].set_ylabel('Probability correct')
	fig[0,1].set_ylabel('Probability correct')
##############################################################################

# FIGURE 5c
##############################################################################
from code import model_fit
uncertainty_left, uncertainty_right = model_fit.uncertainty_curve_from_posterior(exp1, n_sims=10000)
file_path = FIGS / 'exp1_uncertainty.svg'
with Figure(file_path, n_cols=2, n_rows=1, width='single', height=40) as fig:
	plots.plot_uncertainty(fig[0,0], uncertainty_left, color=exp1.left.color)
	plots.plot_uncertainty(fig[0,1], uncertainty_right, color=exp1.right.color)
	fig[0,0].set_ylim(0, 1)
	fig[0,1].set_ylim(0, 1)
##############################################################################


# FIGURE 6
##############################################################################
file_path = FIGS / 'exp2_results.eps'
with Figure(file_path, n_rows=1, n_cols=2, width='single', height=60) as fig:
	plots.plot_landing_curve(fig[0,0], exp2.left, show_individuals=True, show_average=True)
	plots.plot_landing_curve(fig[0,1], exp2.right, show_individuals=True, show_average=True)
##############################################################################


# FIGURE 7
##############################################################################
file_path = FIGS / 'exp2_landing_image.svg'
plots.landing_position_image(exp2, file_path)
##############################################################################


# FIGURE 8
##############################################################################
file_path = FIGS / 'exp2_posteriors.eps'
with Figure(file_path, n_cols=2, n_rows=2, width='single', height=80) as fig:
	plots.plot_prior(fig[0,0], exp2, 'τ')
	plots.plot_prior(fig[0,1], exp2, 'δ')
	plots.plot_posterior(fig[0,0], exp2, 'τ')
	plots.plot_posterior(fig[0,1], exp2, 'δ')
	plots.plot_posterior_difference(fig[1,0], exp2, 'τ', hdi=0.95, rope=(-9, 9), xlim=(-9, 49))
	plots.plot_posterior_difference(fig[1,1], exp2, 'δ', hdi=0.95, rope=(-4, 4), xlim=(-12, 12))
	plots.draw_letter_grid(fig[0,0], letter_width=36, n_letters=7)
##############################################################################


# EXTENDED DATA: FIGURE 9
##############################################################################
file_path = FIGS / 'exp1_posteriors_alt.eps'
with Figure(file_path, n_cols=2, n_rows=2, width='double', height=80) as fig:
	for param, axis in zip(['α', 'β', 'γ', 'ε'], fig):
		plots.plot_prior(axis, exp1, param, label='Prior', transform_to_param_bounds=True)
		plots.plot_posterior(axis, exp1, param, label='Posterior')
		plots.plot_posterior(axis, exp1, param, label='Posterior using uniform prior', posterior_file=MODEL_FIT / 'exp1_posterior_uniform.nc', linestyle=':')
		plots.plot_posterior(axis, exp1.left, param, label='Posterior using left-heavy dataset only', posterior_file=MODEL_FIT / 'exp1_posterior_left.nc')
		plots.plot_posterior(axis, exp1.right, param, label='Posterior using right-heavy dataset only', posterior_file=MODEL_FIT / 'exp1_posterior_right.nc')
		fig[0,0].legend(frameon=False)
##############################################################################


# EXTENDED DATA: FIGURE 10
##############################################################################
file_path = FIGS / 'exp2_posteriors_ab.svg'
with Figure(file_path, n_cols=4, n_rows=2, width='double', height=80) as fig:
	# above
	plots.plot_prior(fig[0,0], exp2, 'τ')
	plots.plot_prior(fig[0,1], exp2, 'δ')
	plots.plot_posterior(fig[0,0], exp2, 'τ', posterior_file=MODEL_FIT / 'exp2_posterior_above.nc')
	plots.plot_posterior(fig[0,1], exp2, 'δ', posterior_file=MODEL_FIT / 'exp2_posterior_above.nc')
	plots.plot_posterior_difference(fig[1,0], exp2, 'τ', hdi=0.95, rope=(-9, 9), xlim=(-9, 59), posterior_file=MODEL_FIT / 'exp2_posterior_above.nc')
	plots.plot_posterior_difference(fig[1,1], exp2, 'δ', hdi=0.95, rope=(-4, 4), xlim=(-14, 14), posterior_file=MODEL_FIT / 'exp2_posterior_above.nc')
	plots.draw_letter_grid(fig[0,0], letter_width=36, n_letters=7)
	# below
	plots.plot_prior(fig[0,2], exp2, 'τ')
	plots.plot_prior(fig[0,3], exp2, 'δ')
	plots.plot_posterior(fig[0,2], exp2, 'τ', posterior_file=MODEL_FIT / 'exp2_posterior_below.nc')
	plots.plot_posterior(fig[0,3], exp2, 'δ', posterior_file=MODEL_FIT / 'exp2_posterior_below.nc')
	plots.plot_posterior_difference(fig[1,2], exp2, 'τ', hdi=0.95, rope=(-9, 9), xlim=(-9, 59), posterior_file=MODEL_FIT / 'exp2_posterior_below.nc')
	plots.plot_posterior_difference(fig[1,3], exp2, 'δ', hdi=0.95, rope=(-4, 4), xlim=(-14, 14), posterior_file=MODEL_FIT / 'exp2_posterior_below.nc')
	plots.draw_letter_grid(fig[0,2], letter_width=36, n_letters=7)
##############################################################################


# EXTENDED DATA: FIGURE 11
##############################################################################
file_path = FIGS / 'exp2_posteriors_alt.svg'
with Figure(file_path, n_cols=4, n_rows=2, width='double', height=80) as fig:
	# independent ζ and ξ
	plots.plot_prior(fig[0,2], exp2, 'τ')
	plots.plot_prior(fig[0,3], exp2, 'δ')
	plots.plot_posterior(fig[0,2], exp2, 'τ', posterior_file=MODEL_FIT / 'exp2_posterior_indy_ζξ.nc')
	plots.plot_posterior(fig[0,3], exp2, 'δ', posterior_file=MODEL_FIT / 'exp2_posterior_indy_ζξ.nc')
	plots.plot_posterior_difference(fig[1,2], exp2, 'τ', hdi=0.95, rope=(-9, 9), xlim=(-9, 59), posterior_file=MODEL_FIT / 'exp2_posterior_indy_ζξ.nc')
	plots.plot_posterior_difference(fig[1,3], exp2, 'δ', hdi=0.95, rope=(-4, 4), xlim=(-14, 14), posterior_file=MODEL_FIT / 'exp2_posterior_indy_ζξ.nc')
	plots.draw_letter_grid(fig[0,2], letter_width=36, n_letters=7)
	# uniform priors
	exp2.set_priors({
		'τ': ('uniform', (0., 252.)),
		'δ': ('uniform', (0., 60.)),
	})
	plots.plot_prior(fig[0,0], exp2, 'τ')
	plots.plot_prior(fig[0,1], exp2, 'δ')
	plots.plot_posterior(fig[0,0], exp2, 'τ', posterior_file=MODEL_FIT / 'exp2_posterior_uniform.nc')
	plots.plot_posterior(fig[0,1], exp2, 'δ', posterior_file=MODEL_FIT / 'exp2_posterior_uniform.nc')
	plots.plot_posterior_difference(fig[1,0], exp2, 'τ', hdi=0.95, rope=(-9, 9), xlim=(-9, 59), posterior_file=MODEL_FIT / 'exp2_posterior_uniform.nc')
	plots.plot_posterior_difference(fig[1,1], exp2, 'δ', hdi=0.95, rope=(-4, 4), xlim=(-14, 14), posterior_file=MODEL_FIT / 'exp2_posterior_uniform.nc')
	plots.draw_letter_grid(fig[0,0], letter_width=36, n_letters=7)
##############################################################################






# Plot individual participant results for Experiment 1
##############################################################################
# for participant in exp1.iter_with_excludes():
# 	file_name = f'{participant.task_id}_{participant.ID}.pdf'
# 	file_path = RESULTS / 'exp1' / 'individual_results' / file_name
# 	with Figure(file_path, n_cols=2, width=150) as fig:
# 		plots.plot_learning_curve(fig[0,0], participant)
# 		plots.plot_test_curve(fig[0,1], participant)
##############################################################################


# Plot individual participant results for Experiment 2
##############################################################################
# for participant in exp2.iter_with_excludes():
# 	file_name = f'{participant.task_id}_{participant.ID}.pdf'
# 	file_path = RESULTS / 'exp2' / 'individual_results' / file_name
# 	with Figure(file_path, n_cols=2, width=150) as fig:
# 		plots.plot_learning_curve(fig[0,0], participant)
# 		plots.plot_landing_curve(fig[0,1], participant)
##############################################################################


# Create screen images of individual trials from Experiment 2
##############################################################################
# for participant in exp2.iter_with_excludes():
# 	file_path = RESULTS / 'exp2' / 'individual_results' / f'{participant.task_id}_{participant.ID}'
# 	if not file_path.exists():
# 		file_path.mkdir(parents=True)
# 	for i, trial in enumerate(participant.iter_free_fixation_trials()):
# 		img = plots.make_trial_image(participant, trial)
# 		img.save(file_path / f'{i}.pdf')
##############################################################################
