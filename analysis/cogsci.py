import ovp


'''
Plot distribution of uncertainty in English and Polish.
'''
##############################################################################
# from ovp import plots

# languages = {
# 	'en': 'English',
# 	'pl': 'Polish',
# }

# file_path = ovp.ROOT / 'manuscript' / 'cogsci' / 'figs' / 'lang_uncertainty.eps'
# with ovp.Figure(file_path, n_rows=1, n_cols=2, width=85, height=32) as fig:
# 	for i, (lang, lang_name) in enumerate(languages.items()):
# 		uncertainty_symm = ovp.json_read(ovp.DATA/'lang_uncertainty'/'gamma0.0'/f'{lang}.json')
# 		uncertainty_asymm = ovp.json_read(ovp.DATA/'lang_uncertainty'/'gamma0.5'/f'{lang}.json')
# 		plots.plot_uncertainty(fig[0,i], uncertainty_asymm[str(7)], color='MediumSeaGreen', show_min=True)
# 		plots.plot_uncertainty(fig[0,i], uncertainty_symm[str(7)], color='black', show_min=True)
# 		fig[0,i].text(4, 3.4, lang_name, ha='center')
# 		fig[0,i].set_ylim(0, 4)
##############################################################################


'''
Plot Experiment 1 results, posteriors, and extimated uncertainty curves.
'''
##############################################################################
# from ovp import model_fit, plots

# experiment = ovp.Experiment('exp1')
# experiment.set_exclusion_threshold(7, 8)
# experiment.set_params({
# 	'α': ( 0.0625, 0.9999),
# 	'β': ( 0.0001, 1.0000),
# 	'γ': (-0.9999, 0.9999),
# 	'ε': ( 0.0001, 0.9999),
# })
# experiment.set_priors({
# 	'α': ('beta', (8, 2)),
# 	'β': ('beta', (2, 8)),
# 	'γ': ('beta', (4, 2)),
# 	'ε': ('beta', (2, 16)),
# })

# file_path = ovp.ROOT / 'manuscript' / 'cogsci' / 'figs' / 'exp1_results.eps'
# with ovp.Figure(file_path, n_cols=2, width=85, height=50) as fig:
# 	plots.plot_test_curve(fig[0,0], experiment.left, show_individuals=True)
# 	plots.plot_test_curve(fig[0,1], experiment.right, show_individuals=True)
# 	plots.draw_brace(fig[0,0], (2,4), 0.2, 'High\ninfo')
# 	plots.draw_brace(fig[0,0], (6,7), 0.2, 'Low\ninfo')
# 	plots.draw_brace(fig[0,1], (4,6), 0.2, 'High\ninfo')
# 	plots.draw_brace(fig[0,1], (1,2), 0.2, 'Low\ninfo')

# file_path = ovp.ROOT / 'manuscript' / 'cogsci' / 'figs' / 'exp1_posteriors.eps'
# with ovp.Figure(file_path, n_cols=2, n_rows=2, width=85, height=50) as fig:
# 	for param, axis in zip(['α', 'β', 'γ', 'ε'], fig):
# 		plots.plot_prior(axis, experiment, param, transform_to_param_bounds=True)
# 		plots.plot_posterior(axis, experiment, param)
# 	fig[0,0].set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
# 	fig[0,1].set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
# 	fig[1,0].set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
# 	fig[1,1].set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
# 	fig[0,0].set_ylabel('Density')
# 	fig[1,0].set_ylabel('Density')
# 	fig.auto_deduplicate_axes = False

# uncertainty_left, uncertainty_right = model_fit.uncertainty_curve_from_posterior(experiment, n_sims=10000)
# file_path = ovp.ROOT / 'manuscript' / 'cogsci' / 'figs' / 'exp1_uncertainty.eps'
# with ovp.Figure(file_path, n_cols=2, n_rows=1, width=85, height=32) as fig:
# 	plots.plot_uncertainty(fig[0,0], uncertainty_left, color=experiment.left.color)
# 	plots.plot_uncertainty(fig[0,1], uncertainty_right, color=experiment.right.color)
# 	fig[0,0].set_ylim(0, 1)
# 	fig[0,1].set_ylim(0, 1)
##############################################################################


'''
Plot Experiemnt 2 posteriors
'''
##############################################################################
# from ovp import plots

# experiment = ovp.Experiment('exp2')
# experiment.set_params({
# 	'τ': (0, 252),
# 	'δ': (0, 60),
# 	'ζ': (0, 60),
# 	'ξ': (0, 60),
# })
# experiment.set_priors({
# 	'ζ': ('exponential', (0.1,)),
# 	'ξ': ('exponential', (0.1,)),
# })
# experiment.left.set_priors({
# 	'τ': ('normal', (72., 20.)),
# 	'δ': ('gamma', (20., 8.)),
	
# })
# experiment.right.set_priors({
# 	'τ': ('normal', (144., 20.)),
# 	'δ': ('gamma', (30., 8.)),
# })
# experiment.set_exclusion_threshold(7, 8)

# file_path = ovp.ROOT / 'manuscript' / 'cogsci' / 'figs' / 'exp2_posteriors.eps'
# with ovp.Figure(file_path, n_cols=2, n_rows=2, width=85, height=60) as fig:
# 	plots.plot_prior(fig[0,0], experiment, 'τ')
# 	plots.plot_prior(fig[0,1], experiment, 'δ')
# 	plots.plot_posterior(fig[0,0], experiment, 'τ')
# 	plots.plot_posterior(fig[0,1], experiment, 'δ')
# 	plots.plot_posterior_difference(fig[1,0], experiment, 'τ', hdi=0.95, rope=(-9, 9))
# 	plots.plot_posterior_difference(fig[1,1], experiment, 'δ', hdi=0.95, rope=(-4, 4))
# 	plots.draw_letter_grid(fig[0,0], letter_width=36, n_letters=7)
##############################################################################
