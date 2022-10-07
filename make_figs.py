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


# EXPERIMENT 2
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


def figure2():
	file_path = FIGS / 'fig2.eps'
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


def figure4():
	file_path = FIGS / 'fig4.eps'
	with Figure(file_path, n_rows=1, n_cols=2, width='single', height=60) as fig:
		plots.plot_test_curve(fig[0,0], exp1.left, show_individuals=True)
		plots.plot_test_curve(fig[0,1], exp1.right, show_individuals=True)
		plots.draw_brace(fig[0,0], (2,4), 0.2, 'High\ninfo')
		plots.draw_brace(fig[0,0], (6,7), 0.2, 'Low\ninfo')
		plots.draw_brace(fig[0,1], (4,6), 0.2, 'High\ninfo')
		plots.draw_brace(fig[0,1], (1,2), 0.2, 'Low\ninfo')


def figure5():
	from code import model_predict

	file_path = FIGS / 'fig5a.svg'
	with Figure(file_path, n_cols=4, n_rows=1, width='double', height=40) as fig:
		for param, axis in zip(['α', 'β', 'γ', 'ε'], fig):
			plots.plot_prior(axis, exp1, param, transform_to_param_bounds=True)
			plots.plot_posterior(axis, exp1, param)

	sim_datasets = model_predict.simulate_from_posterior(exp1, n_sims=100)
	file_path = FIGS / 'fig5b.svg'
	with Figure(file_path, n_cols=2, n_rows=1, width='single', height=40) as fig:
		plots.plot_posterior_predictive(fig[0,0], sim_datasets, exp1.left, lexicon_index=0, show_legend=False)
		plots.plot_posterior_predictive(fig[0,1], sim_datasets, exp1.right, lexicon_index=1)
		fig[0,0].set_ylabel('Probability correct')
		fig[0,1].set_ylabel('Probability correct')

	uncertainty_left, uncertainty_right = model_predict.uncertainty_curve_from_posterior(exp1, n_sims=10000)
	file_path = FIGS / 'fig5c.svg'
	with Figure(file_path, n_cols=2, n_rows=1, width='single', height=40) as fig:
		plots.plot_uncertainty(fig[0,0], uncertainty_left, color=exp1.left.color)
		plots.plot_uncertainty(fig[0,1], uncertainty_right, color=exp1.right.color)
		fig[0,0].set_ylim(0, 1.3)
		fig[0,1].set_ylim(0, 1.3)


def figure6():
	file_path = FIGS / 'fig6.svg'
	with Figure(file_path, n_rows=1, n_cols=2, width='single', height=60) as fig:
		plots.plot_landing_curve(fig[0,0], exp2.left, show_individuals=True, show_average=True)
		plots.plot_landing_curve(fig[0,1], exp2.right, show_individuals=True, show_average=True)
	file_path = FIGS / 'fig6b.svg'
	plots.landing_position_image(exp2, file_path)


def figure7():
	file_path = FIGS / 'fig7.eps'
	with Figure(file_path, n_cols=2, n_rows=2, width='single', height=80) as fig:
		plots.plot_prior(fig[0,0], exp2, 'τ')
		plots.plot_prior(fig[0,1], exp2, 'δ')
		plots.plot_posterior(fig[0,0], exp2, 'τ')
		plots.plot_posterior(fig[0,1], exp2, 'δ')
		plots.plot_posterior_difference(fig[1,0], exp2, 'τ', hdi=0.95, rope=(-9, 9), xlim=(-9, 49))
		plots.plot_posterior_difference(fig[1,1], exp2, 'δ', hdi=0.95, rope=(-4, 4), xlim=(-12, 12))
		plots.draw_letter_grid(fig[0,0], letter_width=36, n_letters=7)


def figure8():
	from code import model

	file_path = FIGS / 'fig8.eps'

	WORD_LENGTH = 7
	N_SYMBOLS = 26

	lexicon = [(i,) * WORD_LENGTH for i in range(N_SYMBOLS)]
	alpha_vals = [0.9, 0.7, 0.5]
	beta_gamma_vals = [
		(0.2, -0.5), (0.4, -0.5), (0.8, -0.5),
		(0.2,  0.0), (0.4,  0.0), (0.8,  0.0),
		(0.2,  0.5), (0.4,  0.5), (0.8,  0.5)
	]

	n_rows = len(alpha_vals)
	n_cols = len(beta_gamma_vals)
	with Figure(file_path, n_rows, n_cols, width='double', height=60) as fig:
		for i, alpha in enumerate(alpha_vals):
			for j, (beta, gamma) in enumerate(beta_gamma_vals):
				fig[i,j].plot((-1, WORD_LENGTH), (1/N_SYMBOLS, 1/N_SYMBOLS), linestyle=':', linewidth=1, color='black')
				reader = model.Reader(lexicon, alpha, beta, gamma)
				for dist in reader.phi:
					fig[i,j].plot(dist)
				fig[i,j].set_ylim(0,1)
				fig[i,j].set_xlim(-0.25, WORD_LENGTH-0.75)
				fig[i,j].set_xticks(range(WORD_LENGTH))
				if i < len(alpha_vals) - 1:
					fig[i,j].set_xticklabels([])
				else:
					fig[i,j].set_xticklabels(range(1,WORD_LENGTH+1))
				if i == len(alpha_vals)-1:
					fig[i,j].set_xlabel(f'$β$ = {beta}')
				if j == 0:
					fig[i,j].set_ylabel(f'$α$ = {alpha}')
				if i == 0 and (j-1) % 3 == 0:
					if gamma < 0:
						fig[i,j].set_title(f'$γ$ = {gamma} (left-visual-field advantage)', fontsize=7)
					elif gamma > 0:
						fig[i,j].set_title(f'$γ$ = {gamma} (right-visual-field advantage)', fontsize=7)
					else:
						fig[i,j].set_title(f'$γ$ = 0 (symmetric visual span)', fontsize=7)


# EXTENDED DATA

def figure9():
	file_path = FIGS / 'fig9.eps'
	with Figure(file_path, n_cols=2, n_rows=2, width='double', height=80) as fig:
		for param, axis in zip(['α', 'β', 'γ', 'ε'], fig):
			plots.plot_prior(axis, exp1, param, label='Prior', transform_to_param_bounds=True)
			plots.plot_posterior(axis, exp1, param, label='Posterior')
			plots.plot_posterior(axis, exp1, param, label='Posterior using uniform prior', posterior_file=MODEL_FIT / 'exp1_posterior_uniform.nc', linestyle=':')
			plots.plot_posterior(axis, exp1.left, param, label='Posterior using left-heavy dataset only', posterior_file=MODEL_FIT / 'exp1_posterior_left.nc')
			plots.plot_posterior(axis, exp1.right, param, label='Posterior using right-heavy dataset only', posterior_file=MODEL_FIT / 'exp1_posterior_right.nc')
			fig[0,0].legend(frameon=False)


def figure10():
	file_path = FIGS / 'fig10.svg'
	with Figure(file_path, n_cols=4, n_rows=4, width='double', height=160) as fig:
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
		# independent ζ and ξ
		plots.plot_prior(fig[2,2], exp2, 'τ')
		plots.plot_prior(fig[2,3], exp2, 'δ')
		plots.plot_posterior(fig[2,2], exp2, 'τ', posterior_file=MODEL_FIT / 'exp2_posterior_indy_ζξ.nc')
		plots.plot_posterior(fig[2,3], exp2, 'δ', posterior_file=MODEL_FIT / 'exp2_posterior_indy_ζξ.nc')
		plots.plot_posterior_difference(fig[3,2], exp2, 'τ', hdi=0.95, rope=(-9, 9), xlim=(-9, 59), posterior_file=MODEL_FIT / 'exp2_posterior_indy_ζξ.nc')
		plots.plot_posterior_difference(fig[3,3], exp2, 'δ', hdi=0.95, rope=(-4, 4), xlim=(-14, 14), posterior_file=MODEL_FIT / 'exp2_posterior_indy_ζξ.nc')
		plots.draw_letter_grid(fig[2,2], letter_width=36, n_letters=7)
		# uniform priors
		exp2.set_priors({
			'τ': ('uniform', (0., 252.)),
			'δ': ('uniform', (0., 60.)),
		})
		plots.plot_prior(fig[2,0], exp2, 'τ')
		plots.plot_prior(fig[2,1], exp2, 'δ')
		plots.plot_posterior(fig[2,0], exp2, 'τ', posterior_file=MODEL_FIT / 'exp2_posterior_uniform.nc')
		plots.plot_posterior(fig[2,1], exp2, 'δ', posterior_file=MODEL_FIT / 'exp2_posterior_uniform.nc')
		plots.plot_posterior_difference(fig[3,0], exp2, 'τ', hdi=0.95, rope=(-9, 9), xlim=(-9, 59), posterior_file=MODEL_FIT / 'exp2_posterior_uniform.nc')
		plots.plot_posterior_difference(fig[3,1], exp2, 'δ', hdi=0.95, rope=(-4, 4), xlim=(-14, 14), posterior_file=MODEL_FIT / 'exp2_posterior_uniform.nc')
		plots.draw_letter_grid(fig[2,0], letter_width=36, n_letters=7)


figure_functions = {
	'2': figure2,
	'4': figure4,
	'5': figure5,
	'6': figure6,
	'7': figure7,
	'8': figure8,
	'9': figure9,
	'10': figure10,
}

if __name__ == '__main__':

	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument('fig_num', action='store', type=str, help='Figure number (or comma-separated numbers)')
	args = parser.parse_args()

	if args.fig_num == 'test':
		print('👍')
		exit()

	if args.fig_num == 'all':
		args.fig_num = ','.join(figure_functions.keys())

	for fig_num in args.fig_num.split(','):
		if fig_num not in figure_functions:
			print(f'Invalid figure number: {fig_num}')
		figure_functions[fig_num]()
