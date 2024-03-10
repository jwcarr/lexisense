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


# EXPERIMENT 3
exp3 = Experiment('exp3')
exp3.set_exclusion_threshold(7, 8)
exp3.set_params({
    'τ': (0, 252),
    'δ': (0, 60),
    'ζ': (0, 60),
    'ξ': (0, 60),
})
exp3.set_priors({
    'δ': ('gamma', (23., 10.)),
    'ζ': ('exponential', (0.1,)),
    'ξ': ('exponential', (0.1,)),
})
exp3.left.set_priors({
    'τ': ('normal', (94., 30.)),
})
exp3.right.set_priors({
    'τ': ('normal', (128., 30.)),
})


def figure1():
	from code import model

	file_path = FIGS / 'fig1.svg'

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
					fig[i,j].plot(dist, linewidth=1)
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


def figure2():
	file_path = FIGS / 'fig2.eps'

	with Figure(file_path, n_cols=2, n_rows=2, width='single', height=80) as fig:

		uncertainty = json_read(DATA / 'lang_uncertainty' / 'gamma0.0' / 'en.json')
		plots.plot_uncertainty(fig[0,0], uncertainty['7'], color='black', show_min=True, label='$γ$ = 0')
		uncertainty = json_read(DATA / 'lang_uncertainty' / 'gamma0.5' / 'en.json')
		plots.plot_uncertainty(fig[0,0], uncertainty['7'], color='MediumSeaGreen', show_min=True, label='$γ$ = 0.5')
		fig[0,0].legend(frameon=False)
		fig[0,0].set_ylim(0, 4)
		fig[0,0].set_title('English', fontsize=7)

		uncertainty = json_read(DATA / 'lang_uncertainty' / 'gamma0.0' / 'sw.json')
		plots.plot_uncertainty(fig[0,1], uncertainty['7'], color='black', show_min=True, label='$γ$ = 0')
		uncertainty = json_read(DATA / 'lang_uncertainty' / 'gamma0.5' / 'sw.json')
		plots.plot_uncertainty(fig[0,1], uncertainty['7'], color='MediumSeaGreen', show_min=True, label='$γ$ = 0.5')
		fig[0,1].set_ylim(0, 4)
		fig[0,1].set_title('Swahili', fontsize=7)

		uncertainty = json_read(DATA / 'lang_uncertainty' / 'gamma0.0' / 'pl.json')
		plots.plot_uncertainty(fig[1,0], uncertainty['7'], color='black', show_min=True, label='$γ$ = 0')
		uncertainty = json_read(DATA / 'lang_uncertainty' / 'gamma0.5' / 'pl.json')
		plots.plot_uncertainty(fig[1,0], uncertainty['7'], color='MediumSeaGreen', show_min=True, label='$γ$ = 0.5')
		fig[1,0].set_ylim(0, 4)
		fig[1,0].set_title('Polish', fontsize=7)

		uncertainty = json_read(DATA / 'lang_uncertainty' / 'gamma0.0' / 'he.json')
		plots.plot_uncertainty(fig[1,1], uncertainty['7'], color='black', show_min=True, label='$γ$ = 0')
		uncertainty = json_read(DATA / 'lang_uncertainty' / 'gamma0.5' / 'he.json')
		plots.plot_uncertainty(fig[1,1], uncertainty['7'], color='MediumSeaGreen', show_min=True, label='$γ$ = 0.5')
		fig[1,1].set_ylim(0, 4)
		fig[1,1].set_title('Hebrew', fontsize=7)


def figure5():
	file_path = FIGS / 'fig05.eps'

	with Figure(file_path, n_cols=2, n_rows=1, width='single', height=60) as fig:

		plots.plot_test_curve(fig[0,0], exp1.left, show_individuals=True, add_jitter=False)
		plots.draw_brace(fig[0,0], (2,4), 0.2, 'High\ninfo')
		plots.draw_brace(fig[0,0], (6,7), 0.2, 'Low\ninfo')

		plots.plot_test_curve(fig[0,1], exp1.right, show_individuals=True, add_jitter=False)
		plots.draw_brace(fig[0,1], (4,6), 0.2, 'High\ninfo')
		plots.draw_brace(fig[0,1], (1,2), 0.2, 'Low\ninfo')


def figure6():
	file_path = FIGS / 'fig06.eps'

	with Figure(file_path, n_cols=2, n_rows=2, width='single', height=80) as fig:

		plots.plot_prior(fig[0,0], exp1, 'α', transform_to_param_bounds=True)
		plots.plot_posterior(fig[0,0], exp1, 'α')

		plots.plot_prior(fig[0,1], exp1, 'β', transform_to_param_bounds=True)
		plots.plot_posterior(fig[0,1], exp1, 'β')

		plots.plot_prior(fig[1,0], exp1, 'γ', transform_to_param_bounds=True)
		plots.plot_posterior(fig[1,0], exp1, 'γ')

		plots.plot_prior(fig[1,1], exp1, 'ε', transform_to_param_bounds=True)
		plots.plot_posterior(fig[1,1], exp1, 'ε')

		fig.auto_deduplicate_axes = False


def figure7():
	file_path = FIGS / 'fig07.eps'

	from code import model_predict
	sim_datasets = model_predict.simulate_from_posterior(exp1, n_sims=100)

	with Figure(file_path, n_cols=2, n_rows=1, width='single', height=40) as fig:

		plots.plot_posterior_predictive(fig[0,0], sim_datasets, exp1.left, lexicon_index=0, show_legend=False, show_veridical=True)

		plots.plot_posterior_predictive(fig[0,1], sim_datasets, exp1.right, lexicon_index=1, show_veridical=True)

def figure8():
	file_path = FIGS / 'fig08.eps'

	from code import model_predict
	uncertainty_left, uncertainty_right = model_predict.uncertainty_curve_from_posterior(exp1, n_sims=10000)

	with Figure(file_path, n_cols=2, n_rows=1, width='single', height=40) as fig:

		plots.plot_uncertainty(fig[0,0], uncertainty_left, color=exp1.left.color)
		fig[0,0].set_ylim(0, 1.4)
		fig[0,0].set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4])
		
		plots.plot_uncertainty(fig[0,1], uncertainty_right, color=exp1.right.color)
		fig[0,1].set_ylim(0, 1.4)
		fig[0,1].set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4])


def figure10():
	file_path = FIGS / 'fig10.svg'

	plots.landing_position_image(exp2, file_path)


def figure11():
	file_path = FIGS / 'fig11.eps'

	with Figure(file_path, n_cols=2, n_rows=1, width='single', height=40) as fig:

		plots.plot_landing_curve(fig[0,0], exp2.left, show_individuals=True, show_average=True)
		plots.plot_landing_curve(fig[0,1], exp2.right, show_individuals=True, show_average=True)


def figure12():
	file_path = FIGS / 'fig12.eps'
	
	with Figure(file_path, n_cols=2, n_rows=2, width='single', height=80) as fig:

		plots.plot_prior(fig[0,0], exp2, 'τ')
		plots.plot_posterior(fig[0,0], exp2, 'τ')
		plots.draw_letter_grid(fig[0,0], letter_width=36, n_letters=7)
		
		plots.plot_prior(fig[1,0], exp2, 'δ')
		plots.plot_posterior(fig[1,0], exp2, 'δ')

		plots.plot_posterior_difference(fig[0,1], exp2, 'τ', hdi=0.95, rope=(-9, 9), xlim=(-9, 49))

		plots.plot_posterior_difference(fig[1,1], exp2, 'δ', hdi=0.95, rope=(-4, 4), xlim=(-12, 12))


def figure13():
	file_path = FIGS / 'fig13.svg'

	plots.landing_position_image(exp3, file_path)


def figure14():
	file_path = FIGS / 'fig14.eps'

	with Figure(file_path, n_cols=2, n_rows=1, width='single', height=40) as fig:

		plots.plot_landing_curve(fig[0,0], exp3.left, show_individuals=True, show_average=True)
		plots.plot_landing_curve(fig[0,1], exp3.right, show_individuals=True, show_average=True)


def figure15():
	file_path = FIGS / 'fig15.eps'
	
	with Figure(file_path, n_cols=2, n_rows=1, width='single', height=40) as fig:

		plots.plot_prior(fig[0,0], exp3, 'τ')
		plots.plot_posterior(fig[0,0], exp3, 'τ')
		plots.draw_letter_grid(fig[0,0], letter_width=36, n_letters=7)

		plots.plot_posterior_difference(fig[0,1], exp3, 'τ', hdi=0.95, rope=(-10, 10), xlim=(-9, 49))




# SUPPLEMENTARY MATERIAL

def figureS1():
	file_path = FIGS / 'supp1.eps'

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

	with Figure(file_path, n_rows=9, n_cols=5, width=6, height=8, units='inches') as fig:

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


def figureS2():
	file_path = FIGS / 'supp2.eps'

	with Figure(file_path, n_cols=2, n_rows=2, width=6, height=3.15, units='inches') as fig:

		fig.auto_deduplicate_axes = False

		for param, axis in zip(['α', 'β', 'γ', 'ε'], fig):
			plots.plot_prior(axis, exp1, param, label='Prior', transform_to_param_bounds=True)
			plots.plot_posterior(axis, exp1, param, label='Posterior')
			plots.plot_posterior(axis, exp1, param, label='Posterior using uniform prior', posterior_file=MODEL_FIT / 'exp1_posterior_uniform.nc', linestyle=':')
			plots.plot_posterior(axis, exp1.left, param, label='Posterior using left-heavy dataset only', posterior_file=MODEL_FIT / 'exp1_posterior_left.nc')
			plots.plot_posterior(axis, exp1.right, param, label='Posterior using right-heavy dataset only', posterior_file=MODEL_FIT / 'exp1_posterior_right.nc')
			fig[0,0].legend(frameon=False)


def figureS3():
	file_path = FIGS / 'supp3.eps'

	exp2.set_priors({
		'τ': ('uniform', (0., 252.)),
		'δ': ('uniform', (0., 60.)),
	})

	with Figure(file_path, n_cols=2, n_rows=2, width='single', height=80) as fig:

		plots.plot_prior(fig[0,0], exp2, 'τ')
		plots.plot_posterior(fig[0,0], exp2, 'τ', posterior_file=MODEL_FIT / 'exp2_posterior_uniform.nc')
		plots.draw_letter_grid(fig[0,0], letter_width=36, n_letters=7)

		plots.plot_prior(fig[1,0], exp2, 'δ')
		plots.plot_posterior(fig[1,0], exp2, 'δ', posterior_file=MODEL_FIT / 'exp2_posterior_uniform.nc')

		plots.plot_posterior_difference(fig[0,1], exp2, 'τ', hdi=0.95, rope=(-9, 9), xlim=(-9, 49), posterior_file=MODEL_FIT / 'exp2_posterior_uniform.nc')
		
		plots.plot_posterior_difference(fig[1,1], exp2, 'δ', hdi=0.95, rope=(-4, 4), xlim=(-12, 12), posterior_file=MODEL_FIT / 'exp2_posterior_uniform.nc')
		

def figureS4():
	file_path = FIGS / 'supp4.eps'

	with Figure(file_path, n_cols=2, n_rows=4, width='single', height=160) as fig:

		plots.plot_prior(fig[0,0], exp2, 'τ')
		plots.plot_posterior(fig[0,0], exp2, 'τ', posterior_file=MODEL_FIT / 'exp2_posterior_indy_ζξ.nc')
		plots.draw_letter_grid(fig[0,0], letter_width=36, n_letters=7)
		plots.plot_posterior_difference(fig[0,1], exp2, 'τ', hdi=0.95, rope=(-9, 9), xlim=(-9, 49), posterior_file=MODEL_FIT / 'exp2_posterior_indy_ζξ.nc')

		plots.plot_prior(fig[1,0], exp2, 'δ')
		plots.plot_posterior(fig[1,0], exp2, 'δ', posterior_file=MODEL_FIT / 'exp2_posterior_indy_ζξ.nc')		
		plots.plot_posterior_difference(fig[1,1], exp2, 'δ', hdi=0.95, rope=(-4, 4), xlim=(-12, 12), posterior_file=MODEL_FIT / 'exp2_posterior_indy_ζξ.nc')
		
		plots.plot_prior(fig[2,0], exp2, 'ζ')
		plots.plot_posterior(fig[2,0], exp2, 'ζ', posterior_file=MODEL_FIT / 'exp2_posterior_indy_ζξ.nc')		
		plots.plot_posterior_difference(fig[2,1], exp2, 'ζ', hdi=0.95, xlim=(-19, 19), posterior_file=MODEL_FIT / 'exp2_posterior_indy_ζξ.nc')
		
		plots.plot_prior(fig[3,0], exp2, 'ξ')
		plots.plot_posterior(fig[3,0], exp2, 'ξ', posterior_file=MODEL_FIT / 'exp2_posterior_indy_ζξ.nc')		
		plots.plot_posterior_difference(fig[3,1], exp2, 'ξ', hdi=0.95, xlim=(-12, 12), posterior_file=MODEL_FIT / 'exp2_posterior_indy_ζξ.nc')
		

def figureS5():
	file_path = FIGS / 'supp5.eps'

	exp3.set_priors({
		'τ': ('uniform', (0., 252.)),
		'δ': ('uniform', (0., 60.)),
	})

	with Figure(file_path, n_cols=2, n_rows=2, width='single', height=80) as fig:

		plots.plot_prior(fig[0,0], exp3, 'τ')
		plots.plot_posterior(fig[0,0], exp3, 'τ', posterior_file=MODEL_FIT / 'exp3_posterior_uniform.nc')
		plots.draw_letter_grid(fig[0,0], letter_width=36, n_letters=7)

		plots.plot_prior(fig[1,0], exp3, 'δ')
		plots.plot_posterior(fig[1,0], exp3, 'δ', posterior_file=MODEL_FIT / 'exp3_posterior_uniform.nc')

		plots.plot_posterior_difference(fig[0,1], exp3, 'τ', hdi=0.95, rope=(-10, 10), xlim=(-9, 49), posterior_file=MODEL_FIT / 'exp3_posterior_uniform.nc')
		

def figureS6():
	file_path = FIGS / 'supp6.eps'

	del exp3.priors['δ']
	exp3.left.set_priors({
		'δ': ('gamma', (22., 10.)),
	})
	exp3.right.set_priors({
		'δ': ('gamma', (24., 10.)),
	})

	with Figure(file_path, n_cols=2, n_rows=4, width='single', height=160) as fig:

		plots.plot_prior(fig[0,0], exp3, 'τ')
		plots.plot_posterior(fig[0,0], exp3, 'τ', posterior_file=MODEL_FIT / 'exp3_posterior_full.nc')
		plots.draw_letter_grid(fig[0,0], letter_width=36, n_letters=7)
		plots.plot_posterior_difference(fig[0,1], exp3, 'τ', hdi=0.95, rope=(-10, 10), xlim=(-9, 49), posterior_file=MODEL_FIT / 'exp3_posterior_full.nc')

		plots.plot_prior(fig[1,0], exp3, 'δ')
		plots.plot_posterior(fig[1,0], exp3, 'δ', posterior_file=MODEL_FIT / 'exp3_posterior_full.nc')		
		plots.plot_posterior_difference(fig[1,1], exp3, 'δ', hdi=0.95, xlim=(-12, 12), posterior_file=MODEL_FIT / 'exp3_posterior_full.nc')
		
		plots.plot_prior(fig[2,0], exp3, 'ζ')
		plots.plot_posterior(fig[2,0], exp3, 'ζ', posterior_file=MODEL_FIT / 'exp3_posterior_full.nc')		
		plots.plot_posterior_difference(fig[2,1], exp3, 'ζ', hdi=0.95, xlim=(-19, 19), posterior_file=MODEL_FIT / 'exp3_posterior_full.nc')
		
		plots.plot_prior(fig[3,0], exp3, 'ξ')
		plots.plot_posterior(fig[3,0], exp3, 'ξ', posterior_file=MODEL_FIT / 'exp3_posterior_full.nc')		
		plots.plot_posterior_difference(fig[3,1], exp3, 'ξ', hdi=0.95, xlim=(-12, 12), posterior_file=MODEL_FIT / 'exp3_posterior_full.nc')
		

figure_functions = {
	'1': figure1,
	'2': figure2,
	'5': figure5,
	'6': figure6,
	'7': figure7,
	'8': figure8,
	'10': figure10,
	'11': figure11,
	'12': figure12,
	'13': figure13,
	'14': figure14,
	'15': figure15,

	'S1': figureS1,
	'S2': figureS2,
	'S3': figureS3,
	'S4': figureS4,
	'S5': figureS5,
	'S6': figureS6,
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
