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

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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
	file_path = FIGS / 'fig2.pdf'
	# fig, axes = plt.subplots(1, 3, figsize=(7, 2))

	# uncertainty = json_read(DATA / 'lang_uncertainty' / 'gamma0.0' / 'en.json')
	# plots.plot_uncertainty(axes[0], uncertainty['7'], color='black', show_min=True, label='γ = 0')
	# uncertainty = json_read(DATA / 'lang_uncertainty' / 'gamma0.5' / 'en.json')
	# plots.plot_uncertainty(axes[0], uncertainty['7'], color='MediumSeaGreen', show_min=True, label='γ = 0.5')
	# axes[0].legend(frameon=False)
	# axes[0].set_ylim(0, 4)
	# axes[0].set_title('English')

	# uncertainty = json_read(DATA / 'lang_uncertainty' / 'gamma0.0' / 'pl.json')
	# plots.plot_uncertainty(axes[1], uncertainty['7'], color='black', show_min=True)
	# uncertainty = json_read(DATA / 'lang_uncertainty' / 'gamma0.5' / 'pl.json')
	# plots.plot_uncertainty(axes[1], uncertainty['7'], color='MediumSeaGreen', show_min=True)
	# axes[1].set_ylim(0, 4)
	# axes[1].set_title('Polish')
	# axes[1].set_yticklabels([])
	# axes[1].set_ylabel(None)

	# uncertainty = json_read(DATA / 'lang_uncertainty' / 'gamma0.0' / 'he.json')
	# plots.plot_uncertainty(axes[2], uncertainty['7'], color='black', show_min=True)
	# uncertainty = json_read(DATA / 'lang_uncertainty' / 'gamma0.5' / 'he.json')
	# plots.plot_uncertainty(axes[2], uncertainty['7'], color='MediumSeaGreen', show_min=True)
	# axes[2].set_ylim(0, 4)
	# axes[2].set_title('Hebrew')
	# axes[2].set_yticklabels([])
	# axes[2].set_ylabel(None)

	# fig.tight_layout()
	# fig.savefig(file_path)
	# quit()

	
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
	with Figure(file_path, n_rows=9, n_cols=5, width='double', height=210) as fig:
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
	from code import model_predict

	sim_datasets = model_predict.simulate_from_posterior(exp1, n_sims=100)
	uncertainty_left, uncertainty_right = model_predict.uncertainty_curve_from_posterior(exp1, n_sims=10000)

	file_path = FIGS / 'fig5a.svg'

	fig = plt.figure(figsize=(7, 4))
	gs = gridspec.GridSpec(3, 4, figure=fig, height_ratios=[1, 1, 1.3])

	gsA = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0:2, 0:2])

	ax = fig.add_subplot(gsA[0, 0])
	plots.plot_test_curve(ax, exp1.left, show_individuals=True, add_jitter=False)
	plots.draw_brace(ax, (2,4), 0.2, 'High\ninfo')
	plots.draw_brace(ax, (6,7), 0.2, 'Low\ninfo')
	ax.set_ylabel('Probability correct')

	ax = fig.add_subplot(gsA[0, 1])
	plots.plot_test_curve(ax, exp1.right, show_individuals=True, add_jitter=False)
	plots.draw_brace(ax, (4,6), 0.2, 'High\ninfo')
	plots.draw_brace(ax, (1,2), 0.2, 'Low\ninfo')
	ax.set_ylabel(None)
	ax.set_yticklabels([])

	gsB = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[0:2, 2:4])

	ax = fig.add_subplot(gsB[0, 0])
	plots.plot_prior(ax, exp1, 'α', transform_to_param_bounds=True)
	plots.plot_posterior(ax, exp1, 'α')
	# ax.set_ylabel('Density')

	ax = fig.add_subplot(gsB[0, 1])
	plots.plot_prior(ax, exp1, 'β', transform_to_param_bounds=True)
	plots.plot_posterior(ax, exp1, 'β')

	ax = fig.add_subplot(gsB[1, 0])
	plots.plot_prior(ax, exp1, 'γ', transform_to_param_bounds=True)
	plots.plot_posterior(ax, exp1, 'γ')
	# ax.set_ylabel('Density')

	ax = fig.add_subplot(gsB[1, 1])
	plots.plot_prior(ax, exp1, 'ε', transform_to_param_bounds=True)
	plots.plot_posterior(ax, exp1, 'ε')

	gsC = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[2, 0:2])

	ax = fig.add_subplot(gsC[0, 0])
	plots.plot_posterior_predictive(ax, sim_datasets, exp1.left, lexicon_index=0, show_legend=False, show_veridical=True)
	ax.set_ylabel('Probability correct')

	ax = fig.add_subplot(gsC[0, 1])
	plots.plot_posterior_predictive(ax, sim_datasets, exp1.right, lexicon_index=1, show_veridical=True)
	ax.set_ylabel(None)
	ax.set_yticklabels([])

	gsD = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[2, 2:4])

	ax = fig.add_subplot(gsD[0, 0])
	plots.plot_uncertainty(ax, uncertainty_left, color=exp1.left.color)
	plots.plot_uncertainty(ax, uncertainty_right, color=exp1.right.color)
	ax.set_ylim(0, 1.3)

	fig.tight_layout(pad=0.1, w_pad=0.1, h_pad=1.0)

	fig.savefig(file_path)


def figure6():
	plots.landing_position_image(exp2.left, FIGS / 'fig6_landing_left.svg')
	plots.landing_position_image(exp2.right, FIGS / 'fig6_landing_right.svg')

	file_path = FIGS / 'fig6.svg'

	fig = plt.figure(layout='tight', figsize=(7, 4))
	gs = gridspec.GridSpec(2, 4, figure=fig)

	# gsA = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, 0:2])

	ax = fig.add_subplot(gs[0, 0:2])
	ax.axis('off')

	# gsB = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, 2:4])

	ax = fig.add_subplot(gs[0, 2])
	plots.plot_landing_curve(ax, exp2.left, show_individuals=True, show_average=True)

	ax = fig.add_subplot(gs[0, 3])
	plots.plot_landing_curve(ax, exp2.right, show_individuals=True, show_average=True)

	# gsC = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[1, 0:4])

	ax = fig.add_subplot(gs[1, 0])
	plots.plot_prior(ax, exp2, 'τ')
	plots.plot_posterior(ax, exp2, 'τ')
	plots.draw_letter_grid(ax, letter_width=36, n_letters=7)

	ax = fig.add_subplot(gs[1, 1])
	plots.plot_posterior_difference(ax, exp2, 'τ', hdi=0.95, rope=(-9, 9), xlim=(-9, 49))

	ax = fig.add_subplot(gs[1, 2])
	plots.plot_prior(ax, exp2, 'δ')
	plots.plot_posterior(ax, exp2, 'δ')

	ax = fig.add_subplot(gs[1, 3])
	plots.plot_posterior_difference(ax, exp2, 'δ', hdi=0.95, rope=(-4, 4), xlim=(-12, 12))

	fig.savefig(file_path)


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
