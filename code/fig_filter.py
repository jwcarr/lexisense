import core
import model


WORD_LENGTH = 5
N_SYMBOLS = 10


def generate_filter_figure(file_path, lexicon, alpha_vals, beta_gamma_vals):
	alphabet_size = len(set(sum(lexicon, tuple())))
	n_subplots = len(alpha_vals)*len(beta_gamma_vals)
	n_cols = len(beta_gamma_vals)
	with core.Figure(file_path, n_subplots, n_cols, width='double') as fig:
		for i, alpha in enumerate(alpha_vals):
			for j, (beta, gamma) in enumerate(beta_gamma_vals):
				fig[i,j].plot((-1, WORD_LENGTH), (1/alphabet_size, 1/alphabet_size), linestyle=':', linewidth=1, color='black')
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
						fig[i,j].set_title(f'$γ$ = {gamma} (better perception to the left)', fontsize=7)
					elif gamma > 0:
						fig[i,j].set_title(f'$γ$ = {gamma} (better perception to the right)', fontsize=7)
					else:
						fig[i,j].set_title(f'$γ$ = {gamma} (symmetrical perceptual span)', fontsize=7)


lexicon = [(i,) * WORD_LENGTH for i in range(N_SYMBOLS)]
alpha_vals = [0.9, 0.7, 0.5]
beta_gamma_vals = [
	(0.2, -0.3), (0.4, -0.3), (0.8, -0.3),
	(0.2,  0.0), (0.4,  0.0), (0.8,  0.0),
	(0.2,  0.3), (0.4,  0.3), (0.8,  0.3)
]

generate_filter_figure(core.VISUALS / f'filter.pdf', lexicon, alpha_vals, beta_gamma_vals)
