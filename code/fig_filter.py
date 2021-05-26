import core
import model

def generate_filter_figure(file_path, lexicon, alpha_vals, beta_delta_vals, gamma):
	alphabet_size = len(set(sum(lexicon, tuple())))
	n_subplots = len(alpha_vals)*len(beta_delta_vals)
	n_cols = len(beta_delta_vals)
	with core.Figure(file_path, n_subplots, n_cols, width='double') as fig:
		for i, alpha in enumerate(alpha_vals):
			for j, (beta, delta) in enumerate(beta_delta_vals):
				fig[i,j].plot((-1, 5), (1/alphabet_size, 1/alphabet_size), linestyle=':', linewidth=1, color='black')
				reader = model.Reader(lexicon, alpha, beta, gamma, delta)
				for dist in reader.phi:
					fig[i,j].plot(dist)
				fig[i,j].set_ylim(0,1)
				fig[i,j].set_xlim(-0.25,4.25)
				fig[i,j].set_xticks(range(5))
				if i < len(alpha_vals) - 1:
					fig[i,j].set_xticklabels([])
				else:
					fig[i,j].set_xticklabels(range(1,6))
				if i == len(alpha_vals)-1:
					fig[i,j].set_xlabel(f'$β$ = {beta}')
				if j == 0:
					fig[i,j].set_ylabel(f'$α$ = {alpha}')
				if i == 0 and (j-1) % 3 == 0:
					if delta < 0:
						fig[i,j].set_title(f'$δ$ = {delta} (better perception to the left)', fontsize=7)
					elif delta > 0:
						fig[i,j].set_title(f'$δ$ = {delta} (better perception to the right)', fontsize=7)
					else:
						fig[i,j].set_title(f'$δ$ = {delta} (symmetrical perceptual span)', fontsize=7)


lexicon = [(i,) * 5 for i in range(10)]
alpha_vals = [0.9, 0.7, 0.5]
beta_delta_vals = [
	(0.2, -0.3), (0.4, -0.3), (0.8, -0.3),
	(0.2,  0.0), (0.4,  0.0), (0.8,  0.0),
	(0.2,  0.3), (0.4,  0.3), (0.8,  0.3)
]

generate_filter_figure(core.VISUALS / f'filter.pdf', lexicon, alpha_vals, beta_delta_vals, gamma=2)
