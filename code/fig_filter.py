from fig import Figure
import model

def generate_filter_figure(file_path, lexicon, alpha_vals, beta_vals, gamma):
	alphabet_size = len(set(sum(lexicon, tuple())))
	fig = Figure(len(alpha_vals)*len(beta_vals), len(beta_vals), figsize=(7.09, 3))
	for i, alpha in enumerate(alpha_vals):
		for j, beta in enumerate(beta_vals):
			fig[i,j].plot((-1, 5), (1/alphabet_size, 1/alphabet_size), linestyle=':', linewidth=1, color='black')
			reader = model.Reader(lexicon, alpha, beta, gamma)
			for dist in reader.phi:
				fig[i,j].plot(dist)
			fig[i,j].set_ylim(0,1)
			fig[i,j].set_xlim(-0.25,4.25)
			fig[i,j].set_xticks(range(5))
			if i < len(alpha_vals) - 1:
				fig[i,j].set_xticklabels([])
			else:
				fig[i,j].set_xticklabels(range(1,6))
			if j > 0:
				fig[i,j].set_yticklabels([])
			if i == len(alpha_vals)-1:
				fig[i,j].set_xlabel(f'β = {beta}')
			if j == 0:
				fig[i,j].set_ylabel(f'α = {alpha}')
	fig.save(file_path)


lexicon = [(i,) * 5 for i in range(10)]
alpha_vals = [0.999, 0.8, 0.6, 0.4, 0.2]
beta_vals = [0.125, 0.25, 0.5, 1, 2]

generate_filter_figure('../manuscript/figs/filter.eps', lexicon, alpha_vals, beta_vals, 1)
generate_filter_figure('../manuscript/figs/filter_asy.eps', lexicon, alpha_vals, beta_vals, 2)
