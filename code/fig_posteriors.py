from fig import Figure
import core
import model


def generate_plot_of_posteriors(file_path, lexicon, alpha, beta):
	fig = Figure(40, 5, figsize=(7.09, 5))
	reader = model.Reader(lexicon, alpha, beta)
	for t in range(8):
		for j in range(5):
			posterior_over_words = reader.estimate_posterior(t, j)
			fig[t,j].bar(range(8), posterior_over_words, color='black')
			fig[t,j].set_ylim(0,1)
			fig[t,j].set_xticks(range(8))
			fig[t,j].set_xticklabels(range(1,9))
			if t == 7:
				fig[t,j].set_xlabel(f'Fixation position {j+1}')
			if j == 0:
				fig[t,j].set_ylabel(f'Word {t+1}')
	fig.save(file_path)


generate_plot_of_posteriors('../manuscript/figs/word_inferences1.eps', core.lexicon_l, alpha=0.8, beta=0.5)
generate_plot_of_posteriors('../manuscript/figs/word_inferences2.eps', core.lexicon_l, alpha=0.8, beta=2)
