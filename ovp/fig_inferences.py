import core
import model


def generate_plot_of_word_inferences(file_path, lexicon, alpha, beta, gamma):
	reader = model.Reader(lexicon, alpha, beta, gamma)
	with core.Figure(file_path, 40, 5, width='double', height=5) as fig:
		for t in range(8):
			for j in range(5):
				posterior_over_words = reader.p_word_given_target(t, j, 100)
				fig[t,j].bar(range(8), posterior_over_words, color='black')
				fig[t,j].set_ylim(0,1)
				fig[t,j].set_xticks(range(8))
				fig[t,j].set_xticklabels(range(1,9))
				if t == 7:
					fig[t,j].set_xlabel(f'Fixation position {j+1}')
				if j == 0:
					fig[t,j].set_ylabel(f'Word {t+1}')


lexicon_l = list(map(tuple, core.json_read(core.DATA / 'experiments' / 'online' / 'exp1_left.json')['words']))
lexicon_r = list(map(tuple, core.json_read(core.DATA / 'experiments' / 'online' / 'exp1_right.json')['words']))

generate_plot_of_word_inferences(core.VISUALS / 'word_inferences1.pdf', lexicon_r, alpha=0.7, beta=0.4, gamma=0)
generate_plot_of_word_inferences(core.VISUALS / 'word_inferences2.pdf', lexicon_r, alpha=0.7, beta=0.4, gamma=0.9)