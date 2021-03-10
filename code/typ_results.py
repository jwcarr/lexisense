import core
import fig_uncertainty

def plot_languages(figure_file, uncertainty_data, lengths, languages):
	figure = Figure(len(lengths), 3, (7.09, 2))
	for axis, length in zip(figure, lengths):
		for language in languages:
			uncertainty_by_position = core.pickle_read(uncertainty_data + f'{language}_{length}.pkl')
			diff = deficit(uncertainty_by_position)
			print(language, diff)
			fig_uncertainty.plot_uncertainty(axis, uncertainty_by_position, label=language_names[language], color=language_colors[language])
	# figure.axes[0,0].legend()
	figure.save(figure_file)


language_names = {
	'de': 'German',
	'en': 'English',
	'es': 'Spanish',
	'gr': 'Greek',
	'it': 'Italian',
	'nl': 'Dutch',
	'pl': 'Polish',
	'sw': 'Swahili',
}

language_colors = {
	'de': 'gray',
	'en': 'gray',
	'es': 'gray',
	'gr': 'gray',
	'it': 'gray',
	'nl': 'gray',
	'pl': 'gray',
	'sw': 'gray',
}


plot_languages(core.FIGS / 'typ_uncertainty1.eps', core.DATA / 'typ_uncertainty' / 'gamma1/', [5, 7, 9], ['de', 'en', 'es', 'gr', 'it', 'nl', 'pl', 'sw'])
# plot_languages(core.FIGS / 'typ_uncertainty2.eps', core.DATA / 'typ_uncertainty' / 'gamma2/', [5, 7, 9], ['de', 'en', 'es', 'gr', 'it', 'nl', 'pl', 'sw'])
