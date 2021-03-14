import core
import fig_uncertainty

def plot_languages(figure_file, uncertainty_data, lengths, languages):
	figure = core.Figure(5, 5, width='double')
	for length, axis in zip(lengths, figure):
		for language in languages:
			try:
				uncertainty_by_position = core.pickle_read(uncertainty_data / f'{language}_{length}.pkl')
			except:
				continue
			fig_uncertainty.plot_uncertainty(axis, uncertainty_by_position, label=core.language_names[language], color=core.language_colors[language], ylim=(1,7))
	# figure.axes[0,0].legend()
	figure.save(figure_file)


plot_languages(core.VISUALS / 'typ_uncertainty0.0.pdf', core.DATA / 'typ_uncertainty' / 'gamma0.0', [5, 6, 7, 8, 9], ['de', 'en', 'es', 'gr', 'it', 'nl', 'pl', 'sw'])
plot_languages(core.VISUALS / 'typ_uncertainty0.3.pdf', core.DATA / 'typ_uncertainty' / 'gamma0.3', [5, 6, 7, 8, 9], ['de', 'en', 'es', 'gr', 'it', 'nl', 'pl', 'sw'])
