import core
import fig_uncertainty

def plot_languages(figure_file, uncertainty_data, lengths, languages):
	figure = core.Figure(3, 3, width='double')
	for length, axis in zip(lengths, figure):
		for language in languages:
			try:
				uncertainty_by_position = core.pickle_read(uncertainty_data / f'{language}_{length}.pkl')
			except:
				continue
			fig_uncertainty.plot_uncertainty(axis, uncertainty_by_position, label=core.language_names[language], color=core.language_colors[language])
	# figure.axes[0,0].legend()
	figure.save(figure_file)


plot_languages(core.VISUALS / 'typ_uncertainty0.0.pdf', core.DATA / 'typ_uncertainty' / 'gamma0.0', [5, 7, 9], ['de', 'en', 'es', 'gr', 'it', 'nl', 'pl', 'sw'])
# plot_languages(core.VISUALS / 'typ_uncertainty0.2.pdf', core.DATA / 'typ_uncertainty' / 'gamma0.2', [5, 7, 9], ['de', 'en', 'es', 'gr', 'it', 'nl', 'pl', 'sw'])
