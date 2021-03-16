import numpy as np
import core

def plot_guidelines(axis, fixation_positions, mean_uncertainty, color):
	middle = len(fixation_positions) // 2 + 1
	first_x = fixation_positions[:middle]
	first_y = mean_uncertainty[:middle]
	last_x = fixation_positions[middle-1:][::-1]
	last_y = mean_uncertainty[middle-1:][::-1]
	for i in range(len(last_x)-(len(fixation_positions)%2)):
		axis.plot([first_x[i], last_x[i]], [first_y[i], last_y[i]], c=color, linestyle=':', linewidth=.5)

def plot_min_uncertainty(axis, uncertainty_by_position, color):
	min_uncertainty = min(uncertainty_by_position)
	argmin_uncertainty = np.argmin(uncertainty_by_position) + 1
	x, = axis.plot([argmin_uncertainty, argmin_uncertainty], [0, min_uncertainty], color=color, linewidth=.5)
	if color == 'black':
		x.set_dashes([0, 2, 1, 1])
	else:
		x.set_dashes([1, 3])

def plot_mean_diff(axis, fixation_positions, mean_uncertainty, color):
	middle = len(fixation_positions) // 2 + 1
	left_uncertainties = mean_uncertainty[:middle-1]
	right_uncertainties = list(reversed(mean_uncertainty[middle-1:]))[:len(left_uncertainties)]
	uncertainty_reduction = np.mean(right_uncertainties) - np.mean(left_uncertainties)
	if color == 'black':
		axis.text(1, 1.5, str(round(uncertainty_reduction, 2)), color=color, ha='left', fontsize=6)
	else:
		axis.text(len(fixation_positions), 1.5, str(round(uncertainty_reduction, 2)), color=color, ha='right', fontsize=6)

def plot_uncertainty(axis, uncertainty_by_position, label=None, color=None, ylim=(1, 7), show_guidelines=True, show_min=True, show_mean_diff=True):
	length = len(uncertainty_by_position)
	positions = list(range(1, length+1))
	if show_guidelines:
		plot_guidelines(axis, positions, uncertainty_by_position, color)
	if show_min:
		plot_min_uncertainty(axis, uncertainty_by_position, color)
	if show_mean_diff:
		plot_mean_diff(axis, positions, uncertainty_by_position, color)
	axis.plot(positions, uncertainty_by_position, label=label, linewidth=1, color=color)
	xpad = (length - 1) * 0.05
	axis.set_xlim(1-xpad, length+xpad)
	axis.set_ylim(*ylim)
	axis.set_xticks(range(1, length+1))
	axis.set_xticklabels(range(1, length+1))

def plot_languages(figure_file, uncertainty_data, languages, lengths):
	figure = core.Figure(len(languages)*len(lengths), len(lengths), width='double', height=6.5)
	for i, language in enumerate(languages):
		for j, length in enumerate(lengths):
			axis = figure[i,j]

			uncertainty_by_position = core.pickle_read(uncertainty_data / 'gamma0.3' / f'{language}_{length}.pkl')
			plot_uncertainty(axis, uncertainty_by_position, color='red')

			uncertainty_by_position = core.pickle_read(uncertainty_data / 'gamma0.0' / f'{language}_{length}.pkl')
			plot_uncertainty(axis, uncertainty_by_position, color='black')

			axis.set_xlabel(f'{length}-letter words')
			axis.set_ylabel(core.language_names[language])
	figure.save(figure_file)


if __name__ == '__main__':

	plot_languages(
		core.FIGS / 'typ_uncertainty.eps',
		core.DATA / 'typ_uncertainty',
		['nl', 'en', 'de', 'gr', 'it', 'pl', 'es', 'sw'],
		[5 ,6, 7, 8, 9]
	)
