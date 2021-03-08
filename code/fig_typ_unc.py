import matplotlib.pyplot as plt
import core

def deficit(uncertainty_by_position):
	length = len(uncertainty_by_position)
	half_length = length // 2
	diffs = []
	for i, j in zip(range(half_length), range(length-1, half_length, -1)):
		diffs.append(uncertainty_by_position[j] - uncertainty_by_position[i])
	return sum(diffs) / half_length

def plot_guidelines(axis, fixation_positions, mean_uncertainty, color=None):
	if color is None:
		color = 'gray'
	if len(fixation_positions) <= 2:
		return
	middle = len(fixation_positions) // 2 + 1
	first_x = fixation_positions[:middle]
	first_y = mean_uncertainty[:middle]
	last_x = fixation_positions[middle-1:][::-1]
	last_y = mean_uncertainty[middle-1:][::-1]
	slopes = []
	for i in range(len(last_x)-(len(fixation_positions)%2)):
		axis.plot([first_x[i], last_x[i]], [first_y[i], last_y[i]], c=color, linestyle=':', linewidth=.5)
		slopes.append((last_y[i] - first_y[i]) / (last_x[i] - first_x[i]))
	return sum(slopes) / len(slopes)

def plot_uncertainty(axis, uncertainty_by_position, label=None, color=None):
	length = len(uncertainty_by_position)
	positions = list(range(1, length+1))
	axis.plot(positions, uncertainty_by_position, label=label, linewidth=1, color=color)
	mean_slope = plot_guidelines(axis, positions, uncertainty_by_position, color=color)
	xpad = (length - 1) * 0.05
	axis.set_xlim(1-xpad, length+xpad)
	axis.set_ylim(3, 9)
	axis.set_xticks(range(1, length+1))
	axis.set_xticklabels(range(1, length+1))
	axis.set_xlabel('Fixation position')
	axis.set_ylabel('Uncertainty (bits)')
	axis.set_title('%i-letter words'%(length), fontsize=7, y=0.8)
	# axis.text(1, 0, 'h')

def plot_languages(figure_file, uncertainty_data, lengths, languages):
	figure = core.Figure(len(lengths), 3, (7.09, 2))
	for axis, length in zip(figure, lengths):
		for language in languages:
			uncertainty_by_position = core.unpickle(uncertainty_data + f'{language}_{length}.pkl')
			diff = deficit(uncertainty_by_position)
			print(language, diff)
			plot_uncertainty(axis, uncertainty_by_position, label=language_names[language], color=language_colors[language])
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


# plot_languages('../manuscript/figs/typ_uncertainty1.eps', '../data/typ_uncertainty/gamma1/', [5, 7, 9], ['de', 'en', 'es', 'gr', 'it', 'nl', 'pl', 'sw'])
plot_languages('../manuscript/figs/typ_uncertainty2.eps', '../data/typ_uncertainty/gamma2/', [5, 7, 9], ['de', 'en', 'es', 'gr', 'it', 'nl', 'pl', 'sw'])
