def deficit(uncertainty_by_position):
	length = len(uncertainty_by_position)
	half_length = length // 2
	diffs = []
	for i, j in zip(range(half_length), range(length-1, half_length, -1)):
		print(i, j, uncertainty_by_position[j] - uncertainty_by_position[i])
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

def plot_uncertainty(axis, uncertainty_by_position, label=None, color=None, ylim=(3, 8), show_guidelines=True):
	length = len(uncertainty_by_position)
	positions = list(range(1, length+1))
	axis.plot(positions, uncertainty_by_position, label=label, linewidth=1, color=color)
	if show_guidelines:
		mean_slope = plot_guidelines(axis, positions, uncertainty_by_position, color=color)
	xpad = (length - 1) * 0.05
	axis.set_xlim(1-xpad, length+xpad)
	axis.set_ylim(*ylim)
	axis.set_xticks(range(1, length+1))
	axis.set_xticklabels(range(1, length+1))
	axis.set_xlabel('Fixation position')
	axis.set_ylabel('Uncertainty (bits)')
	axis.set_title('%i-letter words'%(length), fontsize=7)
