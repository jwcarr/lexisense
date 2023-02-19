import csv
from pathlib import Path
import eyekit
from experiment import Experiment


ROOT = Path(__file__).parent.parent.resolve()
EXP_DATA = ROOT / 'data' / 'experiments'


def calculate_landing_position(trial):
	seq = trial['fixations']
	word_ia = trial['word'][0:0:7]
	for fixation in seq:
		if fixation.start > trial['start_word_presentation'] and fixation.start < trial['end_word_presentation']:
			if fixation in word_ia:
				px_position = int(fixation.x - word_ia.onset)
				return px_position
	return None

def write_csv(file_path, header, table):
	with open(file_path, 'w') as csv_file:
		csv_writer = csv.writer(csv_file)
		csv_writer.writerow(header)
		csv_writer.writerows(table)



exp1 = Experiment('exp1')

table = []
for subject_i, participant in enumerate(exp1):
	condition = participant.task_id.split('_')[1]
	if participant.learning_score() < 7:
		continue
	for trial_i, trial in enumerate(participant.iter_training_trials()):
		table.append([
			subject_i,
			condition,
			'training',
			trial_i,
			None,
			trial['target_item'],
			trial['selected_item'],
			int(trial['selected_item'] == trial['target_item']),
		])
	for trial_i, trial in enumerate(participant.iter_all_test_trials()):
		table.append([
			subject_i,
			condition,
			'test',
			trial_i,
			trial['fixation_position'],
			trial['target_item'],
			trial['selected_item'],
			int(trial['selected_item'] == trial['target_item']),
		])

header = ['subject', 'condition', 'trial_type', 'trial_num', 'fixation_position', 'target_item', 'selected_item', 'correct']

write_csv(EXP_DATA / 'exp1.csv', header, table)



exp2 = Experiment('exp2')

table = []
for subject_i, participant in enumerate(exp2):
	condition = participant.task_id.split('_')[1]
	if participant.learning_score() < 7:
		continue
	for trial_i, trial in enumerate(participant.iter_training_trials()):
		table.append([
			subject_i,
			condition,
			'training',
			trial_i,
			trial['target_item'],
			trial['selected_item'],
			int(trial['selected_item'] == trial['target_item']),
			None,
		])
	for trial_i, trial in enumerate(participant.iter_all_test_trials()):
		table.append([
			subject_i,
			condition,
			'test',
			trial_i,
			trial['target_item'],
			trial['selected_item'],
			int(trial['selected_item'] == trial['target_item']),
			calculate_landing_position(trial),
		])

header = ['subject', 'condition', 'trial_type', 'trial_num', 'target_item', 'selected_item', 'correct', 'landing_position']

write_csv(EXP_DATA / 'exp2.csv', header, table)
