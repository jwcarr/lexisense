'''
This script extracts the relevant fixation data from the ASC file and merges
it into the participant's JSON file. Run from the command line like so:

	python merge_fixation_data.py exp2_left 01

'''

from collections import defaultdict
from pathlib import Path
import eyekit


ROOT = Path(__file__).parent.parent.resolve()
EXP_DATA = ROOT / 'data' / 'experiments'


asc_variables = [
	'trial_type',
	'target_item',
	'word_position_x',
	'word_position_y',
	'start_word_presentation',
	'end_word_presentation',
	'trial_abandoned'
]


def partition_trials_by_type(trials):
	'''
	Iterate over non-abandoned trials and separate them by the trial_type
	variable.
	'''
	trials_by_type = defaultdict(list)
	for trial in trials:
		if trial['trial_abandoned']:
			continue
		trials_by_type[trial['trial_type']].append(trial)
	return trials_by_type

def extract_trials_from_asc_file(asc_path):
	'''
	Use Eyekit to extract trials from an ASC file and return the trials
	partitioned by type.
	'''
	extracted_trials = eyekit.io.import_asc(asc_path,
		variables=[
			'trial_type',
			'target_item',
			'word_position_x',
			'word_position_y',
			'start_word_presentation',
			'end_word_presentation',
			'trial_abandoned',
		]
	)
	extracted_trials_by_type = partition_trials_by_type(extracted_trials)
	return extracted_trials_by_type

def create_text_block(trial, word_forms):
	'''
	Create an Eyekit TextBlock object for a given trial.
	'''
	x, y = trial['word_position']
	return eyekit.TextBlock(word_forms[trial['target_item']],
		position=(x, y+20), # +20 because PsychoPy's anchor point is 20px above the baseline
		font_face='Courier New',
		font_size=60,
		align='center',
		autopad=False,
	)

def merge_fixations_into_user_data(json_path, asc_path):
	'''
	Extract fixations from an ASC file and merge them into the user's JSON file.
	'''
	user_data = eyekit.io.load(json_path)
	extracted_trials_by_type = extract_trials_from_asc_file(asc_path)
	for trial_type, extracted_trials in extracted_trials_by_type.items():
		for trial, fixation_data in zip(user_data['responses'][trial_type], extracted_trials):
			# check that data logged in JSON matches data logged in ASC
			assert trial['target_item'] == int(fixation_data['target_item'])
			assert trial['word_position'][0] == int(fixation_data['word_position_x'])
			assert trial['word_position'][1] == int(fixation_data['word_position_y'])
			# make fixation sequence start at time = 0
			shift = fixation_data['fixations'].shift_start_time_to_zero()
			trial['start_word_presentation'] = fixation_data['start_word_presentation'] - shift
			trial['end_word_presentation'] = fixation_data['end_word_presentation'] - shift
			# add fixation data to JSON
			trial['fixations'] = fixation_data['fixations']
			trial['word'] = create_text_block(trial, user_data['word_forms'])
	eyekit.io.save(user_data, json_path)


if __name__ == '__main__':

	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument('task_id', action='store', type=str, help='Task ID')
	parser.add_argument('user_id', action='store', type=str, help='User ID')
	args = parser.parse_args()

	json_path = EXP_DATA / args.task_id / f'{args.user_id}.json'
	asc_path = EXP_DATA / args.task_id / f'{args.user_id}.asc'

	merge_fixations_into_user_data(json_path, asc_path)
