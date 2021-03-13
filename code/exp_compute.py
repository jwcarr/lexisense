'''

This file computes uncertainty by fixation position on a set of experimental
words. It is designed to be run on a cluster. First you compute the entropy of
the posterior distribution of a target word in a given fixation position:

	python exp_compute.py --language exp1_left --position 0 --target 0

Each target and each positon is run as a separate process.

Once this has been completed for all targets/positions, you merge all this
data together:

	python exp_compute.py --merge

To create shell scripts for all language/length combinations:

	python exp_compute.py --script

'''

import json
import pickle
import model

LEXICON_LOCATION = 'lex'
RESULTS_LOCATION = 'eres'
SLURM_SCRIPT = '''#!/bin/bash
#SBATCH -N1
#SBATCH -n1
#SBATCH --time=0:03:00
module load python3/3.8
python exp_compute.py --language {language} --position {position} --target $SLURM_ARRAY_TASK_ID --alpha {alpha} --beta {beta} --gamma {gamma}
'''


def load_lexicon(language):
	with open(f'{LEXICON_LOCATION}/{args.language}.json') as file:
		json_data = json.load(file)
	return list(map(tuple, json_data['words']))

def read_result(language, position, target):
	with open(f'{RESULTS_LOCATION}/{language}_{position}_{target}.pkl', 'rb') as file:
		return pickle.load(file)

def write_result(language, position, target, uncertainty):
	with open(f'{RESULTS_LOCATION}/{language}_{position}_{target}.pkl', 'wb') as file:
		pickle.dump(uncertainty, file)

def script(languages, n_positions, n_words, alpha, beta, gamma):
	for language in languages:
		for position in range(n_positions):
			script = SLURM_SCRIPT.format(language=language, position=position, alpha=alpha, beta=beta, gamma=gamma)
			filename = f'{language}_{position}.sh'
			with open(filename, 'w') as file:
				file.write(script)
			print(f'sbatch --array=0-{n_words-1} -p regular2 {filename}')

def merge(languages, n_positions, n_words):
	for language in languages:
		uncertainty_by_position = [0] * n_positions
		for position in range(n_positions):
			for target in range(n_words):
				uncertainty_by_position[position] += read_result(language, position, target)
			uncertainty_by_position[position] /= n_words
		with open(f'{RESULTS_LOCATION}/{language}.pkl', 'wb') as file:
			pickle.dump(uncertainty_by_position, file)


if __name__ == '__main__':

	import argparse
	
	parser = argparse.ArgumentParser()
	parser.add_argument('--language', action='store', type=str, help='language code')
	parser.add_argument('--position', action='store', type=int, help='fixation position')
	parser.add_argument('--target', action='store', type=int, help='target word')

	parser.add_argument('--alpha', action='store', type=float, default=0.9, help='alpha parameter')
	parser.add_argument('--beta', action='store', type=float, default=0.1, help='beta parameter')
	parser.add_argument('--gamma', action='store', type=float, default=0, help='gamma parameter')

	parser.add_argument('--script', action='store_true', help='generate slurm scripts')
	parser.add_argument('--merge', action='store_true', help='merge the results into a matrix')
	args = parser.parse_args()

	if args.script:
		script(['e1_left', 'e1_right'], 5, 8, args.alpha, args.beta, args.gamma)
		exit()

	if args.merge:
		merge(['e1_left', 'e1_right'], 5, 8)
		exit()
	
	lexicon = load_lexicon(args.language)
	reader = model.Reader(lexicon, args.alpha, args.beta, args.gamma)
	uncertainty = reader.uncertainty(args.target, args.position, exact=True)
	write_result(args.language, args.position, args.target, uncertainty)
