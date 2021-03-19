'''

This file computes uncertainty by fixation position on the typological corpus
data. It is designed to be run on a cluster. First you compute the entropy of
the posterior distribution of each target word in a given fixation position:

	python typ_compute.py --language en --length 5 --position 0

Each fixation position should be run as a separate process. The results will
periodically be stored in case the process is killed (and these stored results
should be resurrected automatically).

Once this has been completed for all fixation positions, you can calculate
uncertainty by fixation position:

	python typ_compute.py --language en --length 5 --merge

In the example above, we are computing the results for 5-letter English words.
This can then be repeated for each language/length combination. To create
shell scripts for all language/length combinations:

	python typ_compute.py --script

To check the status of a language/length combination, use:

	python typ_compute.py --check --language en --length 5 
'''

import pickle
import model

LEXICON_LOCATION = 'lex'
RESULTS_LOCATION = 'res'
SLURM_SCRIPT = '''#!/bin/bash
#SBATCH -N1
#SBATCH -n1
#SBATCH --time={time}
module load python3/3.8
python typ_compute.py --language {language} --length {length} --position $SLURM_ARRAY_TASK_ID --alpha {alpha} --beta {beta} --gamma {gamma}
'''
TIMES = {5:'0:30:00', 6:'1:00:00', 7:'2:00:00', 8:'4:00:00', 9:'8:00:00'} # e.g. length 5 results require 30mins of compute time


def load_lexicon(language, length):
	with open(f'{LEXICON_LOCATION}/{args.language}.pkl', 'rb') as file:
		lexicon_by_length = pickle.load(file)
	return lexicon_by_length[length]

def read_results(language, length, position):
	try:
		with open(f'{RESULTS_LOCATION}/{language}_{length}_{position}.pkl', 'rb') as file:
			uncertainty_by_target = pickle.load(file)
		return uncertainty_by_target
	except FileNotFoundError:
		return {}

def write_results(language, length, position, uncertainty_by_target):
	with open(f'{RESULTS_LOCATION}/{language}_{length}_{position}.pkl', 'wb') as file:
		pickle.dump(uncertainty_by_target, file)

def script(languages, lengths, alpha, beta, gamma):
	for language in languages:
		for length in lengths:
			script = SLURM_SCRIPT.format(language=language, length=length, time=TIMES[length], alpha=alpha, beta=beta, gamma=gamma)
			filename = f'{language}_{length}.sh'
			with open(filename, 'w') as file:
				file.write(script)
			print(f'sbatch --array=0-{length-1} -p regular2 {filename}')

def check(lexicon, language, length):
	for position in range(length):
		entropy_by_target = read_results(language, length, position)
		completed_count = 0
		for target_word in lexicon:
			if target_word in entropy_by_target:
				completed_count += 1
		percentage = int((completed_count / len(lexicon)) * 100)
		print(f'{language}_{length}_{position}: {percentage}% complete')

def merge(lexicon, language, length):
	uncertainty_by_position = []
	for position in range(length):
		entropy_by_target = read_results(language, length, position)
		U = sum([probability * entropy_by_target[target_word] for target_word, probability in lexicon.items()])
		uncertainty_by_position.append(U)
	with open(f'{RESULTS_LOCATION}/{language}_{length}.pkl', 'wb') as file:
		pickle.dump(uncertainty_by_position, file)


if __name__ == '__main__':

	import argparse
	
	parser = argparse.ArgumentParser()
	parser.add_argument('--language', action='store', type=str, help='language code')
	parser.add_argument('--length', action='store', type=int, help='word length')
	parser.add_argument('--position', action='store', type=int, help='fixation position')

	parser.add_argument('--alpha', action='store', type=float, default=0.9, help='alpha parameter')
	parser.add_argument('--beta', action='store', type=float, default=0.1, help='beta parameter')
	parser.add_argument('--gamma', action='store', type=float, default=0, help='gamma parameter')

	parser.add_argument('--script', action='store_true', help='generate slurm scripts')
	parser.add_argument('--check', action='store_true', help='print the percentage of targets that have been completed')
	parser.add_argument('--merge', action='store_true', help='merge the results into a matrix')
	args = parser.parse_args()

	if args.script:
		script(['de', 'en', 'es', 'gr', 'it', 'nl', 'pl', 'sw'], [5, 6, 7, 8, 9], args.alpha, args.beta, args.gamma)
		exit()
	
	lexicon = load_lexicon(args.language, args.length)

	if args.check:
		check(lexicon, args.language, args.length)
		exit()

	if args.merge:
		merge(lexicon, args.language, args.length)
		exit()

	entropy_by_target = read_results(args.language, args.length, args.position)
	reader = model.Reader(lexicon, args.alpha, args.beta, args.gamma)
	for t, target_word in enumerate(reader.lexicon):
		if target_word in entropy_by_target:
			continue
		entropy_by_target[target_word] = reader.uncertainty(target_word, args.position)
		if t % 100 == 0:
			write_results(args.language, args.length, args.position, entropy_by_target)
	write_results(args.language, args.length, args.position, entropy_by_target)
