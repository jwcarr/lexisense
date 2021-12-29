'''

This file computes uncertainty by fixation position on the typological corpus
data. It is designed to be run on a cluster. Since this takes a long time to
run, each fixation position is computed separately, like this:

	python typ_compute.py --language en --length 5 --position 0

To create shell scripts for all languages/lengths/positions, use:

	python typ_compute.py --script

Once all jobs have been completed, the results can be merged:

	python typ_compute.py --merge

'''

import json
import model


LEXICON_LOCATION = 'lex'
RESULTS_LOCATION = 'res'

LANGUAGES = ['de', 'en', 'es', 'gr', 'it', 'nl', 'pl', 'sw']
LENGTHS = [5, 6, 7, 8, 9]

SLURM_SCRIPT = '''#!/bin/bash
#SBATCH -N1
#SBATCH -n1
#SBATCH --time=0:20:00
module load python3/3.8
python typ_compute.py --language {language} --length {length} --position $SLURM_ARRAY_TASK_ID --alpha {alpha} --beta {beta} --gamma {gamma}
'''


def load_lexicon(language, length):
	with open(f'{LEXICON_LOCATION}/{args.language}.json') as file:
		lexicon_by_length = json.load(file)
	return lexicon_by_length[str(length)]

def write_result(language, length, position, uncertainty):
	with open(f'{RESULTS_LOCATION}/{language}_{length}_{position}', 'w') as file:
		file.write(str(uncertainty))

def read_result(language, length, position):
	with open(f'{RESULTS_LOCATION}/{language}_{length}_{position}') as file:
		return float(file.read())

def script(languages, lengths, alpha, beta, gamma):
	for language in languages:
		for length in lengths:
			script = SLURM_SCRIPT.format(language=language, length=length, alpha=alpha, beta=beta, gamma=gamma)
			filename = f'{language}_{length}.sh'
			with open(filename, 'w') as file:
				file.write(script)
			print(f'sbatch --array=0-{length-1} -p regular2 {filename}')

def merge(languages, lengths):
	for language in languages:
		language_data = {}
		for length in lengths:
			language_data[str(length)] = []
			for position in range(length):
				uncertainty = read_result(language, length, position)
				language_data[str(length)].append(uncertainty)
		with open(f'{RESULTS_LOCATION}/{language}.json', 'w') as file:
			json.dump(language_data, file, indent='\t')


if __name__ == '__main__':

	import argparse
	
	parser = argparse.ArgumentParser()
	parser.add_argument('--language', action='store', type=str, help='language code')
	parser.add_argument('--length', action='store', type=int, help='word length')
	parser.add_argument('--position', action='store', type=int, help='fixation position')

	parser.add_argument('--alpha', action='store', type=float, default=0.9, help='alpha parameter')
	parser.add_argument('--beta', action='store', type=float, default=0.2, help='beta parameter')
	parser.add_argument('--gamma', action='store', type=float, default=0.0, help='gamma parameter')

	parser.add_argument('--script', action='store_true', help='generate slurm scripts')
	parser.add_argument('--merge', action='store_true', help='merge the results into a single file')
	args = parser.parse_args()

	if args.script:
		script(LANGUAGES, LENGTHS, args.alpha, args.beta, args.gamma)
		exit()

	if args.merge:
		merge(LANGUAGES, LENGTHS)
		exit()

	lexicon = load_lexicon(args.language, args.length)
	reader = model.Reader(lexicon, args.alpha, args.beta, args.gamma)
	uncertainty = reader.uncertainty(args.position, method='fast', n_sims=1000)
	write_result(args.language, args.length, args.position, uncertainty)
