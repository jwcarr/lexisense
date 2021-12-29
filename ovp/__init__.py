from pathlib import Path
import json
from figure import Figure


# Paths to common project directories
ROOT = Path(__file__).parent.parent.resolve()
DATA = ROOT / 'data'
FIGS = ROOT / 'manuscript' / 'figs'
VISUALS = ROOT / 'visuals'
EXP_DATA = DATA / 'experiments'
MODEL_FIT = DATA / 'model_fit'


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
	'de': 'black',
	'en': 'navy',
	'es': 'yellow',
	'gr': 'blue',
	'it': 'green',
	'nl': 'orange',
	'pl': 'red',
	'sw': 'purple',
}


def json_write(obj, file_path, compress=False):
	if compress:
		with open(file_path, 'w') as file:
			json.dump(obj, file, separators=(',', ':'))
	else:
		with open(file_path, 'w') as file:
			json.dump(obj, file, indent='\t', separators = (',', ': '))

def json_read(file_path):
	with open(file_path) as file:
		return json.load(file)


