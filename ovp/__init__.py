from pathlib import Path
import json
from .plots import Figure
from .experiment import Experiment


# Paths to common project directories
ROOT = Path(__file__).parent.parent.resolve()
DATA = ROOT / 'data'
FIGS = ROOT / 'manuscript' / 'figs'
RESULTS = ROOT / 'results'
EXP_DATA = DATA / 'experiments'
MODEL_FIT = DATA / 'model_fit'


experiment.DATA_DIR = EXP_DATA
experiment.MODEL_FIT_DIR = MODEL_FIT


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


