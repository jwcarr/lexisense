import pickle as pickle
import json
from pathlib import Path

# Paths to common project directories
ROOT = Path(__file__).parent.parent.resolve()
DATA = ROOT / 'data'
FIGS = ROOT / 'manuscript' / 'figs'
VISUALS = ROOT / 'visuals'


def pickle_write(obj, file_path):
	with open(file_path, mode='wb') as file:
		pickle.dump(obj, file)

def pickle_read(file_path):
	with open(file_path, mode='rb') as file:
		return pickle.load(file)

def json_write(obj, file_path):
	with open(file_path, 'w') as file:
		json.dump(obj, file, indent='\t')

def json_read(file_path):
	with open(file_path) as file:
		return json.load(file)
