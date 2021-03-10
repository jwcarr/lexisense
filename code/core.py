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

lexicon_l = [
	(15,  7, 3, 1, 0),
	(16,  8, 3, 1, 0),
	(17,  9, 4, 1, 0),
	(18, 10, 4, 1, 0),
	(15, 11, 5, 2, 0),
	(16, 12, 5, 2, 0),
	(17, 13, 6, 2, 0),
	(18, 14, 6, 2, 0),
]

lexicon_r = [
	(0, 1, 3,  7, 15),
	(0, 1, 3,  8, 16),
	(0, 1, 4,  9, 17),
	(0, 1, 4, 10, 18),
	(0, 2, 5, 11, 15),
	(0, 2, 5, 12, 16),
	(0, 2, 6, 13, 17),
	(0, 2, 6, 14, 18),
]
