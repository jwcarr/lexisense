from os import listdir, path, walk
import pickle as pickle_
import json


def iter_directory(directory):
	for root, dirs, files in walk(directory, topdown=False):
		for file in files:
			if file[0] != '.':
				yield path.join(root, file)

def iter_passage_ids():
	for passage_id in ['1A', '1B', '2A', '2B', '3A', '3B', '4A', '4B', '5A', '5B', '6A', '6B']:
		yield passage_id

def iter_participants(data_file, data_file2=None):
	with open(data_file, mode='r') as file:
		data = json.load(file)
	for participant_id, participant_data in data.items():
		yield participant_id, participant_data
	if data_file2:
		with open(data_file2, mode='r') as file:
			data = json.load(file)
		for participant_id, participant_data in data.items():
			yield participant_id, participant_data

def load_fixation_data(fixation_data_file):
	with open(fixation_data_file, mode='r') as file:
		return json.load(file)

def load_passages(passages_dir):
	passages = {}
	for passage_file in listdir(passages_dir):
		if not passage_file.endswith('.txt'):
			continue
		passage_id, _ = passage_file.split('.')
		passage_path = path.join(passages_dir, passage_file)
		passages[passage_id] = eye_tracking.Passage(passage_path,
			first_character_position=(368, 155),
			character_spacing=16,
			line_spacing=64)
	return passages

def load_duration_masses(duration_mass_dir, max_n, arrangement='by_passage'):
	duration_mass_dir = path.join(duration_mass_dir, arrangement)
	duration_masses = {}
	for n in range(1, max_n+1):
		duration_mass_file = path.join(duration_mass_dir, '%igram.pkl'%n)
		duration_masses[n] = unpickle(duration_mass_file)
	return duration_masses

def load_ngram_freqs(frequencies_dir, max_n):
	ngram_freqs = {}
	for n in range(1, max_n+1):
		freqs_file = path.join(frequencies_dir, '%igram.pkl'%n)
		ngram_freqs[n] = unpickle(freqs_file)
	return ngram_freqs

def load_expected_ngram_freqs(frequencies_dir, max_n):
	adjusted_ngram_freqs = {}
	for n in range(2, max_n+1):
		freqs_file = path.join(frequencies_dir, '%igram_from_%igram.pkl'%(n, n-1))
		adjusted_ngram_freqs[n] = unpickle(freqs_file)
	return adjusted_ngram_freqs

def load_scrambled_ngram_freqs(frequencies_dir, max_n):
	ngram_freqs = {}
	for n in range(2, max_n+1):
		freqs_file = path.join(frequencies_dir, 'scrambled_%igram.pkl'%n)
		ngram_freqs[n] = unpickle(freqs_file)
	return ngram_freqs

def pickle(obj, file_path):
	with open(file_path, mode='wb') as file:
		pickle_.dump(obj, file)

def unpickle(file_path):
	with open(file_path, mode='rb') as file:
		return pickle_.load(file)

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
