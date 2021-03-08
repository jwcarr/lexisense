from os import listdir, path, walk
from subprocess import call, STDOUT, DEVNULL
import pickle as pickle_
import json
import re
import numpy as np
import matplotlib.pyplot as plt
import cairosvg

plt.rcParams['svg.fonttype'] = 'none' # don't convert fonts to curves in SVGs
plt.rcParams.update({'font.size': 7})

class Figure:

	def __init__(self, n_subplots, n_cols, figsize=None):
		self.n_cols = n_cols
		self.n_rows = n_subplots // self.n_cols
		if n_subplots % self.n_cols > 0:
			self.n_rows += 1
		if figsize is None:
			figsize = self.n_cols * 3, self.n_rows * 2.2
		self.fig, self.axes = plt.subplots(self.n_rows, self.n_cols, figsize=figsize, squeeze=False)

	def __iter__(self):
		for axis_index in np.ndindex((self.n_rows, self.n_cols)):
			yield self.axes[axis_index]

	def iter_rows(self):
		for row_index in range(self.n_rows):
			yield self.axes[row_index, :]

	def iter_cols(self):
		for col_index in range(self.n_cols):
			yield self.axes[:, col_index]

	def deduplicate_axes(self):
		for row in self.axes:
			if len(set([cell.get_ylabel() for cell in row])) == 1:
				for i in range(1, self.n_cols):
					row[i].set_ylabel('')
			if len(set([str(cell.get_yticks()) for cell in row])) == 1:
				for i in range(1, self.n_cols):
					row[i].set_yticklabels([])
		for col in self.axes.T:
			if len(set([cell.get_xlabel() for cell in col])) == 1:
				for i in range(self.n_rows-1):
					col[i].set_xlabel('')
			if len(set([str(cell.get_xticks()) for cell in col])) == 1:
				for i in range(self.n_rows-1):
					col[i].set_xticklabels([])

	def save(self, figure_file, title=None):
		# for remaining_axis in self:
		# 	remaining_axis.axis('off')
		self.deduplicate_axes()
		if title is not None:
			self.fig.suptitle(title)
			self.fig.tight_layout(pad=0.5, h_pad=1, w_pad=1, rect=(0, 0, 1, 0.95))
		else:
			self.fig.tight_layout(pad=0.5, h_pad=1, w_pad=1)
		self.fig.savefig(figure_file, format='svg')
		format_svg_labels(figure_file)
		if not figure_file.endswith('.svg'):
			convert_svg(figure_file, figure_file)


def convert_svg(svg_file_path, out_file_path, png_width=1000):
	filename, extension = path.splitext(out_file_path)
	if extension == '.pdf':
		cairosvg.svg2pdf(url=svg_file_path, write_to=out_file_path)
	elif extension == '.eps':
		cairosvg.svg2eps(url=svg_file_path, write_to=out_file_path)
	elif extension == '.png':
		cairosvg.svg2png(url=svg_file_path, write_to=out_file_path, dpi=300)
	else:
		raise ValueError('Cannot save to this format. Use either .pdf, .eps, or .png')

def format_svg_labels(svg_file_path):
	'''
	Applies some nicer formatting to an SVG plot, including setting
	the font to Helvetica and adding italics. Requires you to set
	this at the top of the script:
	plt.rcParams['svg.fonttype'] = 'none'
	'''
	with open(svg_file_path, mode='r', encoding='utf-8') as file:
		svg = file.read()
	svg = re.sub(r'font-family:.*?;', 'font-family:Helvetica Neue;', svg)
	with open(svg_file_path, mode='w', encoding='utf-8') as file:
		file.write(svg)

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
