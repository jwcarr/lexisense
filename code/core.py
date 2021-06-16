from pathlib import Path
import json

# Paths to common project directories
ROOT = Path(__file__).parent.parent.resolve()
DATA = ROOT / 'data'
FIGS = ROOT / 'manuscript' / 'figs'
VISUALS = ROOT / 'visuals'
EXP_DATA = DATA / 'experiments' / 'online'


# Widths of single and double column figures
single_column_width = 3.46 # 88mm
double_column_width = 7.09 # 180mm


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


try:
	import mplcairo
	import matplotlib
	matplotlib.use("module://mplcairo.macosx")
except:
	pass

import matplotlib.pyplot as plt
plt.rcParams.update({'font.sans-serif':'Helvetica Neue', 'font.size': 7})

class Figure:

	def __init__(self, file_path, n_subplots, n_cols=None, width='single', height=None):
		self.file_path = Path(file_path).resolve()

		if n_cols is None:
			self.n_cols = n_subplots
		else:
			self.n_cols = n_cols

		self.n_rows = n_subplots // self.n_cols
		if n_subplots % self.n_cols > 0:
			self.n_rows += 1

		if width == 'single':
			self.width = single_column_width
		elif width == 'double':
			self.width = double_column_width
		else:
			self.width = width

		if height is None:
			self.height = (self.width / self.n_cols) * self.n_rows
		else:
			self.height = height

	def __enter__(self):
		self.fig, self.axes = plt.subplots(self.n_rows, self.n_cols, figsize=(self.width, self.height), squeeze=False)
		self.used = [[False]*self.n_cols for _ in range(self.n_rows)]
		return self

	def __exit__(self, exc_type, exc_value, exc_traceback):
		self.deduplicate_axes()
		self.turn_off_unused_axes()
		self.fig.tight_layout(pad=0.5, h_pad=1, w_pad=1)
		self.fig.savefig(self.file_path)
		plt.close(self.fig)

	def __getitem__(self, index):
		i, j = index
		self.used[i][j] = True
		return self.axes[i,j]

	def __iter__(self):
		for i in range(self.n_rows):
			for j in range(self.n_cols):
				yield self[i,j]

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

	def turn_off_unused_axes(self):
		for i, row in enumerate(self.axes):
			for j, cell in enumerate(row):
				if not self.used[i][j]:
					cell.axis('off')

	def unpack(self):
		return (axis for axis in self)

	def unpack_row(self, row_i):
		return (self[row_i, j] for j in range(self.n_cols))

	def unpack_column(self, col_j):
		return (self[i, col_j] for i in range(self.n_rows))
