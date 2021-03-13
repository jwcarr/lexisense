from pathlib import Path

try:
	import mplcairo
	import matplotlib
	matplotlib.use("module://mplcairo.macosx")
except:
	pass

import matplotlib.pyplot as plt
plt.rcParams.update({'font.sans-serif':'Helvetica Neue', 'font.size': 7})


SINGLE_COLUMN_WIDTH = 3.46 # 88mm
DOUBLE_COLUMN_WIDTH = 7.09 # 180mm


class Figure:

	def __init__(self, n_subplots, n_cols, width='single', height_adjust=1):
		self.n_cols = n_cols
		self.n_rows = n_subplots // self.n_cols
		if n_subplots % self.n_cols > 0:
			self.n_rows += 1
		if width == 'single':
			figsize = (SINGLE_COLUMN_WIDTH, SINGLE_COLUMN_WIDTH / self.n_cols * self.n_rows * height_adjust)
		elif width == 'double':
			figsize = (DOUBLE_COLUMN_WIDTH, DOUBLE_COLUMN_WIDTH / self.n_cols * self.n_rows * height_adjust)
		else:
			figsize = (width, width / self.n_cols * self.n_rows * height_adjust)
		self.fig, self.axes = plt.subplots(self.n_rows, self.n_cols, figsize=figsize, squeeze=False)
		self.used = [[False]*self.n_cols for _ in range(self.n_rows)]

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

	def save(self, figure_file, title=None):
		figure_file = Path(figure_file).resolve()
		self.deduplicate_axes()
		self.turn_off_unused_axes()
		if title is not None:
			self.fig.suptitle(title)
			self.fig.tight_layout(pad=0.5, h_pad=1, w_pad=1, rect=(0, 0, 1, 0.95))
		else:
			self.fig.tight_layout(pad=0.5, h_pad=1, w_pad=1)
		self.fig.savefig(figure_file)
