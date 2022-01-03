try:
	import mplcairo
	import matplotlib
	matplotlib.use("module://mplcairo.macosx")
except:
	pass

import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


plt.rcParams.update({'font.sans-serif':'Helvetica Neue', 'font.size': 7})


# Widths of single and double column figures
SINGLE_COLUMN_WIDTH = 3.46 # 88mm
DOUBLE_COLUMN_WIDTH = 7.09 # 180mm


def mm_to_inches(mm):
	return mm / 25.4


class Figure:

	def __init__(self, file_path, n_rows=1, n_cols=1, width='single', height=None):
		self.file_path = Path(file_path).resolve()
		
		self.n_rows = n_rows
		self.n_cols = n_cols

		if width == 'single':
			self.width = SINGLE_COLUMN_WIDTH
		elif width == 'double':
			self.width = DOUBLE_COLUMN_WIDTH
		else:
			self.width = mm_to_inches(width)

		if height is None:
			self.height = (self.width / self.n_cols) * self.n_rows / (2**0.5)
		else:
			self.height = mm_to_inches(height)
		
		self.auto_deduplicate_axes = True

	def __enter__(self):
		self.fig, self.axes = plt.subplots(self.n_rows, self.n_cols, figsize=(self.width, self.height), squeeze=False)
		self.used = [[False]*self.n_cols for _ in range(self.n_rows)]
		return self

	def __exit__(self, exc_type, exc_value, exc_traceback):
		if self.auto_deduplicate_axes:
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

	def next(self):
		pass

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
