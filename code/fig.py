from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt
import cairosvg


plt.rcParams['svg.fonttype'] = 'none'
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

	def __getitem__(self, index):
		return self.axes[index]

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
		figure_file = Path(figure_file).resolve()
		# for remaining_axis in self:
		# 	remaining_axis.axis('off')
		self.deduplicate_axes()
		if title is not None:
			self.fig.suptitle(title)
			self.fig.tight_layout(pad=0.5, h_pad=1, w_pad=1, rect=(0, 0, 1, 0.95))
		else:
			self.fig.tight_layout(pad=0.5, h_pad=1, w_pad=1)
		self.fig.savefig(figure_file, format='svg')
		self._format_labels(figure_file)
		if figure_file.suffix == '.pdf':
			cairosvg.svg2pdf(url=str(figure_file), write_to=str(figure_file))
		elif figure_file.suffix == '.eps':
			cairosvg.svg2eps(url=str(figure_file), write_to=str(figure_file))
		elif figure_file.suffix == '.png':
			cairosvg.svg2png(url=str(figure_file), write_to=str(figure_file), dpi=300)
		elif figure_file.suffix != '.svg':
			raise ValueError('Cannot save to this format. Use either .pdf, .eps, .svg, or .png')

	def _format_labels(self, svg_file_path):
		'''
		
		Applies some nicer formatting to an SVG plot, including setting
		the font to Helvetica and adding italics.

		'''
		with open(svg_file_path, mode='r', encoding='utf-8') as file:
			svg = file.read()
		svg = re.sub(r'font-family:.*?;', 'font-family:Helvetica Neue;', svg)
		with open(svg_file_path, mode='w', encoding='utf-8') as file:
			file.write(svg)
