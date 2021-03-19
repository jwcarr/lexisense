from itertools import product
import numpy as np


class Reader:

	'''

	Model reader.

	'''

	def __init__(self, lexicon, alpha, beta, gamma=0):
		'''

		- lexicon : A list or dictionary containing the word items. If a list, each
		word should be a string of letters or tuple of integers, all of the same
		length. The words are assumed to have a uniform prior. Alternatively, for a
		non-uniform prior, a dictionary can be supplied with words as keys and
		probabilities as values.

		- alpha : A float >= 1/|S| and < 1. The alpha parameter controls the probability
		that the reader will correctly identify the character under fixation. 

		- beta : A float > 0. The beta parameter controls the rate at which the
		probability of successful letter identification approaches chance with
		distance from the fixation position.

		- gamma : A float > 1. The gamma parameter controls how much faster the
		probability of successful letter identification drops to the left vs. to the
		right. If gamma is 1, the perceptual filter is symmetrical.
		
		'''
		if isinstance(lexicon, dict):
			self.lexicon = list(lexicon.keys())
			self.lexicon_size = len(self.lexicon)
			self.prior = np.log([lexicon[word] for word in self.lexicon])
		elif isinstance(lexicon, list):
			self.lexicon = lexicon
			self.lexicon_size = len(self.lexicon)
			self.prior = np.log([1 / self.lexicon_size] * self.lexicon_size)
		else:
			raise ValueError('lexicon should be of type list or dict')
		
		self.word_length = len(self.lexicon[0])
		for word in self.lexicon:
			if len(word) != self.word_length:
				raise ValueError('All words in the lexicon must be of the same length')

		if isinstance(self.lexicon[0], tuple):
			symbols = set(sum(self.lexicon, tuple()))
			if symbols != set(range(len(symbols))):
				raise ValueError('Invalid tuple words')
			self.symbols = list(range(len(symbols)))
		elif isinstance(self.lexicon[0], str):
			self.symbols = list(set(''.join(self.lexicon)))
		else:
			raise ValueError('Each word should be string or tuple')
		self.alphabet_size = len(self.symbols)

		if alpha < 1 / self.alphabet_size or alpha >= 1:
			raise ValueError('alpha must be >= 1/|S| and < 1')
		if beta <= 0:
			raise ValueError('beta must be > 0')
		if gamma >= 1 or gamma <= -1:
			raise ValueError('gamma must be > -1 and < 1')

		chance = 1 / self.alphabet_size
		self.phi = np.full((self.word_length, self.word_length), chance, dtype=float)
		for fixation_position in range(self.word_length):
			for position in range(self.word_length):
				if position > fixation_position:
					self.phi[fixation_position, position] += (alpha-chance) * np.exp( beta * (gamma-1) * abs(fixation_position - position)**2)
				else:
					self.phi[fixation_position, position] += (alpha-chance) * np.exp(-beta * (gamma+1) * abs(fixation_position - position)**2)

		self.p_match = np.log(self.phi)
		self.p_mismatch = np.log((1 - self.phi) / (self.alphabet_size - 1))

	def _choose_alternative_character(self, character):
		'''

		Given a character, return one of the other |S|-1 characters at random.

		'''
		correct_symbol_index = self.symbols.index(character)
		perceived_symbol_index = np.random.randint(self.alphabet_size - 1)
		if perceived_symbol_index >= correct_symbol_index:
			perceived_symbol_index += 1
		return self.symbols[perceived_symbol_index]

	def _create_percept(self, target_word, fixation_position):
		'''

		Create a percept of the target word by adding random noise based on the
		position of the fixation.

		'''
		percept = []
		for position, character in enumerate(target_word):
			if np.random.random() < self.phi[fixation_position, position]:
				percept.append(character)
			else:
				percept.append(self._choose_alternative_character(character))
		return percept

	def _likelihood_percept(self, percept, word, fixation_position):
		'''

		Calculate Pr(p|w,j) - the likelihood of a percept, given that the true
		target was indeed some hypothesized word fixated in some position.
		
		'''
		likelihood = 0.0
		for position in range(self.word_length):
			if percept[position] == word[position]:
				likelihood += self.p_match[fixation_position, position]
			else:
				likelihood += self.p_mismatch[fixation_position, position]
		return likelihood

	def _posterior_given_percept(self, percept, fixation_position):
		'''

		Calculate the posterior probability of each word given a percept fixated in
		a particular position.

		'''
		likelihood = np.zeros(self.lexicon_size, dtype=float)
		for w, word in enumerate(self.lexicon):
			likelihood[w] = self._likelihood_percept(percept, word, fixation_position)
		posterior = likelihood + self.prior
		return posterior - logsumexp(posterior)

	def read(self, target_word, fixation_position, return_index=False, verbose=False):
		'''

		Read a target word at a fixation position. If return_index is True, the
		index of the inferred word is returned rather than the word itself.

		'''
		percept = self._create_percept(target_word, fixation_position)
		posterior = self._posterior_given_percept(percept, fixation_position)
		inferred_word = log_roulette_wheel(posterior)
		if return_index:
			return inferred_word
		if verbose:
			print(f'   Target: {target_word}')
			print(' '*(11+fixation_position) + '^')
			print(f'  Percept: {"".join(percept)}')
			print(f'Inference: {self.lexicon[inferred_word]}')
		return self.lexicon[inferred_word]

	def read_item(self, target_word, fixation_position, return_index=True):
		'''

		Same as the read() method, except this takes the index of one of the
		reader's lexical items rather than an item itself. If return_index is True,
		the index of the inferred word is returned rather than the word itself.
		
		'''
		target_word = self.lexicon[target_word]
		return self.read(target_word, fixation_position, return_index)

	def test(self):
		'''

		Test the reader on each word in each fixation position and return the
		reader's responses.

		'''
		responses = []
		for target_word in range(self.lexicon_size):
			for fixation_position in range(self.word_length):
				inferred_word = self.read_item(target_word, fixation_position)
				responses.append((target_word, fixation_position, inferred_word))
		return responses

	def p_word_given_target(self, target_word, fixation_position, n_sims=10000):
		'''

		Calculate the distribution Pr(w|t,j) – the probability of the reader
		inferring some word given some target in some fixation position. A larger
		number of simulations will produce a more accurate estimate of the
		distribution. If n_sims is set to 0, an exact calculation is performed by
		checking all possible percepts (this is intractable for even moderate word
		lengths and alphabet sizes).

		'''
		target_word = self.lexicon[target_word]
		p_word_given_target = np.zeros(self.lexicon_size, dtype=float)

		if n_sims == 0:
			for percept in product(self.symbols, repeat=self.word_length):
				likelihood_percept = self._likelihood_percept(percept, target_word, fixation_position)
				posterior_given_percept = self._posterior_given_percept(percept, fixation_position)
				p_word_given_target += np.exp(likelihood_percept + posterior_given_percept)
			return p_word_given_target

		for _ in range(n_sims):
			percept = self._create_percept(target_word, fixation_position)
			posterior_given_percept = self._posterior_given_percept(percept, fixation_position)
			p_word_given_target += np.exp(posterior_given_percept)
		return p_word_given_target / n_sims

	def uncertainty(self, target_word, fixation_position, n_sims=10000):
		'''

		Calculate the uncertainty (expected entropy of the posterior) experienced by
		the reader when attempting to identify some target in some fixation
		position. A larger number of simulations will produce a more accurate
		estimate of uncertainty. If n_sims is set to 0, an exact calculation is
		performed by checking all possible percepts (this is intractable for even
		moderate word lengths and alphabet sizes).

		'''
		target_word = self.lexicon[target_word]
		uncertainty = 0

		if n_sims == 0:
			for percept in product(self.symbols, repeat=self.word_length):
				likelihood_percept = np.exp(self._likelihood_percept(percept, target_word, fixation_position))
				posterior_given_percept = np.exp(self._posterior_given_percept(percept, fixation_position))
				uncertainty += likelihood_percept * entropy(posterior_given_percept)
			return uncertainty

		for _ in range(n_sims):
			percept = self._create_percept(target_word, fixation_position)
			posterior_given_percept = np.exp(self._posterior_given_percept(percept, fixation_position))
			uncertainty += entropy(posterior_given_percept)
		return uncertainty / n_sims


def entropy(distribution):
	'''

	Calculate the entropy of a probability distribution.

	'''
	summation = 0
	for p in distribution:
		if p > 0:
			summation += p * np.log2(p)
	return -summation


def logaddexp(val1, val2):
	'''

	Add two probabilities that are in the log domain.

	'''
	mx = max(val1, val2)
	return np.log(np.exp(val1 - mx) + np.exp(val2 - mx)) + mx


def logsumexp(array):
	'''

	Sum an array of probabilities that are in the log domain.

	'''
	mx = array.max()
	return np.log(np.sum(np.exp(array - mx))) + mx


def log_roulette_wheel(distribution):
	'''

	Sample an index from a probability distribution that is in the log domain.
	
	'''
	random_prob = np.log(np.random.random())
	summation = distribution[0]
	for i in range(1, len(distribution)):
		if random_prob < summation:
			return i - 1
		summation = logaddexp(summation, distribution[i])
	return i
