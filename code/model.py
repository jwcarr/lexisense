from itertools import product
import numpy as np


class Reader:

	'''

	Model reader.

	'''

	def __init__(self, lexicon, alpha, beta, gamma=1):
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
			self.prior = np.array([lexicon[word] for word in self.lexicon])
		elif isinstance(lexicon, list):
			self.lexicon = lexicon
			self.lexicon_size = len(self.lexicon)
			self.prior = np.array([1 / self.lexicon_size] * self.lexicon_size)
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

		self.phi = np.zeros((self.word_length, self.word_length), dtype=float)
		for fixation_position in range(self.word_length):
			for position in range(self.word_length):
				if position < fixation_position:
					self.phi[fixation_position, position] = (
						(alpha * self.alphabet_size - 1) * np.exp(-beta * (gamma+1) * abs(fixation_position - position)) + 1
					) / self.alphabet_size
				else:
					self.phi[fixation_position, position] = (
						(alpha * self.alphabet_size - 1) * np.exp(beta * (gamma-1) * abs(fixation_position - position)) + 1
					) / self.alphabet_size

		self.p_match = self.phi.copy()
		self.p_mismatch = (1 - self.phi) / (self.alphabet_size - 1)

		self.percept_type_cache = {}

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

	def _create_percept_from_type(self, target_word, percept_type):
		'''

		Create a percept of the target word that is of some specified percept type.

		'''
		percept = []
		for success, character in zip(percept_type, target_word):
			if success:
				percept.append(character)
			else:
				percept.append(self._choose_alternative_character(character))
		return percept

	def _likelihood_percept(self, percept, word, fixation_position):
		'''

		Calculate Pr(p|w,j) - the likelihood of a percept, given that the true
		target was indeed some hypothesized word fixated in some position.
		
		'''
		likelihood = 1.0
		for position in range(self.word_length):
			if percept[position] == word[position]:
				likelihood *= self.p_match[fixation_position, position]
			else:
				likelihood *= self.p_mismatch[fixation_position, position]
		return likelihood

	def _likelihood_percept_type(self, percept_type, fixation_position):
		'''

		Calculate Pr(p|w,j) using the percept's type (which is the same value
		regardless of w).
		
		'''
		likelihood = 1.0
		for position, success in enumerate(percept_type):
			if success:
				likelihood *= self.p_match[fixation_position, position]
			else:
				likelihood *= self.p_mismatch[fixation_position, position]
		return likelihood

	def _iter_percept_types(self, fixation_position):
		'''

		Iterate over percept types and their likelihoods. This information is cached
		the first time it's computed.

		'''
		if fixation_position not in self.percept_type_cache:
			percept_type_likelihoods = {}
			for percept_type in product((True, False), repeat=self.word_length):
				n_percepts_of_this_type = (self.alphabet_size - 1) ** (self.word_length - sum(percept_type))
				likelihood_percept = self._likelihood_percept_type(percept_type, fixation_position)
				likelihood_percept_type = likelihood_percept * n_percepts_of_this_type
				percept_type_likelihoods[percept_type] = likelihood_percept_type
			self.percept_type_cache[fixation_position] = percept_type_likelihoods
		for percept_type, likelihood_percept_type in self.percept_type_cache[fixation_position].items():
			yield percept_type, likelihood_percept_type

	def _posterior_given_percept(self, percept, fixation_position):
		'''

		Calculate the posterior probability of each word given a percept fixated in
		a particular position.

		'''
		likelihood = np.zeros(self.lexicon_size, dtype=float)
		for w, word in enumerate(self.lexicon):
			likelihood[w] = self._likelihood_percept(percept, word, fixation_position)
		posterior = likelihood * self.prior
		return posterior / posterior.sum()

	def _sample_from_posterior(self, posterior):
		'''

		Sample an index from a probability distribution.
		
		'''
		random_prob = np.random.random()
		summation = posterior[0]
		for inferred_word in range(1, len(posterior)):
			if random_prob < summation:
				return inferred_word - 1
			summation += posterior[inferred_word]
		return inferred_word

	def read(self, target_word, fixation_position, return_index=False, verbose=False):
		'''

		Read a target word at a fixation position. If return_index is True, the
		index of the inferred word is returned rather than the word itself.

		'''
		percept = self._create_percept(target_word, fixation_position)
		posterior = self._posterior_given_percept(percept, fixation_position)
		inferred_word = self._sample_from_posterior(posterior)
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

	def calculate_posterior(self, target_word, fixation_position):
		'''

		Calculate the distribution Pr(w|t,j) exactly. Since this is intractable for
		even moderately sized lexicons, estimate_posterior() or simulate_posterior()
		should be used instead.

		'''
		target_word = self.lexicon[target_word]
		posterior_given_target = np.zeros(self.lexicon_size, dtype=float)
		for i, percept in enumerate(product(self.symbols, repeat=self.word_length)):
			likelihood_percept = self._likelihood_percept(percept, target_word, fixation_position)
			posterior_given_percept = self._posterior_given_percept(percept, fixation_position)
			posterior_given_target += likelihood_percept * posterior_given_percept
		return posterior_given_target

	def estimate_posterior(self, target_word, fixation_position):
		'''

		Estimate the distribution Pr(w|t,j). Similar to calculate_posterior(), but
		instead of iterating over all possible percepts (|S|^m), we just iterate
		over types of percept (2^m) and weight the posterior by the number of
		percepts there are of each type ((|S|-1)^n_failures). This has a similar
		accuracy to simulate_posterior(), but it's faster.

		'''
		target_word = self.lexicon[target_word]
		posterior_given_target = np.zeros(self.lexicon_size, dtype=float)
		for percept_type, likelihood_percept_type in self._iter_percept_types(fixation_position):
			percept = self._create_percept_from_type(target_word, percept_type)
			posterior_given_percept = self._posterior_given_percept(percept, fixation_position)
			posterior_given_target += likelihood_percept_type * posterior_given_percept
		return posterior_given_target

	def simulate_posterior(self, target_word, fixation_position, n_sims=10000, non_zero=False):
		'''

		Estimate the distribution Pr(w|t,j) by simulating a large number of reading
		trials. If non_zero is set to True, the posterior distribution will not
		contain any zeros, but this will tend to overestimate the posterior of low
		probability words, especially if n_sims is not substantially greater than
		the size of the lexicon.
		
		'''
		if non_zero:
			posterior_over_words = np.ones(self.lexicon_size, dtype=int)
			n_sims -= self.lexicon_size
		else:
			posterior_over_words = np.zeros(self.lexicon_size, dtype=int)
		for _ in range(n_sims - self.lexicon_size):
			inferred_word = self.read_item(target_word, fixation_position)
			posterior_over_words[inferred_word] += 1
		return posterior_over_words / n_sims

	def uncertainty(self, target_word, fixation_position, exact=False):
		'''

		Calculate the uncertainty (entropy) experienced by the reader when
		attempting to identify some target in some fixation position.

		'''
		if exact:
			posterior_over_words = self.calculate_posterior(target_word, fixation_position)
		else:
			posterior_over_words = self.estimate_posterior(target_word, fixation_position)
		return -sum([p * np.log2(p) for p in posterior_over_words if p > 0])
