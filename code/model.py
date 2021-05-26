from itertools import product
import numpy as np
from numba import njit


class Reader:

	'''

	Model reader.

	'''

	def __init__(self, lexicon, alpha=0.8, beta=0.1, gamma=2.0, delta=0.0, epsilon=0.01):
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

		- gamma : A float > 0. The gamma parameter controls the rate at which the
		probability of successful letter identification approaches chance with
		distance from the fixation position.

		- delta : A float > -1 and < 1. The delta parameter controls how much faster
		the probability of successful letter identification drops to the left vs. to
		the right. If delta is 0, the perceptual filter is symmetrical.

		- epsilon : A float > 0 and < 1. The epsilon parameter controls the
		probability that the reader will make a selection error after making an
		inference.
		
		'''
		self.lexicon_size = len(lexicon)
		if isinstance(lexicon, dict):
			# separate lexicon and prior
			lexicon, prior = zip(*lexicon.items())
			self.prior = np.array(prior, dtype=np.float64)
		elif isinstance(lexicon, list):
			# assume uniform prior
			self.prior = np.full(self.lexicon_size, 1 / self.lexicon_size, dtype=np.float64)
		else:
			raise ValueError('lexicon should be of type list or dict')
		
		# check all words are of the same length and type
		self.word_length = len(lexicon[0])
		self.word_type = type(lexicon[0])
		if self.word_type not in [tuple, str]:
			raise ValueError('Words should be represented as strings or tuples')
		for word in lexicon:
			if len(word) != self.word_length or not isinstance(word, self.word_type):
				raise ValueError('All words in the lexicon must be of the same length and type')

		# extract the set of symbols that the words are composed of
		if self.word_type is tuple:
			self.original_symbols = sorted(list(set(sum(lexicon, tuple()))))
		else:
			self.original_symbols = sorted(list(set(''.join(lexicon))))
		self.symbols = list(range(len(self.original_symbols)))
		self.alphabet_size = len(self.symbols)

		# create the internal lexicon – an NxM array of ints
		self.lexicon = np.zeros((self.lexicon_size, self.word_length), dtype=np.uint8)
		for w, word in enumerate(lexicon):
			for i, character in enumerate(word):
				self.lexicon[w, i] = self.original_symbols.index(character)

		# check the parameters are valid
		if alpha < 1 / self.alphabet_size or alpha >= 1:
			raise ValueError('alpha must be >= 1/|S| and < 1')
		if beta <= 0:
			raise ValueError('beta must be > 0')
		if gamma <= 0:
			raise ValueError('gamma must be > 0')
		if delta >= 1 or delta <= -1:
			raise ValueError('delta must be > -1 and < 1')
		if epsilon >= 1 or epsilon <= 0:
			raise ValueError('epsilon must be > 0 and < 1')
		self.epsilon = epsilon

		# generate the perceptual filter and precompute p_match and p_mismatch
		chance = 1 / self.alphabet_size
		self.phi = np.full((self.word_length, self.word_length), chance, dtype=np.float64)
		for fixation_position in range(self.word_length):
			for position in range(self.word_length):
				if position > fixation_position:
					self.phi[fixation_position, position] += (alpha - chance) * np.exp( beta * abs(position - fixation_position)**gamma * (delta - 1))
				else:
					self.phi[fixation_position, position] += (alpha - chance) * np.exp(-beta * abs(position - fixation_position)**gamma * (delta + 1))
		self.p_match = self.phi
		self.p_mismatch = (1 - self.phi) / (self.alphabet_size - 1)

	def _transcribe(self, word):
		'''

		Transcribe from original symbols to internal symbols.

		'''
		return [self.original_symbols.index(character) for character in word]

	def _back_transcribe(self, word):
		'''

		Transcribe from internal symbols to original symbols.

		'''
		if self.word_type is tuple:
			return tuple([self.original_symbols[character] for character in word])
		else:
			return ''.join([self.original_symbols[character] for character in word])

	def _choose_alternative_character(self, character):
		'''

		Given a character, return one of the other |S|-1 characters at random.

		'''
		perceived_symbol = np.random.randint(self.alphabet_size - 1)
		if perceived_symbol >= character:
			perceived_symbol += 1
		return perceived_symbol

	def _create_percept(self, target_word, fixation_position):
		'''

		Create a percept of the target word by adding random noise according to the
		reader's perceptual filter.

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
		likelihood = 1.0
		for position in range(self.word_length):
			if percept[position] == word[position]:
				likelihood *= self.p_match[fixation_position, position]
			else:
				likelihood *= self.p_mismatch[fixation_position, position]
		return likelihood

	def _posterior_given_percept(self, percept, fixation_position):
		'''

		Calculate the posterior probability of each word given a percept fixated in
		a particular position.

		'''
		likelihood = np.zeros(self.lexicon_size, dtype=np.float64)
		for w, word in enumerate(self.lexicon):
			likelihood[w] = self._likelihood_percept(percept, word, fixation_position)
		posterior = likelihood * self.prior
		return posterior / posterior.sum() # normalize

	def _make_mistake(self, inferred_word):
		'''

		Given an inferred word, return one of the other n-1 words at random.

		'''
		selected_word = np.random.randint(self.lexicon_size - 1)
		if selected_word >= inferred_word:
			selected_word += 1
		return selected_word

	def get_word_from_index(self, word_index):
		'''

		Get the word (in original symbols) for a given word index.

		'''
		return self._back_transcribe(self.lexicon[word_index])

	def get_index_from_word(self, word):
		'''

		Get the index for some word (expressed in original symbols).

		'''
		search_word = self._transcribe(word)
		for w, word in enumerate(self.lexicon):
			n_matches = sum([search_char == char for search_char, char in zip(search_word, word)])
			if n_matches == self.word_length:
				return w
		raise ValueError('This word is not in my lexicon')

	def read(self, target_word, fixation_position):
		'''

		Read a target word at a fixation position and show the reader's percept,
		inference, and ultimate selection. This is mostly intended for playing
		around with the model.

		'''
		if len(target_word) != self.word_length:
			raise ValueError(f'I can only read words of length {self.word_length}')
		if fixation_position >= self.word_length:
			raise ValueError('The specified fixation_position is beyond my word length')
		percept = self._create_percept(self._transcribe(target_word), fixation_position)
		posterior = self._posterior_given_percept(percept, fixation_position)
		inferred_word = roulette_wheel(posterior)
		if np.random.random() < self.epsilon:
			selected_word = self._make_mistake(inferred_word)
		else:
			selected_word = inferred_word
		print(f'   Target (t): {target_word}')
		if self.word_type is tuple:
			print(f' Fixation (j): {fixation_position}')
		else:
			print(' Fixation (j): ' + ' '*fixation_position + '^')
		print(f'  Percept (p): {self._back_transcribe(percept)}')
		print(f'Inference (w): {self._back_transcribe(self.lexicon[inferred_word])}')
		print(f'Selection (o): {self._back_transcribe(self.lexicon[selected_word])}')

	def test(self):
		'''

		Test the reader on each word in each fixation position and return the
		reader's responses as a dataset. These responses may include selection
		errors, as determined by epsilon. This is mostly useful for generating
		synthetic datasets.

		'''
		responses = []
		for target_word in range(self.lexicon_size):
			for fixation_position in range(self.word_length):
				percept = self._create_percept(self.lexicon[target_word], fixation_position)
				posterior = self._posterior_given_percept(percept, fixation_position)
				inferred_word = roulette_wheel(posterior)
				if np.random.random() < self.epsilon:
					selected_word = self._make_mistake(inferred_word)
				else:
					selected_word = inferred_word
				responses.append((target_word, fixation_position, selected_word))
		return responses

	def uncertainty(self, fixation_position, method='standard', n_sims=1000):
		'''

		Calculate the uncertainty (expected entropy of the posterior) experienced by
		the reader when fixating in a particular fixation position. There are
		three methods:

		exhaustive: Perform the calculation using all possible percepts. This gives
		an exact deterministic result, but it's intractable for even moderate word
		lengths and alphabet sizes. This is mainly useful for small test cases.

		standard: Perform the calculation by simulating some number of reading
		events as specified by n_sims.

		fast: Same as standard, except the computation is performed using
		JIT-compiled code. This is much faster, but the code is less readable.

		'''
		if fixation_position >= self.word_length:
			raise ValueError('The specified fixation_position is beyond my word length')

		if method == 'exhaustive':
			uncertainty = 0
			for target_word, p_target in zip(self.lexicon, self.prior):
				for percept in product(self.symbols, repeat=self.word_length):
					likelihood_percept = self._likelihood_percept(percept, target_word, fixation_position)
					posterior_given_percept = self._posterior_given_percept(percept, fixation_position)
					uncertainty += p_target * likelihood_percept * entropy(posterior_given_percept)
			return uncertainty

		if method == 'standard':
			uncertainty = 0
			for target_word, p_target in zip(self.lexicon, self.prior):
				for _ in range(n_sims):
					percept = self._create_percept(target_word, fixation_position)
					posterior_given_percept = self._posterior_given_percept(percept, fixation_position)
					uncertainty += p_target * entropy(posterior_given_percept)
			return uncertainty / n_sims

		if method == 'fast':
			return jitted_uncertainty(self.lexicon, self.prior, self.phi, fixation_position, n_sims)

		raise ValueError('method should be exhaustive, standard, or fast.')

	def p_word_given_target(self, target_word, fixation_position, method='standard', n_sims=1000):
		'''

		Calculate the distribution Pr(w|t,j) – the probability of the reader
		inferring each word given some target in some fixation position. Note that
		this does not incorporate the probability of a selection error, as
		determined by epsilon. There are three methods:

		exhaustive: Perform the calculation using all possible percepts. This gives
		an exact deterministic result, but it's intractable for even moderate word
		lengths and alphabet sizes. This is mainly useful for small test cases.

		standard: Perform the calculation by simulating some number of reading
		events as specified by n_sims.

		fast: Same as standard, except the computation is performed using
		JIT-compiled code. This is much faster, but the code is less readable.

		'''
		if not isinstance(target_word, int) or target_word >= self.lexicon_size:
			raise ValueError('The target word should be specified as an index')
		if fixation_position >= self.word_length:
			raise ValueError('The specified fixation_position is beyond my word length')

		target_word = self.lexicon[target_word]

		if method == 'exhaustive':
			p_word_given_target = np.zeros(self.lexicon_size, dtype=np.float64)
			for percept in product(self.symbols, repeat=self.word_length):
				likelihood_percept = self._likelihood_percept(percept, target_word, fixation_position)
				posterior_given_percept = self._posterior_given_percept(percept, fixation_position)
				p_word_given_target += likelihood_percept * posterior_given_percept
			return p_word_given_target

		if method == 'standard':
			p_word_given_target = np.zeros(self.lexicon_size, dtype=np.float64)
			for _ in range(n_sims):
				percept = self._create_percept(target_word, fixation_position)
				posterior_given_percept = self._posterior_given_percept(percept, fixation_position)
				p_word_given_target += posterior_given_percept
			return p_word_given_target / n_sims

		if method == 'fast':
			return jitted_p_word_given_target(self.lexicon, self.prior, self.phi, target_word, fixation_position, n_sims)

		raise ValueError('method should be exhaustive, standard, or fast.')


def entropy(distribution):
	'''

	Calculate the entropy of a probability distribution.

	'''
	summation = 0
	for p in distribution:
		if p > 0:
			summation += p * np.log2(p)
	return -summation


def roulette_wheel(distribution):
	'''

	Sample an index from a probability distribution.
	
	'''
	random_prob = np.random.random()
	summation = distribution[0]
	for i in range(1, len(distribution)):
		if random_prob < summation:
			return i - 1
		summation += distribution[i]
	return i


@njit(cache=True)
def logsumexp(array):
	'''

	Sum an array in the log domain.

	'''
	array_max = array.max()
	return np.log2(np.sum(np.exp2(array - array_max))) + array_max


@njit(cache=True)
def jitted_p_word_given_target(lexicon, prior, phi,
	target_word, fixation_position, n_sims=1000):
	'''

	This is the jitted version of Reader.p_word_given_target(). All probability
	calculations are done in the log domain.

	'''
	lexicon_size, word_length = lexicon.shape
	alphabet_size_minus_1 = lexicon.max()

	log_prior = np.log2(prior)
	phi_given_fixation = phi[fixation_position]
	log_p_match = np.log2(phi_given_fixation)
	log_p_mismatch = np.log2((1 - phi_given_fixation) / alphabet_size_minus_1)
	
	percept = np.zeros(word_length, dtype=np.uint8)
	log_posteriors = np.zeros((n_sims, lexicon_size), dtype=np.float64)

	for s in range(n_sims):

		for i in range(word_length):
			if np.random.random() < phi_given_fixation[i]:
				percept[i] = target_word[i]
			else:
				perceived_symbol = np.random.randint(alphabet_size_minus_1)
				if perceived_symbol >= target_word[i]:
					perceived_symbol += 1
				percept[i] = perceived_symbol

		for w in range(lexicon_size):
			log_posteriors[s, w] = log_prior[w]
			for i in range(word_length):
				if percept[i] == lexicon[w, i]:
					log_posteriors[s, w] += log_p_match[i]
				else:
					log_posteriors[s, w] += log_p_mismatch[i]
		log_posteriors[s] -= logsumexp(log_posteriors[s])

	log_posterior = np.zeros(lexicon_size, dtype=np.float64)
	for w in range(lexicon_size):
		log_posterior[w] = logsumexp(log_posteriors[:, w])
	return np.exp2(log_posterior - np.log2(n_sims))


@njit(cache=True)
def jitted_uncertainty(lexicon, prior, phi,
	fixation_position, n_sims=1000):
	'''

	This is the jitted version of Reader.uncertainty(). All probability
	calculations are done in the log domain.

	'''
	lexicon_size, word_length = lexicon.shape
	alphabet_size_minus_1 = lexicon.max()

	log_prior = np.log2(prior)
	phi_given_fixation = phi[fixation_position]
	log_p_match = np.log2(phi_given_fixation)
	log_p_mismatch = np.log2((1 - phi_given_fixation) / alphabet_size_minus_1)
	
	log_posterior = np.zeros(lexicon_size, dtype=np.float64)
	percept = np.zeros(word_length, dtype=np.uint8)
	uncertainty = 0.0

	for t in range(lexicon_size):
		target = lexicon[t]
		p_target = prior[t]

		for _ in range(n_sims):

			for i in range(word_length):
				if np.random.random() < phi_given_fixation[i]:
					percept[i] = target[i]
				else:
					perceived_symbol = np.random.randint(alphabet_size_minus_1)
					if perceived_symbol >= target[i]:
						perceived_symbol += 1
					percept[i] = perceived_symbol

			for w in range(lexicon_size):
				log_posterior[w] = log_prior[w]
				for i in range(word_length):
					if percept[i] == lexicon[w, i]:
						log_posterior[w] += log_p_match[i]
					else:
						log_posterior[w] += log_p_mismatch[i]
			log_posterior -= logsumexp(log_posterior)

			entropy = -np.sum(np.exp2(log_posterior) * log_posterior) # drop out of log domain
			uncertainty += p_target * entropy

	return uncertainty / n_sims
