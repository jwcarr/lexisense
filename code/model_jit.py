'''

This is a rewrite of model.py using numba to do just-in-time compilation. This
version of the code is mostly useful for high-efficiency applications, since
it typically runs around 100 times faster. All probability calculations are
done in the log domain. Since numba does not have good support for classes,
the usage pattern is slightly different:

	reader = model_jit.Reader(lexicon, alpha=0.9, beta=0.1, gamma=0)
	model_jit.uncertainty(*reader, 0, n_sims=10000)

'''

import numpy as np
from numba import njit


def Reader(lexicon, alpha, beta, gamma):
	'''

	This is largely equivalent to Reader.__init__() in model.py. The main
	difference is that the lexicon is represented as a numpy array of 8-bit
	unsigned ints (instead of a list of tuples/strings).

	'''
	if isinstance(lexicon, dict):
		words = list(lexicon.keys())
		lexicon_size = len(words)
		prior = np.array([lexicon[word] for word in words], dtype=np.float64)
	elif isinstance(lexicon, list):
		words = lexicon
		lexicon_size = len(words)
		prior = np.array([1 / lexicon_size] * lexicon_size, dtype=np.float64)
	else:
		raise ValueError('lexicon should be of type list or dict')
	
	word_length = len(words[0])
	for word in words:
		if len(word) != word_length:
			raise ValueError('All words in the lexicon must be of the same length')

	if isinstance(words[0], tuple):
		symbols = set(sum(words, tuple()))
		if symbols != set(range(len(symbols))):
			raise ValueError('Invalid tuple words')
		lexicon = np.array(words, dtype=np.uint8)
	elif isinstance(words[0], str):
		symbols = list(set(''.join(words)))
		lexicon = np.zeros((lexicon_size, word_length), dtype=np.uint8)
		for w, word in enumerate(words):
			for i, character in enumerate(word):
				lexicon[w, i] = symbols.index(character)
	else:
		raise ValueError('Each word should be string or tuple')
	alphabet_size = len(symbols)

	if alpha < 1 / alphabet_size or alpha >= 1:
		raise ValueError('alpha must be >= 1/|S| and < 1')
	if beta <= 0:
		raise ValueError('beta must be > 0')
	if gamma >= 1 or gamma <= -1:
		raise ValueError('gamma must be > -1 and < 1')

	chance = 1 / alphabet_size
	phi = np.full((word_length, word_length), chance, dtype=np.float64)
	for fixation_position in range(word_length):
		for position in range(word_length):
			if position > fixation_position:
				phi[fixation_position, position] += (alpha-chance) * np.exp( beta * (gamma-1) * abs(fixation_position - position)**2)
			else:
				phi[fixation_position, position] += (alpha-chance) * np.exp(-beta * (gamma+1) * abs(fixation_position - position)**2)

	return lexicon, prior, phi


@njit
def p_word_given_target(lexicon, prior, phi,
	target, fixation_position, n_sims=10000):
	'''

	This is equivalent to Reader.p_word_given_target() in model.py. The first
	three arguments should be passed in from the Reader() function (see above).

	'''
	lexicon_size = lexicon.shape[0]
	word_length = lexicon.shape[1]
	alphabet_size_minus_1 = lexicon.max()

	log_prior = np.log2(prior)
	phi_given_fixation = phi[fixation_position]
	log_p_match = np.log2(phi_given_fixation)
	log_p_mismatch = np.log2((1 - phi_given_fixation) / alphabet_size_minus_1)
	
	target = lexicon[target]
	percept = np.zeros(word_length, dtype=np.uint8)
	log_posteriors = np.zeros((n_sims, lexicon_size), dtype=np.float64)

	for s in range(n_sims):

		for i in range(word_length):
			if np.random.random() < phi_given_fixation[i]:
				percept[i] = target[i]
			else:
				perceived_symbol = np.random.randint(alphabet_size_minus_1)
				if perceived_symbol >= target[i]:
					perceived_symbol += 1
				percept[i] = perceived_symbol

		for w in range(lexicon_size):
			log_posteriors[s, w] = log_prior[w]
			for i in range(word_length):
				if percept[i] == lexicon[w, i]:
					log_posteriors[s, w] += log_p_match[i]
				else:
					log_posteriors[s, w] += log_p_mismatch[i]
		mx = log_posteriors[s].max()
		log_posteriors[s] -= np.log2(np.sum(np.exp2(log_posteriors[s] - mx))) + mx

	log_posterior = np.zeros(lexicon_size, dtype=np.float64)
	for w in range(lexicon_size):
		mx = log_posteriors[:, w].max()
		log_posterior[w] = np.log2(np.sum(np.exp2(log_posteriors[:, w] - mx))) + mx
	return np.exp2(log_posterior - np.log2(n_sims))


@njit
def uncertainty(lexicon, prior, phi,
	fixation_position, n_sims=10000):
	'''

	This is equivalent to Reader.uncertainty() in model.py. The first three
	arguments should be passed in from the Reader() function (see above).

	'''
	lexicon_size = lexicon.shape[0]
	word_length = lexicon.shape[1]
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
			mx = log_posterior.max()
			log_posterior -= np.log2(np.sum(np.exp2(log_posterior - mx))) + mx

			uncertainty += p_target * -np.sum(np.exp2(log_posterior) * log_posterior)

	return uncertainty / n_sims
