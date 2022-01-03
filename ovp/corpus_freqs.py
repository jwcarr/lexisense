'''
Process the original Subtlex files for seven languages and write
probabilities out to JSON files.
'''

from collections import defaultdict
import json
import re


def count_subtlex(subtlex_file, encoding, separator, word_header, freq_header, wordform, accents):
	'''
	Process raw Subtlex file and extract and count frequencies.
	'''
	wordform = re.compile(wordform)
	freqs = defaultdict(int)
	with open(subtlex_file, encoding=encoding) as file:
		for i, line in enumerate(file):
			split_line = line.strip().split(separator)
			if i == 0: # Header row, so establish the column index of the required columns
				word_index = split_line.index(word_header)
				freq_index = split_line.index(freq_header)
				continue
			word = split_line[word_index].replace('"', '').lower()
			for accented_char, char in accents.items():
				word = word.replace(accented_char, char)
			if wordform.fullmatch(word):
				freqs[word] += int(split_line[freq_index])
	return freqs


def count_corpus(corpus_file, wordform):
	'''
	Count word frequencies in some text corpus.
	'''
	wordform = re.compile(wordform)
	freqs = defaultdict(int)
	with open(corpus_file) as file:
		corpus_text = file.read()
	for word in corpus_text.split('_'):
		if wordform.fullmatch(word):
			freqs[word] += 1
	return freqs


def reduce_lexicon(freqs, target_lexicon_size):
	'''
	Reduce a dictionary of frequencies to a target lexicon size. Since in
	the tail of the distribution many words have identical frequency, we
	first find the minimum frequency required to produce a lexicon of at
	least the desired size and then extract words that have that
	frequency or greater.
	'''
	if len(freqs) <= target_lexicon_size:
		return freqs
	min_freq = sorted(freqs.values(), reverse=True)[target_lexicon_size]
	freqs = {word:freq for word, freq in freqs.items() if freq >= min_freq}
	return defaultdict(int, freqs), min_freq


def separate_words_by_length(freqs):
	freqs_by_length = defaultdict(dict)
	for word, freq in freqs.items():
		freqs_by_length[len(word)][word] = freq
	return freqs_by_length


def separate_and_reduce(freqs, target_lexicon_size=3000):
	freqs_by_length = separate_words_by_length(freqs)
	probs_by_length = defaultdict(dict)
	for length, freqs in freqs_by_length.items():
		reduced_freqs, min_freq = reduce_lexicon(freqs, target_lexicon_size)
		print(length, min_freq, len(reduced_freqs), sum(reduced_freqs.values()))
		total_freq = sum(reduced_freqs.values())
		probs_by_length[length] = {word: freq/total_freq for word, freq in reduced_freqs.items()}
	return probs_by_length
