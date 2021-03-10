'''
Process the original Subtlex files for seven languages and write
probabilities out to pickled dictionaries.
'''

from collections import defaultdict
import re
import core


def count_subtlex(subtlex_file, encoding, separator, word_header, freq_header, wordform):
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
	return defaultdict(int, freqs)


def separate_words_by_length(freqs):
	freqs_by_length = defaultdict(dict)
	for word, freq in freqs.items():
		freqs_by_length[len(word)][word] = freq
	return freqs_by_length


def make_probs_file(probs_file, freqs, target_lexicon_size=4000):
	'''

	Process the raw Subtlex file, count the frequencies, reduce to the
	desired lexicon size, and pickle the dictionary.
	
	'''
	probs_by_length = defaultdict(dict)
	freqs_by_length = separate_words_by_length(freqs)
	for length, freqs in freqs_by_length.items():
		reduced_freqs = reduce_lexicon(freqs, target_lexicon_size)
		total_freq = sum(reduced_freqs.values())
		probs_by_length[length] = {word: freq/total_freq for word, freq in reduced_freqs.items()}
	core.pickle_write(probs_by_length, probs_file)


if __name__ == '__main__':

	# DUTCH
	print('DUTCH')
	make_probs_file(core.DATA / 'typ_word_probs' / 'nl.pkl',
		count_subtlex(core.DATA / 'subtlex' / 'SUBTLEX-NL.cd-above2.txt',
			encoding = 'utf-8',
			separator = '\t',
			word_header = 'Word',
			freq_header = 'FREQcount',
			wordform = r'[a-z]{4,12}'
		)
	)

	# ENGLISH
	print('ENGLISH')
	make_probs_file(core.DATA / 'typ_word_probs' / 'en.pkl',
		count_subtlex(core.DATA / 'subtlex' / 'SUBTLEXus74286wordstextversion.txt',
			encoding = 'ascii',
			separator = '\t',
			word_header = 'Word',
			freq_header = 'FREQcount',
			wordform = r'[a-z]{4,12}'
		)
	)

	# GERMAN
	print('GERMAN')
	make_probs_file(core.DATA / 'typ_word_probs' / 'de.pkl',
		count_subtlex(core.DATA / 'subtlex' / 'SUBTLEX-DE_cleaned_with_Google00.txt',
			encoding = 'latin1',
			separator = '\t',
			word_header = 'Word',
			freq_header = 'WFfreqcount',
			wordform = r'[a-zßäöü]{4,12}'
		)
	)

	# GREEK
	print('GREEK')
	make_probs_file(core.DATA / 'typ_word_probs' / 'gr.pkl',
		count_subtlex(core.DATA / 'subtlex' / 'SUBTLEX-GR_full.txt',
			encoding = 'utf-8',
			separator = '\t',
			word_header = '"Word"',
			freq_header = '"FREQcount"',
			wordform = r'[αβγδεζηθικλμνξοπρσςτυφχψωάέήίόύώϊϋΐΰ]{4,12}'
		)
	)

	# ITALIAN
	print('ITALIAN')
	make_probs_file(core.DATA / 'typ_word_probs' / 'it.pkl',
		count_subtlex(core.DATA / 'subtlex' / 'subtlex-it.csv',
			encoding = 'utf-8',
			separator = ',',
			word_header = '"spelling"',
			freq_header = '"FREQcount"',
			wordform = r'[abcdefghilmnopqrstuvzàéèíìóòúù]{4,12}'
		)
	)

	# POLISH
	print('POLISH')
	make_probs_file(core.DATA / 'typ_word_probs' / 'pl.pkl',
		count_subtlex(core.DATA / 'subtlex' / 'subtlex-pl-cd-3.csv',
			encoding = 'utf-8',
			separator = '\t',
			word_header = 'spelling',
			freq_header = 'freq',
			wordform = r'[abcdefghijklmnoprstuwyząćęłńóśźż]{4,12}'
		)
	)

	# SPANSIH
	print('SPANSIH')
	make_probs_file(core.DATA / 'typ_word_probs' / 'es.pkl',
		count_subtlex(core.DATA / 'subtlex' / 'SUBTLEX-ESP.tsv',
			encoding = 'utf-8',
			separator = '\t',
			word_header = 'Word',
			freq_header = 'Freq. count',
			wordform = r'[abcdefghijlmnopqrstuvxyzñáéíóúü]{4,12}'
		)
	)

	# SWAHILI
	print('SWAHILI')
	make_probs_file(core.DATA / 'typ_word_probs' / 'sw.pkl',
		count_corpus(core.DATA / 'corpora' / 'sw_helsinki.txt',
			wordform = r'[abcdefghijklmnoprstuvwyz]{4,12}'
		)
	)
