
'''
In this script we will perform the cross-linguistic analyses.
'''
##############################################################################
import ovp
##############################################################################


'''
First, we need to preprocess the Subtlex data, which is a bit messy and
inconsistent. The raw Subtlex data is stored in data/subtlex/ but is not
committed to the public git repo, so if you need to reproduce this process,
you will first need to obtian the relevant Subtlex data files.

From the raw data files we pull out the counts of 5-9 letter words. We will
only consider words that are composed of native characters(as defined for
each language below). All words will be lowercased and any accents will be
stripped. For example, for Dutch, the Subtlex file is encoded in UTF-8, it
uses a tab separator, the relevent headers are "Word" and "FREQcount", a
valid wordform is a sequence of 5-9 letters from A-Z, and no accents need to
be stripped. Swahili is processed in the same way, except that the counts are
pulled from a continuous text file rather than a file of counts. 

Once the counts have been extracted, the separate_and_reduce function
partitions the counts by word length, reduces each "lexicon" to 3000 words
(+ words that have the same freq as the 3000th word), and converts the freqs
to probabilities. Finally, the probabilities and written to JSON.
'''
##############################################################################
from ovp import corpus_freqs

print('DUTCH')
nl_freqs = corpus_freqs.count_subtlex(
	ovp.DATA/'subtlex'/'SUBTLEX-NL.cd-above2.txt',
	encoding = 'utf-8',
	separator = '\t',
	word_header = 'Word',
	freq_header = 'FREQcount',
	wordform = r'[a-z]{5,9}',
	accents = {},
)
nl_probs = corpus_freqs.separate_and_reduce(nl_freqs, target_lexicon_size=3000)
ovp.json_write(nl_probs, ovp.DATA/'lang_word_probs'/'nl.json', compress=True)
quit()

print('ENGLISH')
en_freqs = corpus_freqs.count_subtlex(
	ovp.DATA/'subtlex'/'SUBTLEXus74286wordstextversion.txt',
	encoding = 'ascii',
	separator = '\t',
	word_header = 'Word',
	freq_header = 'FREQcount',
	wordform = r'[a-z]{5,9}',
	accents = {},
)
en_probs = corpus_freqs.separate_and_reduce(en_freqs, target_lexicon_size=3000)
ovp.json_write(en_probs, ovp.DATA/'lang_word_probs'/'en.json', compress=True)

print('GERMAN')
de_freqs = corpus_freqs.count_subtlex(
	ovp.DATA/'subtlex'/'SUBTLEX-DE_cleaned_with_Google00.txt',
	encoding = 'latin1',
	separator = '\t',
	word_header = 'Word',
	freq_header = 'WFfreqcount',
	wordform = r'[a-zßäöü]{5,9}',
	accents = {'ä':'a', 'ö':'o', 'ü':'u'},
)
de_probs = corpus_freqs.separate_and_reduce(de_freqs, target_lexicon_size=3000)
ovp.json_write(de_probs, ovp.DATA/'lang_word_probs'/'de.json', compress=True)

print('GREEK')
gr_freqs = corpus_freqs.count_subtlex(
	ovp.DATA/'subtlex'/'SUBTLEX-GR_full.txt',
	encoding = 'utf-8',
	separator = '\t',
	word_header = '"Word"',
	freq_header = '"FREQcount"',
	wordform = r'[αβγδεζηθικλμνξοπρσςτυφχψωάέήίόύώϊϋΐΰ]{5,9}',
	accents = {'ά':'α', 'έ':'ε', 'ή':'η', 'ί':'ι', 'ό':'ο', 'ύ':'υ', 'ώ':'ω', 'ϊ':'ι', 'ϋ':'υ', 'ΐ':'ι', 'ΰ':'υ'},
)
gr_probs = corpus_freqs.separate_and_reduce(gr_freqs, target_lexicon_size=3000)
ovp.json_write(gr_probs, ovp.DATA/'lang_word_probs'/'gr.json', compress=True)

print('ITALIAN')
it_freqs = corpus_freqs.count_subtlex(
	ovp.DATA/'subtlex'/'subtlex-it.csv',
	encoding = 'utf-8',
	separator = ',',
	word_header = '"spelling"',
	freq_header = '"FREQcount"',
	wordform = r'[abcdefghilmnopqrstuvzàéèíìóòúù]{5,9}',
	accents = {'à':'a', 'é':'e', 'è':'e', 'í':'i', 'ì':'i', 'ó':'o', 'ò':'o', 'ú':'u', 'ù':'u'},
)
it_probs = corpus_freqs.separate_and_reduce(it_freqs, target_lexicon_size=3000)
ovp.json_write(it_probs, ovp.DATA/'lang_word_probs'/'it.json', compress=True)

print('POLISH')
pl_freqs = corpus_freqs.count_subtlex(
	ovp.DATA/'subtlex'/'subtlex-pl-cd-3.csv',
	encoding = 'utf-8',
	separator = '\t',
	word_header = 'spelling',
	freq_header = 'freq',
	wordform = r'[abcdefghijklłmnoprstuwyząćęńóśźż]{5,9}',
	accents = {'ą':'a', 'ć':'c', 'ę':'e', 'ń':'n', 'ó':'o', 'ś':'s', 'ź':'z', 'ż':'z'},
)
pl_probs = corpus_freqs.separate_and_reduce(pl_freqs, target_lexicon_size=3000)
ovp.json_write(pl_probs, ovp.DATA/'lang_word_probs'/'pl.json', compress=True)

print('SPANSIH')
es_freqs = corpus_freqs.count_subtlex(
	ovp.DATA/'subtlex'/'SUBTLEX-ESP.tsv',
	encoding = 'utf-8',
	separator = '\t',
	word_header = 'Word',
	freq_header = 'Freq. count',
	wordform = r'[abcdefghijlmnopqrstuvxyzñáéíóúü]{5,9}',
	accents = {'ñ':'n', 'á':'a', 'é':'e', 'í':'i', 'ó':'o', 'ú':'u', 'ü':'u'},
)
es_probs = corpus_freqs.separate_and_reduce(es_freqs, target_lexicon_size=3000)
ovp.json_write(es_probs, ovp.DATA/'lang_word_probs'/'es.json', compress=True)

print('SWAHILI')
sw_freqs = count_corpus(
	ovp.DATA/'corpora'/'sw_helsinki.txt',
	wordform = r'[abcdefghijklmnoprstuvwyz]{5,9}'
)
sw_probs = corpus_freqs.separate_and_reduce(sw_freqs, target_lexicon_size=3000)
ovp.json_write(sw_probs, ovp.DATA/'lang_word_probs'/'sw.json', compress=True)
##############################################################################


'''
'''
##############################################################################
##############################################################################


'''
'''
##############################################################################
##############################################################################


'''
'''
##############################################################################
##############################################################################


'''
'''
##############################################################################
##############################################################################