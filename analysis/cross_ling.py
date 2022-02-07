'''
In this script we will perform the cross-linguistic analyses. Import the ovp
package and define the language codes/names.
'''
##############################################################################
import ovp

languages = {
	'nl': 'Dutch',
	'en': 'English',
	'de': 'German',
	'gr': 'Greek',
	'it': 'Italian',
	'pl': 'Polish',
	'es': 'Spanish',
	'sw': 'Swahili',
}
##############################################################################


'''
First, we need to preprocess the Subtlex data, which is a bit messy and
inconsistent. The raw Subtlex data is stored in data/subtlex/ but is not
committed to the public git repo, so if you need to reproduce this process,
you will first need to obtian the relevant Subtlex data files. However, the
word probabilities computed below are included in the public repo under
data/land_word_probs/, so it should not be necessary to run this block of
code, unless you need to reproduce the word probability files.

From the raw Subtlex files we pull out the counts of 5-9 letter words. We will
only consider words that are composed of native characters(as defined for
each language below). All words will be lowercased and any accents will be
stripped. For example, for Dutch, the Subtlex file is encoded in UTF-8, it
uses a tab separator, the relevant headers are "Word" and "FREQcount", a
valid wordform is a sequence of 5-9 letters from A-Z, and no accents need to
be stripped. Swahili is processed in the same way, except that the counts are
pulled from a continuous text file rather than a file of counts. 

Once the counts have been extracted, the separate_and_reduce function
partitions the counts by word length (i.e. into "lexicons"), reduces each
lexicon to 3000 words (+ words that have the same freq as the 3000th word),
and converts the freqs to probabilities. Finally, the probabilities are
written to JSON files under data/lang_word_probs/
'''
##############################################################################
# from ovp import corpus_freqs

# print('DUTCH')
# nl_freqs = corpus_freqs.count_subtlex(
# 	ovp.DATA/'subtlex'/'SUBTLEX-NL.cd-above2.txt',
# 	encoding = 'utf-8',
# 	separator = '\t',
# 	word_header = 'Word',
# 	freq_header = 'FREQcount',
# 	wordform = r'[a-z]{5,9}',
# 	accents = {},
# )
# nl_probs = corpus_freqs.separate_and_reduce(nl_freqs, target_lexicon_size=3000)
# ovp.json_write(nl_probs, ovp.DATA/'lang_word_probs'/'nl.json', compress=True)

# print('ENGLISH')
# en_freqs = corpus_freqs.count_subtlex(
# 	ovp.DATA/'subtlex'/'SUBTLEXus74286wordstextversion.txt',
# 	encoding = 'ascii',
# 	separator = '\t',
# 	word_header = 'Word',
# 	freq_header = 'FREQcount',
# 	wordform = r'[a-z]{5,9}',
# 	accents = {},
# )
# en_probs = corpus_freqs.separate_and_reduce(en_freqs, target_lexicon_size=3000)
# ovp.json_write(en_probs, ovp.DATA/'lang_word_probs'/'en.json', compress=True)

# print('GERMAN')
# de_freqs = corpus_freqs.count_subtlex(
# 	ovp.DATA/'subtlex'/'SUBTLEX-DE_cleaned_with_Google00.txt',
# 	encoding = 'latin1',
# 	separator = '\t',
# 	word_header = 'Word',
# 	freq_header = 'WFfreqcount',
# 	wordform = r'[a-zßäöü]{5,9}',
# 	accents = {'ä':'a', 'ö':'o', 'ü':'u'},
# )
# de_probs = corpus_freqs.separate_and_reduce(de_freqs, target_lexicon_size=3000)
# ovp.json_write(de_probs, ovp.DATA/'lang_word_probs'/'de.json', compress=True)

# print('GREEK')
# gr_freqs = corpus_freqs.count_subtlex(
# 	ovp.DATA/'subtlex'/'SUBTLEX-GR_full.txt',
# 	encoding = 'utf-8',
# 	separator = '\t',
# 	word_header = '"Word"',
# 	freq_header = '"FREQcount"',
# 	wordform = r'[αβγδεζηθικλμνξοπρσςτυφχψωάέήίόύώϊϋΐΰ]{5,9}',
# 	accents = {'ά':'α', 'έ':'ε', 'ή':'η', 'ί':'ι', 'ό':'ο', 'ύ':'υ', 'ώ':'ω', 'ϊ':'ι', 'ϋ':'υ', 'ΐ':'ι', 'ΰ':'υ'},
# )
# gr_probs = corpus_freqs.separate_and_reduce(gr_freqs, target_lexicon_size=3000)
# ovp.json_write(gr_probs, ovp.DATA/'lang_word_probs'/'gr.json', compress=True)

# print('ITALIAN')
# it_freqs = corpus_freqs.count_subtlex(
# 	ovp.DATA/'subtlex'/'subtlex-it.csv',
# 	encoding = 'utf-8',
# 	separator = ',',
# 	word_header = '"spelling"',
# 	freq_header = '"FREQcount"',
# 	wordform = r'[abcdefghilmnopqrstuvzàéèíìóòúù]{5,9}',
# 	accents = {'à':'a', 'é':'e', 'è':'e', 'í':'i', 'ì':'i', 'ó':'o', 'ò':'o', 'ú':'u', 'ù':'u'},
# )
# it_probs = corpus_freqs.separate_and_reduce(it_freqs, target_lexicon_size=3000)
# ovp.json_write(it_probs, ovp.DATA/'lang_word_probs'/'it.json', compress=True)

# print('POLISH')
# pl_freqs = corpus_freqs.count_subtlex(
# 	ovp.DATA/'subtlex'/'subtlex-pl-cd-3.csv',
# 	encoding = 'utf-8',
# 	separator = '\t',
# 	word_header = 'spelling',
# 	freq_header = 'freq',
# 	wordform = r'[abcdefghijklłmnoprstuwyząćęńóśźż]{5,9}',
# 	accents = {'ą':'a', 'ć':'c', 'ę':'e', 'ń':'n', 'ó':'o', 'ś':'s', 'ź':'z', 'ż':'z'},
# )
# pl_probs = corpus_freqs.separate_and_reduce(pl_freqs, target_lexicon_size=3000)
# ovp.json_write(pl_probs, ovp.DATA/'lang_word_probs'/'pl.json', compress=True)

# print('SPANSIH')
# es_freqs = corpus_freqs.count_subtlex(
# 	ovp.DATA/'subtlex'/'SUBTLEX-ESP.tsv',
# 	encoding = 'utf-8',
# 	separator = '\t',
# 	word_header = 'Word',
# 	freq_header = 'Freq. count',
# 	wordform = r'[abcdefghijlmnopqrstuvxyzñáéíóúü]{5,9}',
# 	accents = {'ñ':'n', 'á':'a', 'é':'e', 'í':'i', 'ó':'o', 'ú':'u', 'ü':'u'},
# )
# es_probs = corpus_freqs.separate_and_reduce(es_freqs, target_lexicon_size=3000)
# ovp.json_write(es_probs, ovp.DATA/'lang_word_probs'/'es.json', compress=True)

# print('SWAHILI')
# sw_freqs = corpus_freqs.count_corpus(
# 	ovp.DATA/'corpora'/'sw_helsinki.txt',
# 	wordform = r'[abcdefghijklmnoprstuvwyz]{5,9}'
# )
# sw_probs = corpus_freqs.separate_and_reduce(sw_freqs, target_lexicon_size=3000)
# ovp.json_write(sw_probs, ovp.DATA/'lang_word_probs'/'sw.json', compress=True)
##############################################################################


'''
Now that we have word probability data, let's test that the model reader
produces some sensible output. The following code loads the English-7
lexicon, instantiates a reader with a certain perceptual filter, and exposes
the reader to the word "guarded" in central position (position 3 counting
from 0). You can play around with the parameters to explore what happens
under different languages, word lengths, fixation positions, and perceptual
filters.
'''
##############################################################################
# from ovp import model

# en_probs = ovp.json_read(ovp.DATA/'lang_word_probs'/'en.json')
# lexicon = en_probs['7']

# reader = model.Reader(lexicon, alpha=0.9, beta=0.2, gamma=0.0)
# reader.read('guarded', fixation_position=3)
##############################################################################


'''
To provide an example in the paper, here I compute the top ten inferences made
by the model reader when fixating the words "guarded" and "concertn" in
initial, central, and final position. This will take a few minutes to run –
to produce the results faster, turn down n_sims.
'''
##############################################################################
# from ovp import model

# lexicon = ovp.json_read(ovp.DATA/'lang_word_probs'/'en.json')['7']
# reader = model.Reader(lexicon, alpha=0.9, beta=0.2, gamma=0.5)

# for target_word in ['guarded', 'concern']:
# 	target_word_i = reader.get_index_from_word(target_word)

# 	for fixation_position in [0, 3, 6]: # initial, central, and final positions
# 		print(fixation_position)
# 		p_w_given_t = reader.p_word_given_target(target_word_i, fixation_position, n_sims=100000)
# 		top_ten_words = reversed(p_w_given_t.argsort()[-10:])

# 		for inferred_word_i in top_ten_words:
# 			inferred_word = reader.get_word_from_index(inferred_word_i)
# 			percentage = round(p_w_given_t[inferred_word_i] * 100, 1)
# 			print(f'{percentage}%', inferred_word)
##############################################################################


'''
The code below can be used to compute uncertainty for a given lexicon and
fixation position. This takes a few minutes to run, so to produce results
faster turn down n_sims.
'''
##############################################################################
# from ovp import model

# lexicon = ovp.json_read(ovp.DATA/'lang_word_probs'/'en.json')['7']
# reader = model.Reader(lexicon, alpha=0.9, beta=0.2, gamma=0.0)
# uncertainty = reader.uncertainty(fixation_position=3, n_sims=10)
# uncertainty = reader.p_word_given_target(fixation_position=3, n_sims=10)
# print(uncertainty)
##############################################################################
'''
You could run this code for all 40 lexicons and all positions within each
lexicon (280 positions in total), but this will take a long time (~50 hours).
Therefore, we will perform these computations on a cluster. This process has
been performed already and the results are stored in data/lang_uncertainty/,
so it is only necessary to follow these steps if you need to recompute the
uncertainty results for some reason.

First, we need to produce shell scripts to submit to the cluster:

	$ python ovp/unc_compute.py --script

To submit the scripts to the cluster, you will do something like this:

	$ sbatch --array=0-6 -p cluster_name en_7.sh

Finally, merge the results together into JSON files for each language:

	$ python ovp/unc_compute.py --merge

This process was performed with 1000 simulations and for two different
perceptual filters:

	data/lang_uncertainty/gamma0.0/   α=0.9, β=0.2, γ=0.0
	data/lang_uncertainty/gamma0.5/   α=0.9, β=0.2, γ=0.5
'''


'''
Finally, we plot uncertainty by letter position for each of the lexicons using
both the symmetrical and asymmetrical perceptual filters. This uses the
uncertainty estimates that were precomputed above and stored under
data/lang_uncertainty/.
'''
##############################################################################
# from ovp import plots

# file_path = ovp.FIGS/'lang_uncertainty.eps'
# with ovp.Figure(file_path, n_rows=8, n_cols=5, width='double', height=160) as fig:
# 	for i, (lang, lang_name) in enumerate(languages.items()):
# 		uncertainty_symm = ovp.json_read(ovp.DATA/'lang_uncertainty'/'gamma0.0'/f'{lang}.json')
# 		uncertainty_asymm = ovp.json_read(ovp.DATA/'lang_uncertainty'/'gamma0.5'/f'{lang}.json')
# 		for j, length in enumerate(range(5, 10)):
# 			plots.plot_uncertainty(fig[i,j], uncertainty_asymm[str(length)], color='MediumSeaGreen', show_min=True)
# 			plots.plot_uncertainty(fig[i,j], uncertainty_symm[str(length)], color='black', show_min=True)
# 			fig[i,j].set_xlabel(f'{length}-letter words')
# 			fig[i,j].set_ylabel(lang_name)
# 			fig[i,j].set_ylim(0, 4)
##############################################################################
