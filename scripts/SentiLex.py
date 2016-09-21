# -*- coding: utf-8 -*-
# !/usr/bin/env python
from nltk.tokenize.api import TokenizerI

from LexTagger import *
from IoUtilities import *
import string
import nltk
import sys, os
import argparse
from nltk.corpus import wordnet

reload(sys).setdefaultencoding("utf-8")
class DummyTokenizer(TokenizerI):
	"""A tokenizer that divides a string into substrings by splitting according to white space
	"""
	def __init__(self):
		''' constructor '''
		self._string = ' '

	def tokenize(self, s):
		return s.split(self._string)

	def span_tokenize(self, s):
		for span in self.string_span_tokenize(s, self._string):
			yield span

class SentiLex:
	""" Lexicon-based sentiment analysis """

	def __init__(self):
		''' constructor '''

	def scoreText(self, txt):
		''' scores aggregated tags '''
		ind_lst = [key for key in txt]
		scores = []
		icoef = 1
		scoef = 1

		for i in sorted(ind_lst):
			if txt[i]['type'] == 'punct':
				icoef = 1
				scoef = 1
			elif txt[i]['type'] == 'iword':
				icoef = txt[i]['num']
			elif txt[i]['type'] == 'sword':
				scoef = txt[i]['num']
			elif txt[i]['type'] == 'pword':
				iscore = self.intensifyPolarity(txt[i]['num'], icoef)
				sscore = self.shiftPolarity(iscore, scoef)
				scores.append(sscore)
		return sum(scores)

	def getPunctIndex(self, lst):
		''' get punctuation indices '''
		return [i for i, j in enumerate(lst) if j in string.punctuation]

	def mkLex(self, cols):
		''' construct lexicon as dict: 2 columns '''
		lex = {}
		for e in cols:
			lex[e[0]] = float(e[1])
		return lex

	def intensifyPolarity(self, ival, coef):
		''' modify polarity: intensifiers '''
		if type(ival) == float:
			return ival * coef
		else:
			return ival

	def shiftPolarity(self, ival, coef):
		''' change polarity '''
		if type(ival) == float:
			return ival * coef
		elif type(ival) == str:
			if istr.lower() == 'negative':
				return 'positive'
			elif istr.lower() == 'positive':
				return 'negative'
			else:
				return ival
		else:
			return ival  # just keep it as is

	def nom2num(self, istr):
		''' convert nominal polarity to numeric '''
		if istr.lower() == 'negative':
			return float(-1)
		elif istr.lower() == 'positive':
			return float(1)
		elif istr.lower() == 'neutral':
			return float(0)
		else:
			return float(0)

	def num2nom(self, inum):
		''' convert numeric polarity to nominal '''
		if inum > 0:
			return 'positive'
		elif inum < 0:
			return 'negative'
		else:
			return 'neutral'

	def tag_doc(self, tagger, doc, ldict, llist, wtype, out=None, improved=False, multiple_dict={}):
		''' tag document (as list) w.r.t. ldict & llist '''
		if not out:
			out = {}

		if improved:
			doc_tagged = tagger.tagList_new(doc, multiple_dict, ldict)
		else:
			doc_tagged = tagger.tagList(doc, llist)

		doc_tokens = [tagger.genString(tagger.getTokens(i, doc)) for i in doc_tagged]
		doc_scored = [ldict[i] for i in doc_tokens]
		for i in range(len(doc_tagged)):
			ind = tagger.flattenList(doc_tagged[i])
			beg = ind[0]
			out[beg] = {'type': wtype, 'num': doc_scored[i], 'word': doc_tokens[i], 'span': ind}
		return out


class SentiSentenceAnalyser:
	"""
	This class is a wrapper for SentiLex, once initialised it allows to transform a sentence to sentiment
	"""
	def __init__(self, token_sep='+', discountinuous_entry_sep='..', field_separator='\t',
				 shifters_file=None, lexicon_file=None, intens_file=None, tweet_tokenizer=False, newer_version=True,
				 lemmatizer=None):
		"""This function allows to initialise the sentiment analyser

		Args:
			token_sep: the separator that is used to represent continuos combination of tokens token in the sentence,
							e.g. not+just\t1
			discountinuous_entry_sep: the separator used in the polarity,shifter and intensifier file for combination
						of discontinuous words that lead to similar score
			field_separator: the separator used to distinguish in the polarity,shifter and intensifier file between
						the text and the sentiment score
			shifters_file: the file containing shifters
			lexicon_file: the file containing lexicon
			intens_file: the file containing intensifiers
			tweet_tokenizer: if set to false, it will use a dummy tokenizer that splits just by white spaces.
							e.g "How are you?" becomes ["How","are","you?"].
							If the nltk.TweetTokenizer is used instead it will become ["how","are","you","?"]
			newer_version: if this flag is set to true it uses the new version to found longest word combination.
						The new version is faster than the old one.
			lemmatizer: if no lemmatizer is provided, the wordnet lemmatizer is used
		Returns:
			None
		"""
		self.sal = SentiLex()
		self.tag = LexTagger(ts=token_sep, ds=discountinuous_entry_sep)
		self.iou = IoUtilities()
		self.fsep = field_separator
		self.shifters_dict = {}
		self.pol_lex_dict = {}
		self.intens_dict = {}
		self.shifter_list = {}
		self.intens_list = {}
		self.pol_lex_list = {}
		self.shifters_multiple = {}
		self.faster_ver = newer_version
		self.intens_multiple = {}
		self.pol_lex_multiple = {}
		self.lemmatizer = lemmatizer if lemmatizer is not None else nltk.WordNetLemmatizer()
		self.tokenizer =  DummyTokenizer() if not tweet_tokenizer else nltk.TweetTokenizer(preserve_case=False)
		if shifters_file:
			self.load_file(shifters_file, 'shift')

		if lexicon_file:
			self.load_file(lexicon_file, 'lex')

		if intens_file:
			self.load_file(intens_file, 'intens')

	def word_to_multi(self, file_type):
		"""
		Args:
			file_type:
		Returns:
		"""
		elemnt_list = []
		multi_dict = {}
		if file_type == 'shift':
			elemnt_list = self.shifter_list
			multi_dict = self.shifters_multiple
		elif file_type == 'intens':
			elemnt_list = self.intens_list
			multi_dict = self.intens_multiple
		else:
			elemnt_list = self.pol_lex_list
			multi_dict = self.pol_lex_multiple

		for element in elemnt_list:
			if len(element) > 1:
				word = element[0][0]
				if word not in multi_dict:
					multi_dict[word] = []
				multi_dict[word].append([val for sublist in element[1:] for val in sublist])

		for key in multi_dict.keys():
			list_comb = multi_dict.get(key)
			list_comb = sorted(list_comb, key=lambda element: -len(element))
			multi_dict[key] = list_comb
		return

	def load_file(self, filename, file_type='intens'):
		"""
		Args:
			filename:
			file_type string accepted types 'intens', 'lex', 'shift'
		Returns:
		"""
		with open(filename) as in_file:
			scol = self.iou.readColumns(in_file, self.fsep)
			if file_type == 'shift':
				self.shifters_dict = self.sal.mkLex(scol)
				self.shifter_list = self.tag.readLex(self.iou.getColumn(scol, 0))
				# please note that every word is contained a list of one value, for now, no need to fix
			elif file_type =='intens':
				self.intens_dict = self.sal.mkLex(scol)
				self.intens_list = self.tag.readLex(self.iou.getColumn(scol, 0))
			elif file_type == 'lex':
				self.pol_lex_dict = self.sal.mkLex(scol)
				self.pol_lex_list = self.tag.readLex(self.iou.getColumn(scol, 0))
		self.word_to_multi(file_type)

	def sentence_to_sentiment_score(self, sentence, lemmatization=False):
		""" This function transform a given sentence into a sentiment score and an overall sentiment.
		The given sentence is tokenized according to the tokenizer provided in initialization phase or using
		nltk.TweetTokenizer(). Each word is not lemmatize, so if one wants to run it on a lemmatised sentence,
		provided the sentence joined by white-spaces already lemmatized.

		Args:
			sentence: the string containing multiple words.
			lemmatization: if set to true, it lemmatizes each word. Otherwise not. Please note that lemmatization
					slows down the sentiment computation
		Returns
			a score float value and a string representing the overall score
		"""
		if lemmatization:
			tokenized_sentence = self.document_to_lemmatized_clean_vector(sentence, lemmatization=True)
		else:
			tokenized_sentence = [str(x) for x in self.tokenizer.tokenize(sentence)]

		txt = {}  # aggregate of all tags
		puncts = self.sal.getPunctIndex(tokenized_sentence)
		for p in puncts:
			txt[p] = {'type': 'punct'}
		if len(self.shifter_list) > 0:
			txt = self.sal.tag_doc(self.tag, tokenized_sentence, self.shifters_dict, self.shifter_list, 'sword', txt,
								   improved=self.faster_ver, multiple_dict=self.shifters_multiple)
		if len(self.intens_list) > 0:
			txt = self.sal.tag_doc(self.tag, tokenized_sentence, self.intens_dict, self.intens_list, 'iword', txt,
								   improved=self.faster_ver, multiple_dict=self.intens_multiple)
		# will over-write previous tags
		if len(self.pol_lex_list) > 0:
			txt = self.sal.tag_doc(self.tag, tokenized_sentence, self.pol_lex_dict, self.pol_lex_list, 'pword', txt,
								   improved=self.faster_ver, multiple_dict=self.pol_lex_multiple)
		score = self.sal.scoreText(txt)
		return score, self.sal.num2nom(score)

	def document_to_lemmatized_clean_vector(self, text, lemmatization=True):
		"""This function transforms a text into a vector of non stopping, lemmatized words.

		Args:
			text: the sentence/document that must be transformed
			tokenizer: the instance of the class that will split the text into words
			lemmatizer: the instance of the class that lemmatizes each word
			stop_words: the set of words to remove, since they are stopping

		Returns
		a vector of lemmatized words which are not stopping
		"""

		words = self.tokenizer.tokenize(text)
		if lemmatization:
			words = nltk.pos_tag(words)
		lemmatize_vector = []
		for word in words:
			lemma = self.lemmatizing_and_remove_stop_words(word, self.lemmatizer, lemmatization=lemmatization)
			if lemma is not None:
				lemmatize_vector.append(lemma)
		return lemmatize_vector

	def lemmatizing_and_remove_stop_words(self,sample, lemmatizer=None, stop_words_set=None, lemmatization=True):
		"""This function transform a word into a lemmatized word or to None in case it is a stopping word

		Args:
			sample: the word to lemmatize or ignore
		Kwargs:
			lemmatizer: the lemmatizer that will lemmatize each word
			stop_words_set: the set of stop words
			lemmatization:
				true: lemmatization is performed
				false: lemmatization is not performed

		Returns
			the lemmatized word, if it is not a stop word
			the word if lemmatization argument is False
			None if it is a stopping word

		Notes:
			if lemmatizer is not provided, the WordeNetLemmatizer from nltk is used
			if stop_words_set is not provided, a default one from the stop_words package is taken
			Not passing lemmatizer can be inefficient since it is instanced for each word.

		"""
		if stop_words_set is None:
			stop_words_set = set()

		word = sample
		if lemmatization:
			word = sample[0]
			detailed_pos = sample[1]

		if word not in stop_words_set:
			# wordnet don't recognize some pos tags
			if lemmatization:
				# detailed_pos is always referenced after assignment since, it is based on same bool value
				pos = self.get_wordnet_pos(detailed_pos)
				if pos:
					lemma = lemmatizer.lemmatize(word, pos=pos)
				else:
					lemma = lemmatizer.lemmatize(word)
				return lemma

			return word
		else:
			return None

	def get_wordnet_pos(self,treebank_tag):
		if treebank_tag.startswith('J'):
			return wordnet.ADJ
		elif treebank_tag.startswith('V'):
			return wordnet.VERB
		elif treebank_tag.startswith('N'):
			return wordnet.NOUN
		elif treebank_tag.startswith('R'):
			return wordnet.ADV
		else:
			return None


# ----------------------------------------------------------------------#
if __name__ == "__main__":

	argpar = argparse.ArgumentParser(description='Lexicon-based Sentiment Analysis')
	argpar.add_argument('-x', '--xfile', type=file) # document for tagging as sentence per line (lemmas)
	argpar.add_argument('-p', '--pfile', type=file) # polarity lexicon
	argpar.add_argument('-s', '--sfile', type=file) # polarity shifters
	argpar.add_argument('-i', '--ifile', type=file) # polarity intensifiers
	argpar.add_argument('-d', '--dsep', type=str, default='..')
	argpar.add_argument('-t', '--tsep', type=str, default='+')
	argpar.add_argument('-f', '--fsep', type=str, default='\t')
	argpar.add_argument('--tagged',  action='store_true', help='POS-tagged lexicon')

	args = argpar.parse_args()

	tag = LexTagger(args.tsep, args.dsep)
	sal = SentiLex()
	iou = IoUtilities()

	# document per line (?)
	if args.xfile:
		docs = [line.strip().split() for line in args.xfile]
	else:
		tokenizer = DummyTokenizer()
		# demo
		# doc_str = 'bene non perfetto , ma brutto brutto'
		#doc_str = "It 's just a phase , try to distract yourself from cleanliness and keep occupied ; just sounds as though you 're bored and that 's all your mind can focus on , I may be wrong ."
		#doc_str = " I don't know why but these long wekends are always hard."
		doc_str = "Don't be depressed"
		docs = [doc_str.split()]
		docs= [tokenizer.tokenize(doc_str)]
		print doc_str

	sentence_sentiment_analyser = SentiSentenceAnalyser(shifters_file=args.sfile.name, intens_file=args.ifile.name,
														lexicon_file=args.pfile.name, token_sep=args.tsep,
														discountinuous_entry_sep=args.dsep, field_separator=args.fsep,
														newer_version=False, tweet_tokenizer=True)
	for doc in docs:
		print doc
		score,sent = sentence_sentiment_analyser.sentence_to_sentiment_score(" ".join(doc),lemmatization=False)
		print score, sent
	"""
	# shifters
	if args.sfile:
		scol = iou.readColumns(args.sfile, args.fsep)
		sdict = sal.mkLex(scol)
		slist = tag.readLex(iou.getColumn(scol, 0))

	# intensifiers
	if args.ifile:
		icol = iou.readColumns(args.ifile, args.fsep)
		idict = sal.mkLex(icol)
		ilist = tag.readLex(iou.getColumn(icol, 0))

	# polarity
	if args.pfile:
		pcol = iou.readColumns(args.pfile, args.fsep)
		pdict = sal.mkLex(pcol)
		plist = tag.readLex(iou.getColumn(pcol, 0))

	for doc in docs:
		if doc:
			txt = {}  # aggregate of all tags
			puncts = sal.getPunctIndex(doc)
			for p in puncts:
				txt[p] = {'type': 'punct'}

			if args.sfile:
				txt = sal.tag_doc(tag, doc, sdict, slist, 'sword', txt)

			if args.ifile:
				txt = sal.tag_doc(tag, doc, idict, ilist, 'iword', txt)

			# will over-write previous tags
			if args.pfile:
				txt = sal.tag_doc(tag, doc, pdict, plist, 'pword', txt)

			score = sal.scoreText(txt)
			print score, sal.num2nom(score)
	"""