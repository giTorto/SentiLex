# -*- coding: utf-8 -*-
#!/usr/bin/env python

import numpy as np
import sys, os
import argparse
reload(sys).setdefaultencoding("utf-8")

class LexTagger:
	""" Class to tag text with lexicon """
	def __init__(self, ts=' ', ds='..'):
		''' constructor '''
		self.ts = ts # token separator
		self.ds = ds # discountinuous entry separator

	def findSubArray(self, sub_arr, arr_arr):
		''' Find sub-array in an array '''
		ind_arr = []           # array to store found indices
		sub_len = len(sub_arr) # length of the sub-array
		arr_len = len(arr_arr) # length of the full array

		for i in xrange(arr_len):
			if (i + sub_len) <= arr_len and arr_arr[i:i + sub_len] == sub_arr:
				ind_arr.append(range(i, i + sub_len))
		return ind_arr

	def rmSubsets(self, lst):
		''' check list for overlaps & keep longest element '''
		rml = [] # list of items to remove
		for i in range(len(lst)):
			for j in range(len(lst)):
				if i != j:
					si = set(self.flattenList(lst[i]))
					sj = set(self.flattenList(lst[j]))
					# if i is a subset of j, add it to rml
					if si <= sj:
						rml.append(lst[i])
		return [e for e in lst if e not in rml]

	def flattenList(self, lst):
		''' flatten multidimensional list to 1D '''
		return list(np.array(lst).flat)

	def readLex(self, lfile):
		''' read lexicon from file (single column) '''
		lines = [line.strip() for line in lfile]
		lex = []
		for line in lines:
			de = line.split(self.ds)
			le = [d.split(self.ts) for d in de]
			lex.append(le)
		return lex

	def tagDoc(self, doc, lex):
		''' meta function for tagList to loop over list of lists '''
		out = []
		for seg in doc:
			out.append(self.tagList(doc, lex))
		return out

	def tagList(self, lst, lex):
		''' tag list with lexicon, keeping longest matches '''
		matches = []
		for e in lex:
			pm = [] # matches for parts
			for p in e:
				inds = self.findSubArray(p, lst)
				if inds:
					pm += inds
			# check if all parts are present & add to matches
			if len(pm) == len(e):
				matches.append(pm)
			elif len(e) == 1 and len(pm) != len(e): # risky!
				for m in pm:
					matches.append([m])
		if matches:
			matches = self.rmSubsets(matches)
		return matches

	def genString(self, e):
		''' generate string from lexicon entry list '''
		parts = [self.ts.join(x) for x in e]
		whole = self.ds.join(parts)
		return whole

	def getTokens(self, ind, lst):
		''' get tokens for indices from a list of tokens '''
		out = []
		for p in ind:
			pout = []
			for e in p:
				pout.append(lst[e])
			out.append(pout)
		return out

#----------------------------------------------------------------------#
if __name__ == "__main__":

	argpar = argparse.ArgumentParser(description='Lexicon Tagger')
	argpar.add_argument('-l', '--lfile', type=file)
	argpar.add_argument('-d', '--dsep', type=str, default='..')
	argpar.add_argument('-t', '--tsep', type=str, default=' ')
	args = argpar.parse_args()

	tagger = LexTagger(args.tsep, args.dsep)
	lex = tagger.readLex(args.lfile)
	doc_str = 'as if you either go home or shut up'
	doc_lst = doc_str.split()
	matches = tagger.tagList(doc_lst, lex)
	print matches
	for m in matches:
		toks = tagger.getTokens(m, doc_lst)
		print toks
		mstr = tagger.genString(toks)
		print mstr

