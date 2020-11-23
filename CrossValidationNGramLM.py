import json
import math
from nltk import ngrams
from nltk.book import *
import numpy as np

class CrossValidationNGramLM():
	def __init__(self, data_paths, N = 2):
		self.N = N
		self.NGramFreqTrain = FreqDist([])
		self.NGramFreqValid = FreqDist([])
		self.Nm1GramFreq = FreqDist([])
		self.Nm1GramFreq[('<unk>',)] = 0

		instances_num0 = 0
		instances_num1 = 0
		for i, data_path in enumerate(data_paths):
			fin = open(data_path, 'r')
			sents = [line.strip().split() for line in fin]
			fin.close()
			
			for sent in sents:
				s = ['<BOS%d>' % i for i in range(N - 1)] + sent + ['<EOS>']
				if i == 0:
					self.NGramFreqTrain.update(ngrams(s, N))
					instances_num0 += len(list(ngrams(s, N)))
				else:
					self.NGramFreqValid.update(ngrams(s, N))
					instances_num1 += len(list(ngrams(s, N)))
				self.Nm1GramFreq.update(ngrams(s, N - 1))

		self.Nr0 = FreqDist(list(self.NGramFreqTrain.values()))
		self.Nr1 = FreqDist(list(self.NGramFreqValid.values()))
		self.Nr0[0] = len(self.Nm1GramFreq) * len(self.Nm1GramFreq) - len(self.NGramFreqTrain) #len([ng for ng in self.NGramFreqValid if ng not in self.NGramFreqTrain]) + 1
		self.Nr1[0] = len(self.Nm1GramFreq) * len(self.Nm1GramFreq) - len(self.NGramFreqValid) #len([ng for ng in self.NGramFreqTrain if ng not in self.NGramFreqValid]) + 1

		self.Tr01 = {0:0}
		self.Tr10 = {0:0}
		for ng in self.NGramFreqTrain:
			r = self.NGramFreqTrain[ng]
			if not r in self.Tr01:
				self.Tr01[r] = 0
			if ng in self.NGramFreqValid:
				self.Tr01[r] += self.NGramFreqValid[ng]
			else:
				self.Tr10[0] += r
		
		for ng in self.NGramFreqValid:
			r = self.NGramFreqValid[ng]
			if not r in self.Tr10:
				self.Tr10[r] = 0
			if ng in self.NGramFreqTrain:
				self.Tr10[r] += self.NGramFreqTrain[ng]
			else:
				self.Tr01[0] += r

		self.NGramProb = dict()
		for ng in set(self.NGramFreqTrain.keys()) | set(self.NGramFreqValid.keys()):
			if ng in self.NGramFreqTrain:
				r0 = self.NGramFreqTrain[ng]
			else:
				r0 = 0
			if ng in self.NGramFreqValid:
				r1 = self.NGramFreqValid[ng]
			else:
				r1 = 0
			if r0 not in self.Nr0:
				self.Nr0[r0] = 0
				self.Tr01[r0] = 0
			if r1 not in self.Nr1:
				self.Nr1[r1] = 0
				self.Tr10[r1] = 0
			self.NGramProb[ng] = (self.Tr01[r0] + self.Tr10[r1]) / (self.Nr0[r0] * instances_num1 + self.Nr1[r1] * instances_num0)
		self.instances_num1 = instances_num1
		self.instances_num0 = instances_num0

		self.Nm1GramProb = dict()


	def getNGramProb(self, ng):
		if ng in self.NGramProb:
			return self.NGramProb[ng]
		else:
			return (self.Tr01[0] + self.Tr10[0]) / (self.Nr0[0] * self.instances_num1 + self.Nr1[0] * self.instances_num0)
		# prob = sorted(self.NGramProb.items(), keys = lambda x:x[1], reverse = True)
		# return prob

	def getNm1GramProb(self, nm1g):
		if nm1g in self.Nm1GramProb:
			return self.Nm1GramProb[nm1g]
		ret = sum([self.getNGramProb(tuple(list(nm1g) + [word[0]])) for word in self.Nm1GramFreq])
		self.Nm1GramProb[nm1g] = ret
		return ret


	def ProbPPL(self, sent):
		s = ['<BOS%d>' % i for i in range(self.N - 1)] + sent + ['<EOS>']
		s = [token if (token,) in self.Nm1GramFreq else '<unk>' for token in s]
		ppl = 1
		for ng in ngrams(s, self.N):
			ppl *= math.pow(self.getNGramProb(ng) / self.getNm1GramProb(ng[:-1]), -1 / len(s))

		return ppl


