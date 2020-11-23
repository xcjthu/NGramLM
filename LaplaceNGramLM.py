import json
import math
from nltk import ngrams
from nltk.book import *
import numpy as np

class LaplaceNGramLM():
	def __init__(self, data_paths, N = 2):
		self.N = N
		self.NGramFreq = FreqDist([])
		self.Nm1GramFreq = FreqDist([])
		self.Nm1GramFreq[('<unk>',)] = 0

		instances_num = 0
		for data_path in data_paths:
			fin = open(data_path, 'r')
			sents = [line.strip().split() for line in fin]
			fin.close()
			
			for sent in sents:
				s = ['<BOS%d>' % i for i in range(N - 1)] + sent + ['<EOS>']

				self.NGramFreq.update(ngrams(s, N))
				self.Nm1GramFreq.update(ngrams(s, N - 1))
				instances_num += len(list(ngrams(s, N)))
				
		self.NGramProb = dict()
		for ng in self.NGramFreq:
			self.NGramProb[ng] = (self.NGramFreq[ng] + 1) / (instances_num + len(self.Nm1GramFreq) * len(self.Nm1GramFreq))
		
		self.NGramCondProb = dict()
		for ng in self.NGramFreq:
			self.NGramCondProb[ng] = (self.NGramFreq[ng] + 1) / (self.Nm1GramFreq[ng[:-1]] + len(self.Nm1GramFreq))

		self.instances_num = instances_num

	def getNGramProb(self, ng):
		if ng in self.NGramProb:
			return self.NGramProb[ng]
		else:
			return 1 / (self.instances_num + len(self.Nm1GramFreq) * len(self.Nm1GramFreq))
		# prob = sorted(self.NGramProb.items(), keys = lambda x:x[1], reverse = True)
		# return prob


	def ProbPPL(self, sent):
		s = ['<BOS%d>' % i for i in range(self.N - 1)] + sent + ['<EOS>']
		s = [token if (token,) in self.Nm1GramFreq else '<unk>' for token in s]
		ppl = 1
		for ng in ngrams(s, self.N):
			if ng in self.NGramCondProb:
				ppl *= math.pow(self.NGramCondProb[ng], -1 / len(s))
			else:
				ppl *= math.pow(1 / (self.Nm1GramFreq[ng[:-1]] + len(self.Nm1GramFreq)), -1 / len(s))

		return ppl



