import json
import math
from nltk import ngrams
from nltk.book import *
import numpy as np


class GoodTuringNGramLM():
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

		self.NGramNr = FreqDist(list(self.NGramFreq.values()))
		data = list(self.NGramNr.items())
		self.NCoef = np.polyfit([math.log(d[0]) for d in data], [math.log(d[1]) for d in data], 1)
		self.Nth = math.inf
		for r in range(1, max(self.NGramFreq.values()) - 1):
			rx = (r + 1) * math.exp(self.NCoef[0] + self.NCoef[1] * math.log(r + 1)) / math.exp(self.NCoef[0] + self.NCoef[1] * math.log(r))
			if r in self.NGramNr:
				Nr = self.NGramNr[r]
			else:
				continue
			if r + 1 in self.NGramNr:
				Nr1 = self.NGramNr[r + 1]
			else:
				Nr1 = 0
			if abs(rx - (r + 1) * Nr1 / Nr) < 1.65 * math.sqrt((r + 1) * (r + 1) * Nr1 * (1 + Nr1 / Nr) / (Nr * Nr)):
				self.Nth = r
				break

		self.NGramProb = dict()
		for ng in self.NGramFreq:
			self.NGramProb[ng] = self.getRx(self.NGramFreq[ng]) / instances_num
		self.instances_num = instances_num
		print(self.Nth)

		self.Nm1GramProb = dict()
		
			
	def getRx(self, r):
		if r >= self.Nth:
			return (r + 1) * math.exp(self.NCoef[0] + self.NCoef[1] * math.log(r + 1)) / math.exp(self.NCoef[0] + self.NCoef[1] * math.log(r))
		else:
			return (r + 1) * self.NGramNr[r + 1] / self.NGramNr[r]


	def getNGramProb(self, ng):
		if ng in self.NGramProb:
			return self.NGramProb[ng]
		else:
			return self.NGramNr[1] / self.instances_num / (len(self.Nm1GramFreq) * len(self.Nm1GramFreq) - len(self.NGramFreq))
		
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

