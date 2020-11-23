import json
import math
from nltk import ngrams
from nltk.book import *
import numpy as np
from LaplaceNGramLM import LaplaceNGramLM
from GoodTuringNGramLM import GoodTuringNGramLM
from CrossValidationNGramLM import CrossValidationNGramLM
from scipy.stats import spearmanr
import random
from tqdm import tqdm

dataPaths = ['../corpus-new/s4/train.txt', '../corpus-new/s4/valid.txt', '../corpus-new/s4/test.txt']

N = 2
lplm = LaplaceNGramLM(dataPaths[:2], N)
cvlm = CrossValidationNGramLM(dataPaths[:2], N)
gtlm = GoodTuringNGramLM(dataPaths[:2], N)

allNGrams = set()
for path in dataPaths:
	fin = open(path, 'r')
	for line in fin:
		sent = ['<BOS0>'] + line.strip().split() + ['<EOS>']
		for ng in ngrams(sent, N):
			allNGrams.add(ng)

lpProb = []
cvProb = []
gtProb = []
for ng in allNGrams:
	lpProb.append((ng, lplm.getNGramProb(ng)))
	cvProb.append((ng, cvlm.getNGramProb(ng)))
	gtProb.append((ng, gtlm.getNGramProb(ng)))

lpProb.sort(key = lambda x:x[1], reverse = True)
cvProb.sort(key = lambda x:x[1], reverse = True) # ）<EOS> 0.0017791
gtProb.sort(key = lambda x:x[1], reverse = True)

print(lpProb[:10])
print(cvProb[:10])
print(gtProb[:10])

print('spearman correlation:')
print('laplace\t\tcross validation\t\t', spearmanr([d[1] for d in lpProb], [d[1] for d in cvProb]))
print('laplace\t\tgood turing\t\t', spearmanr([d[1] for d in lpProb], [d[1] for d in gtProb]))
print('good turing\t\tcross validation\t\t', spearmanr([d[1] for d in gtProb], [d[1] for d in cvProb]))


fin = open(dataPaths[-1], 'r')
test = [line.strip().split() for line in fin]
fin.close()
print('==' * 10, 'perplexity for single sentence', '==' * 10)
sents = random.sample(test, 2) + [['上海', '举行', '新冠肺炎', '疫情', '防控', '新闻' , '发布会', '，', '介绍', '上海', '疫情', '防控', '有关', '情况', '。']]
for sent in sents:
	print(sent)
	for name, lm in zip(['laplace', 'cross validation', 'good turing'], [lplm, cvlm, gtlm]):
		ppl = lm.ProbPPL(sent)
		print(name, 'ppl:', ppl, 'prob:', math.pow(ppl, -(len(sent) + 2)))
	print('--' * 20)

print('==' * 25)


print('==' * 10, 'perplexity for the whold test set', '==' * 10)

for name, lm in zip(['laplace', 'cross validation', 'good turing'], [lplm, cvlm, gtlm]):
	ppl = []
	for sent in tqdm(test):
		ppl.append(lm.ProbPPL(sent))
	ppl = sum(ppl)
	print(name, ppl / len(test))

print('==' * 25)

