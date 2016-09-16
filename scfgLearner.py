from __future__ import absolute_import, division, print_function

'''""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Learn Stochastic Context-Free Grammars from input with uncertainties.

Author:
Kyuhwa Lee
Imperial College of Science, Technology and Science


Notations:
NJP: Normalized Joint Probability ( JP(S)^(1/len(S)) )
V: Rule score (V= sigma(NJP(S)) * len(S))

Data structures:
input= {'symbols':[], 'values':[]}
 Input stream with uncertainties.

G= {'NT':[rule1, rule2, ...], ...}
 Grammar object.

DLT(global)= OrderedDict{string:{score, count, parent, terms}}
 Description Length Table.

GNode= class{g, dlt, pri, lik, mdl, bestmdl, gid, worse}
 Grammar node of a search tree, gList.
 bestmdl: best MDL score observed so far in the current branch
 worse: for beam search (worse += 1 if new_mdl > bestmdl)

self.t_stat= {string: {count, prob}}
Statistics of terminal symbols.

self.t_dic= {'a':'A', 'b':'B',...}
 Global terminal list.

Basics of Merging & Substituting:
A. Stolcke, PhD Thesis, UCB, p.93-97


TODO:
- Support terminals of 2 or more characters (currently 25 terminal symbols (a-y) are supported)
- Get rid of global constants

""""""""""""""""""""""""""""""""""""""""""""""""""""""""'''
import math, os, sys, time
from collections import OrderedDict
from string import ascii_uppercase
from copy import deepcopy
from operator import itemgetter
from pathos.multiprocessing import Pool
from multiprocessing import Manager, cpu_count, current_process
import sartparser as sp
import q_common as qc


'''""""""""""""""""""""""""""""""""
 Experiment settings

 TODO: Split this part into a separate config file
""""""""""""""""""""""""""""""""'''
ALGORITHM= 'LEEQ' # 'LEEQ' or 'STOLCKE'
VERBOSE= 0; # min 0, max 2.
BEAMSIZE= 3 # beam search width (1= best-first search). Not recommended over 3.
PRUNE_P= 0.01 # prune a rule if low probability
TERM_P= 0.9 # sensor uncertainty to make robust grammar; prob of terminal 'x' being really 'x'
MAX_NGRAMS= 46 # maximum sub-patterns to consider (warning: too large value can lead to very slow performance)
EXPORT_INPUT= False # export input into *.seq each time for parsing


'''""""""""""""""""""""""""""""""""
 Grammar node object
""""""""""""""""""""""""""""""""'''
class GNode:
	def __init__(self, g,dlt,pri,lik,mdl,bestmdl,gid,worse=0):
		self.g= g
		self.dlt= dlt
		self.pri= pri
		self.lik= lik
		self.mdl= mdl
		self.bestmdl= bestmdl
		self.gid= gid
		self.worse= worse


'''""""""""""""""""""""""""""""""""
 Helper Functions
""""""""""""""""""""""""""""""""'''
def uniquify(l):
	"""
	Remove duplicates: doesn't preserve orders
	"""
	return list(set(l))


'''""""""""""""""""""""""""""""""""
 Main Functions
""""""""""""""""""""""""""""""""'''
class ScfgLearner:
	def conv2NT(self, t):
		"""
		Convert input terminals into corresponding NT's
		"""
		str= ''
		for x in t:
			str += self.t_dic[x]
		return str

	def conv2T(self, nt):
		"""
		Convert preterminals to terminals
		"""
		str= ''
		for x in nt:
			str += self.t_dic_rev[x]
		return str

	def getPrior(self, g):
		"""
		Description length of prior probability P(G)
		Stolcke PhD Thesis,"Bayesian learning of probabilistic language models",Sec 2.5.5
		"""

		# parameter prior (terminal symobls + non-terminal symbols)
		num_symbols= len(self.t_dic.keys()) + len(g.keys())
		dl_theta= -math.log(qc.dirichlet(num_symbols),2)

		# structure prior (Poisson distribution of the grammar length)
		dl_S= 0
		mu= 3
		for s in g:
			for r in g[s]:
				rlen= len(r)+1
				# expected bits for (length prior + each rule length * num_symbols)
				dl_S += -math.log(qc.poisson(mu,rlen),3) + rlen*math.log(num_symbols,2)

		pri= dl_S+dl_theta

		return pri


	def grammar2sartgrammar(self, g, dlt, method=0):
		"""
		Convert to a SARTParser's CFGrammar object to get the likelihood.
		"""

		gs= sp.CFGrammar()

		# Define axiom
		gs.addAxiom('Z')

		# Define non-terminals
		for s in g.keys():
			gs.addNonTerminal(s)
		if method==2:
			skip= ascii_uppercase[len(self.t_dic)] # add a SKIP terminal
			gs.addNonTerminal(skip)

		# Define terminals
		for s in self.t_seq:
			gs.addTerminal(s)

		# Define rules
		# Default method
		if method==0:
			for t in self.t_dic:
				gs.addRule(self.t_dic[t], [t], 1.0)

		# Robust method 1: A -> a|b|c|d, B -> a|b|c|d ...
		elif method==1:
			for nt in self.t_dic_rev:
				for t in self.t_dic:
					if self.t_dic[t]==nt:
						rulescore= TERM_P
					else:
						rulescore= self.term_p_other
					gs.addRule(nt, [t], rulescore)

		# Robust method2: A -> a|SKIP, SKIP -> SKIP SKIP|a|b|c|d ...
		elif method==2:
			pskipself= 0.01
			pskip= (1-pskipself) / len(self.t_dic)
			gs.addRule(skip, [skip, skip], pskipself)
			for t in self.t_dic:
				gs.addRule(skip, t, pskip)
			for nt in self.t_dic_rev:
				gs.addRule(nt, [self.t_dic_rev[t]], TERM_P)
				gs.addRule(nt, skip, 1-TERM_P)

		# Actual grammar body
		for nt in self.getNTlist(g):
			sum= 0.0 # sum of all rule scores belonging to nt
			for r in g[nt]: # for each rule of a non-terminal
				rulescore= 0.0
				for t in dlt[r]['terms']:
					if t not in self.t_stat:
						self.bug('%s key is not in self.t_stat'%t, g, dlt)
					rulescore += self.t_stat[t]['prob'] * self.t_stat[t]['count']
				#rulescore *= len(r)
				sum += rulescore
			for r in g[nt]:
				rulescore= 0.0 # the score of each rule
				for t in dlt[r]['terms']:
					rulescore += self.t_stat[t]['prob'] * self.t_stat[t]['count']
				#rulescore *= len(r)
				if sum==0: # when input probabilities of all terminals are 0
					gs.addRule(nt, list(r), 1.0/len(g[nt]))
				else:
					gs.addRule(nt, list(r), rulescore/sum)
		return gs


	def getLikelihood(self, g, dlt, verbose=VERBOSE):
		"""
		Compute likelihood using Viterbi parsing
		"""

		gs= self.grammar2sartgrammar(g, dlt)
		parser= sp.SParser(gs)

		for i in range(len(self.input_list)):
			fs= open(self.testfile % i)
			for s in fs:
				sw= s.strip()
				if len(sw)==0: continue
				if sw[0] != '#':
					parser.parseLine(sw)

		psc= parser.getViterbiParse().probability.scaled
		return min(self.max_mdl, psc)

	def getMDL(self, g, dlt, verbose=False):
		"""
		MDL score
		"""
		return self.getPrior(g) + self.getLikelihood(g, dlt, verbose)

	def getStringCount(self, g, strings):
		"""
		Count the number of appearances of given string in RHS of g
		"""
		c= 0
		for s in g:
			for r in g[s]:
				c += r.count(strings)
		return c

	def getNTlist(self, g):
		"""
		Returns the list of NT's in g except terminal NT's. e.g. [Z,Y,X..]
		"""
		ntlist=[]
		for x in g.keys():
			if x not in self.t_dic.values():
				ntlist.append(x)
		return sorted(ntlist, reverse=True)

	def ngrams(self, seq, maxw=MAX_NGRAMS):
		"""
		Build n-grams from seq up to n=maxw
		"""
		nglist = []
		inlen = len(seq)
		if maxw > inlen: maxw = inlen

		# n-grams with 1 <= n <= maxw
		for w in range(1, maxw + 1):
			for x in range(inlen - w + 1):
				nt = ''.join(seq[x:x + w])
				if nt not in nglist:
					nglist.append(nt)
		return nglist

	def printMsg(self, minVerbosity, *args):
		if VERBOSE < minVerbosity: return
		for msg in args:
			print(msg, end=' ')
		print()

	def printInput(self, minVerbosity, inp):
		"""
		Print input strings
		"""
		if VERBOSE < minVerbosity: return
		print('\n-- New Input Sequence --')
		for x in range(len(inp['symbols'])):
			print('%s  %0.2f'% (inp['symbols'][x],inp['values'][x]))
		print()

	def printDLT(self, minVerbosity, dlt, msg='Description Length Table'):
		"""
		Print Description Length Table
		"""
		if VERBOSE < minVerbosity: return
		print(' '*5 + '-'*15,msg,'-'*15,'<%s>'% current_process().name)
		print(' SCORE   COUNT PARENT STRING',' '*11,'TERMINALS')
		for s in dlt:
			print('%6d  %6d  %4s  %-18s %-s' % (dlt[s]['score'],\
				dlt[s]['count'], dlt[s]['parent'], s, dlt[s]['terms'][0]), end='')
			for t in range(1,len(dlt[s]['terms'])):
				print( ',%s' % dlt[s]['terms'][t], end='' )
				print( ',%s' % dlt[s]['terms'][t], end='' )
			print()
		print()

	def printTSTAT(self, minVerbosity, msg='Terminal Symbol Statistics'):
		if VERBOSE < minVerbosity: return
		print(' --',msg,'--','<%s>'% current_process().name)
		print('%-5s  %-8s  %s' % ('COUNT','PROB','STRING') )
		for s in self.t_stat:
			print( '%-5d  %0.6f  %s' % (self.t_stat[s]['count'],self.t_stat[s]['prob'],s) )
		print()

	def printMDL(self, minVerbosity, g, dlt):
		if VERBOSE < minVerbosity: return
		pri= self.getPrior(g)
		lik= self.getLikelihood(g, dlt)
		print('DL_prior: %.3f'% pri )
		print('DL_likelihood: %.3f'% lik )
		print('MDL: %.3f'% (pri+lik) )
		print()

	def printGrammar(self, minVerbosity, g, dlt, msg=''):
		"""
		Print grammar with msg
		"""
		if VERBOSE < minVerbosity: return
		print(msg)
		for nt in self.getNTlist(g):
			sum= 0.0 # sum of all rule scores belonging to nt
			sortedRules= [] # grammar with each rule sorted by scores
			for r in range(len(g[nt])): # for each RHS rule of a LHS symbol
				rulescore= 0.0
				for t in dlt[g[nt][r]]['terms']:
					if t not in self.t_stat:
						self.bug('%s key is not in self.t_stat'%t, g, dlt)
					rulescore += self.t_stat[t]['prob'] * self.t_stat[t]['count']
				#rulescore *= len(g[nt][r])
				sortedRules.append((rulescore,r))
				sum += rulescore
			sortedRules= sorted(sortedRules, key=itemgetter(0), reverse=True)
			firstLine=True
			for r in sortedRules:
				rs= g[nt][r[1]]
				if firstLine:
					firstLine=False
					print('%s -> '% nt, end='')
				else:
					print('     ', end='')
				rulescore= 0.0 # the score of each rule
				terms= ''
				for t in dlt[rs]['terms']:
					rulescore += self.t_stat[t]['prob'] * self.t_stat[t]['count']
					terms += t + ' '
				#rulescore *= len(rs)
				if sum== 0:
					print('%-18s [%0.6f] (%0.2f) <%s> %s'%
						(rs, 1.0, rulescore, dlt[rs]['parent'], terms))
				else:
					print('%-18s [%0.6f] (%0.2f) <%s> %s'%
						(rs, rulescore/sum, rulescore, dlt[rs]['parent'], terms))
				if dlt[rs]['parent']== None:
					#self.bug('no parent', g, dlt)
					pass
		print()

	def printGrammarSimple(self, minVerbosity, g, msg=''):
		"""
		Print grammar without rule probabilities
		"""
		if VERBOSE < minVerbosity: return
		print(msg)
		for nt in sorted(g.keys(), reverse=True):
			firstLine=True
			for rs in g[nt]:
				if firstLine:
					firstLine=False
					print('%s '% nt, end='')
				else:
					print('  ', end='')
				print('-> %-18s'% rs)

	def printGrammarMDL(self, minVerbosity, g, dlt, msg=''):
		if VERBOSE < minVerbosity: return
		self.printGrammar(g, dlt, msg)
		self.self.printMDL(minVerbosity, g, dlt)

	def printGrammarList(self, minVerbosity, gList, msg=''):
		if VERBOSE < minVerbosity: return
		print('\n'+'='*60)
		print(' ', msg, 'Showing the best grammars')
		print('='*60)
		for n in range(len(gList)):
			self.printGrammar(minVerbosity, gList[n].g, gList[n].dlt, \
				'#%d, MDL: %.6f, DL_pri: %.6f, DL_lik: %.6f'%\
				(n+1, gList[n].mdl, gList[n].pri, gList[n].lik) )
			#self.printDLT(gList[n].dlt)
			print('-'*60)

	def exportGrammarList(self, gList):
		if not os.path.exists('results'):
			os.mkdir('results')
		for n in range(len(gList)):
			self.exportGrammar(gList[n].g, gList[n].dlt, 'results/rank%d.grm'% (n+1), 2 )

	def exportGrammar(self, g, dlt, filename, METHOD=0):
		"""
		Export grammar
		"""

		f=open(filename,'w')
		f.write('Section Terminals\n')
		for s in self.t_seq: f.write('%s '% s)
		f.write('\n\nSection NonTerminals\n')
		for s in g: f.write('%s '% s)
		if METHOD==2:
			skip= ascii_uppercase[len(self.t_dic)]
			f.write(skip)
		f.write('\n\nSection Axiom\n')
		f.write('Z\n\nSection Productions\n')

		# default
		if METHOD==0:
			for t in self.t_dic:
				f.write('%s: %-20s [%0.6f]\n' % (self.t_dic[t], t, 1.0))
		# Robust Input Method 1: A -> a|b|c|d, B -> a|b|c|d ...
		elif METHOD==1:
			for nt in self.t_dic_rev:
				firstLine= True
				for t in self.t_dic:
					if firstLine:
						firstLine= False
						id='%s:'% nt
					else:
						id=' |'
					if self.t_dic[t]==nt:
						rulescore= TERM_P
					else:
						rulescore= self.term_p_other
					f.write('%s %-20s [%0.6f]\n' % (id, t, rulescore))
		# Robust Input Method2: A -> a|SKIP, SKIP -> SKIP SKIP|a|b|c|d ...
		elif METHOD==2:
			pskipself= 0.01
			pskip= (1-pskipself) / len(self.t_dic)
			f.write('%s: %s %-18s [%0.6f]\n' % (skip, skip, skip, pskipself) )
			for t in self.t_dic:
				f.write(' | %-20s [%0.6f]\n' % (t, pskip) )
			for t in self.t_dic_rev:
				f.write('%s: %-20s [%0.6f]\n' % (t, self.t_dic_rev[t], TERM_P))
				f.write(' | %-20s [%0.6f]\n' % (skip, 1-TERM_P))

		for nt in self.getNTlist(g):
			sum= 0.0 # sum of all rule scores belonging to nt
			for r in g[nt]: # for each rule of a non-terminal
				rulescore= 0.0
				for t in dlt[r]['terms']:
					if t not in self.t_stat:
						self.bug('%s key is not in self.t_stat'%t, g, dlt)
					rulescore += self.t_stat[t]['prob'] * self.t_stat[t]['count']
				#rulescore *= len(r)
				sum += rulescore
			firstLine= True
			for r in g[nt]:
				if firstLine:
					firstLine= False
					id='%s:'% nt
				else:
					id=' |'
				rulescore= 0.0 # the score of each rule
				for t in dlt[r]['terms']:
					rulescore += self.t_stat[t]['prob'] * self.t_stat[t]['count']
				#rulescore *= len(r)
				if sum==0: # when input probabilities of all terminals are 0
					f.write('%s %-20s [%0.6f]\n' % (id, ' '.join(list(r)), 1.0/len(g[nt])))
				else:
					f.write('%s %-20s [%0.6f]\n' % (id, ' '.join(list(r)), rulescore/sum))
		f.close()

	def exportInput(self):
		"""
		Export input data
		"""

		tlist= self.t_seq
		for r in range(len(self.input_list)):
			f=open(self.testfile%r, 'w')
			f.write('#')
			for t in tlist: f.write('%8s '% t)
			f.write('\n#')
			for t in tlist: f.write('%-9s'% ('-'*8) )
			f.write('\n')
			input= self.input_list[r]
			for i in range(len(input['symbols'])):
				p= input['values'][i]
				li= (1-p) / (len(tlist)-1)
				s= self.t_dic_rev[input['symbols'][i]]
				for t in tlist:
					if s==t: f.write(' %0.6f'% p)
					else: f.write(' %0.6f'% li)
				f.write(' # %s'% s)
				f.write('\n')
			f.close()

	def bug(self, msg, g='', dlt=''):
		sys.stderr.write('>> BUG FOUND: %s'% msg)
		print('#'*80)
		print('>> BUG FOUND: %s'% msg)
		print('#'*80)
		if g: self.printGrammarSimple(0, g, '## Raw Grammar Rules')
		if dlt: self.printDLT(0, dlt)
		print('#'*80)
		raise RuntimeError

	def buildInput(self, seq):
		"""
		Build input from raw data
		"""
		assert len(seq) % 2==0, 'You have wrong length of input sequence.'
		input= {'symbols':[], 'values':[]}
		for i in range(0,len(seq),2):
			input['symbols'].append(self.conv2NT(seq[i]))
			input['values'].append(float(seq[i+1]))
		return input

	def getNextNT(self, g):
		"""
		Return the next available NT symbol
		"""
		for s in self.upper:
			if s not in g:
				assert s not in self.t_dic.values(), 'self.getNextNT(): assertion error!'
				return s

	def splitTNT(self, st):
		"""
		Return the list of strings split by NT and chunk of T's. [NT,T,NT,NT,T,NT...]
		"""
		seq= []
		wasNT= True
		for x in st:
			if x not in self.t_dic or wasNT:
				seq += x
			else:
				seq[-1] += x
			wasNT= x not in self.t_dic
		return seq

	def addRule(self, g, nt, string):
		"""
		Add a new rule that belongs to nt
		"""
		if nt not in g:
			g[nt]= [string]
		elif string not in g[nt]:
			g[nt].append(string) # if not identical input
		else:
			#print('>> AddRule(): Ignored adding pre-existing strings',string)
			pass
		return g

	def delRule(self, g, nt):
		"""
		Delete a rule that belongs to nt
		"""
		if nt not in g: self.bug('self.delRule(): NT(%s) is not in G!'% nt)
		del g[nt]
		return g

	def sortRules(self, g):
		"""
		Sort RHS rules in dictionary order
		"""
		for s in self.getNTlist(g): g[s]= sorted(g[s])
		return g

	def sortSymbols(self, g, dlt):
		"""
		Sort LHS symbols in reversed alphabetical order(e.g. W, X -> Z, Y)
		"""
		ntlist= self.getNTlist(g)
		for x in range(len(ntlist)):
			if ntlist[x] != self.upper[x]:
				nt_old= ntlist[x]
				nt_new= self.upper[x]
				self.printMsg(2, '>> Re-ordering %s to %s <%s>'% (nt_old, nt_new, current_process().name) )
				for s in g:
					for x in range(len(g[s])):
						r= g[s][x]
						if nt_old in r:
							if r in dlt and dlt[r]['parent'] != None:
								dlt[r]['parent']= dlt[r]['parent'].replace(nt_old, nt_new)
							g[s][x]= r.replace(nt_old, nt_new)
					if s==nt_old:
						if nt_new in g:  self.bug('self.sortSymbols(): nt_new(%s) already in G!'% nt_new)
						g[nt_new]= g.pop(nt_old)
				for s in dlt:
					if nt_old in s:
						new_s= s.replace(nt_old, nt_new)
						if new_s in dlt:  self.bug('self.sortSymbols(): new_s(%s) already in DLT!'% new_s)
						dlt[new_s]= dlt.pop(s)
		return g, dlt

	def updateDLT(self, g, dlt, maxw=MAX_NGRAMS):
		"""
		Update DLT's count and score to reflect the current grammar
		"""

		# invalidate DLT
		for x in dlt:
			dlt[x]['count']= 0

		# build n-grams
		for s in self.getNTlist(g):
			for r in g[s]:
				nglist= self.ngrams(r, maxw)
				for nt in nglist:
					if nt not in dlt:
						self.bug('self.updateDLT(): %s is not in DLT'% nt, g, dlt)
					dlt[nt]['count'] += r.count(nt)

		# sort DLT according to score
		for nt in dlt:
			n= dlt[nt]['count']
			w= len(nt)
			dlt[nt]['score']= (n-1)*(w-1)-2
			##################################################################################
			# THINK: THIS SHOULD BE RECONSIDERED, ALSO, WHY -1?
			if dlt[nt]['parent'] != None: dlt[nt]['score'] -= 1
			##################################################################################
		dlt= OrderedDict(sorted(dlt.items(), key=lambda t: t[1]['score'], reverse=True))

		return dlt

	def updateTStat(self, input, maxw=MAX_NGRAMS):
		"""
		Update self.t_stat with new input (substrings up to length=maxw)
		"""

		inlen= len(input['symbols'])
		if maxw > inlen: maxw= inlen

		# n-grams with 1 <= n <= maxw
		for w in range(1,maxw+1):
			for x in range(inlen-w+1):
				if self.algorithm=='STOLCKE':
					njp= 0.0
				else:
					njp= 1.0
				term= ''
				for i in range(x,x+w):
					njp *= input['values'][i]
					term += self.t_dic_rev[input['symbols'][i]]
				njp= math.pow(njp,1/w)

				if term not in self.t_stat:
					self.t_stat[term]={'count':1, 'prob':njp}
				else:
					t= self.t_stat[term]
					self.t_stat[term]['prob']= (t['count'] * t['prob']	+ njp ) / (t['count']+1)
					self.t_stat[term]['count'] += 1

	def inputDLT(self, input, dlt, maxw=MAX_NGRAMS):
		"""
		Update DLT with new input (substrings up to length=maxw)
		"""

		inlen= len(input['symbols'])
		if maxw > inlen: maxw= inlen

		# n-grams with 1 <= n <= maxw
		for w in range(1,maxw+1):
			for x in range(inlen-w+1):
				nt= ''.join([z[0] for z in input['symbols'][x:x+w]])
				term= ''
				for i in range(x,x+w):
					term += self.t_dic_rev[input['symbols'][i]]
				if nt not in dlt:
					dlt[nt]={'score':-1,'terms':[term],'count':0,'parent':None}

		return dlt

	def getFirstDLT(self, dlt, strings='', parent=False):
		"""
		Return the highest score string X in DLT with conditions:
		1. parent: {False: X has no parent | True: X has parent}
		2. strings: X is substring of strings; '' if don't care
		"""

		global PRUNE_P
		lastScore= -1 # minimum score
		maxItem= []
		for s in dlt:
			if dlt[s]['score'] < lastScore:  break
			#if parent and parent==(dlt[s]['parent']==None): continue
			if strings != '' and s not in strings:  continue
			termprob= 0.0
			for t in dlt[s]['terms']:
				termprob += self.t_stat[t]['prob']
			if termprob <= PRUNE_P:  continue
			lastScore= dlt[s]['score']
			maxItem.append(s)
		return maxItem

	def getFirstDLThack(self, dlt, strings='', parent=False):
		"""
		Consider only limited-length words
		"""

		global PRUNE_P
		maxItem= []
		for s in dlt:
			if len(s) < 2 or len(s) > 5: continue
			termprob= 0.0
			for t in dlt[s]['terms']:
				termprob += self.t_stat[t]['prob']
			if termprob <= PRUNE_P:  continue
			if dlt[s]['count'] > 1:
				maxItem.append(s)
		return maxItem

	def substituteInput(self, g, r, dlt, strings, new_nt, nt='Z'):
		"""
		Substitute symbols in input and update DLT
		"""

		if VERBOSE:  print( '>> SUBSTITUTE input (%s)=%s'% (strings,new_nt) )
		assert nt in g
		assert nt in dlt
		assert new_nt in g
		assert new_nt in dlt
		oldRule= g[nt][r]
		assert strings in oldRule
		assert strings in dlt
		ref= dlt[oldRule]
		newRule= oldRule.replace(strings, new_nt)
		g[nt][r]= oldRule.replace(strings, '_') # will be changed later; this is important
		if newRule not in dlt:
			dlt[newRule]= {'terms':deepcopy(ref['terms']),
				'score':ref['score'], 'count':ref['count'], 'parent':ref['parent']}
		else:
			dlt[newRule]['parent']= ref['parent']
		dlt[oldRule]['parent']= None # it's broken now

		# build new n-grams including new NT
		for ngram in self.ngrams(g[nt][r]): # for each sub-pattern
			newgram= ngram.replace('_', new_nt)
			if newgram in dlt: continue
			ref= ngram.replace('_', strings)
			#if len(ref) > MAX_NGRAMS: continue
			try:
				dlt[newgram]= deepcopy(dlt[ref])
			except Exception:
				self.bug('in self.substitute(): while trying to copy dlt[%s] to dlt[%s]'%(ref,ngram), g, dlt)
		g[nt][r]= g[nt][r].replace('_', new_nt) # back to normal
		dlt= self.updateDLT(g, dlt)
		return g, dlt

	def substitute(self, g, strings, new_nt, dlt, gid):
		"""
		Substitute symbols
		"""

		#print( '>> SUBSTITUTE(%s)=%s on Grammar #%d'% (strings, new_nt, gid) )
		#assert(new_nt not in g)

		# update existing rules
		for nt in self.getNTlist(g):
			for r in range(len(g[nt])):
				#if nt==dlt[strings]['parent']: continue ############ DON't NEED IT
				oldRule= g[nt][r]
				ref= dlt[oldRule]
				if strings in oldRule:
					newRule= oldRule.replace(strings, new_nt)
					g[nt][r]= oldRule.replace(strings, '_') # will be changed later; important
					if newRule not in dlt:
						dlt[newRule]= {'terms':deepcopy(ref['terms']),
							'score':ref['score'], 'count':ref['count'], 'parent':ref['parent']}
					dlt[oldRule]['parent']= None # it's "broken" now
			g[nt]= uniquify(g[nt])
		g= self.addRule(g, new_nt, strings)

		# update child-parent relationship
		if new_nt not in dlt:
			dlt[new_nt]={'terms':deepcopy(dlt[strings]['terms']),'score':-1,'count':0,'parent':None}
		dlt[strings]['parent']= new_nt

		# build new n-grams including new NT
		for s in self.getNTlist(g): # for each NT in g
			for r in range(len(g[s])): # for each rule in NT
				for ngram in self.ngrams(g[s][r]): # for each sub-pattern
					newgram= ngram.replace('_', new_nt)
					if newgram in dlt: continue
					ref= ngram.replace('_', strings)
					#if len(ref) > MAX_NGRAMS: continue
					try:
						dlt[newgram]= deepcopy(dlt[ref])
					except Exception:
						self.bug('in self.substitute(): while trying to copy dlt[%s] to dlt[%s]'%(ref,ngram), g, dlt)
				g[s][r]= g[s][r].replace('_', new_nt) # back to normal
			g[s]= uniquify(g[s])
		dlt= self.updateDLT(g, dlt)

		return g, dlt

	def mergeSet(self, g):
		"""
		Return NT combinations to merge
		"""

		s= []
		l= self.getNTlist(g)
		for n1 in range(len(l)):
			for n2 in range(n1+1,len(l)):
				s.append( (l[n1],l[n2]) )
		return s

	def merge(self, nt1, nt2, new_nt, g, dlt, gid):
		"""
		Merge operation

		Two special cases of Merge:

		1. need to check self-recursion!
		 Y -> X
		 self.merge(X,Y):
		 Y -> Y should be deleted

		2. need to check for uniqueness!
		 Z1 -> AYB (c1)
		    -> AXB (c2)
		 self.merge(X,Y):
		 Z1 -> AYB (c1)
		    -> AYB (c2)
		 should become:
		 Z1 -> AYB (c1+c2)
		"""

		#print( '>> MERGE(%s,%s)=%s on Grammar #%d'% (nt1, nt2, new_nt, gid) )

		# replace nt1, nt2 to new_nt in both LHS & RHS
		for s in self.getNTlist(g):
			for r in range(len(g[s])):
				if dlt[g[s][r]]['parent']:
					if nt1 in dlt[g[s][r]]['parent']:
						dlt[g[s][r]]['parent']= dlt[g[s][r]]['parent'].replace(nt1, new_nt)
					if nt2 in dlt[g[s][r]]['parent']:
						dlt[g[s][r]]['parent']= dlt[g[s][r]]['parent'].replace(nt2, new_nt)
				g[s][r]= g[s][r].replace(nt1, new_nt)
				g[s][r]= g[s][r].replace(nt2, new_nt)
			g[s]= uniquify(g[s]) # case 2

		# add merged rule
		new_rules= g[nt1] + g[nt2]
		delList= []
		for r in range(len(new_rules)):
			new_rules[r]= new_rules[r].replace(nt1, new_nt)
			new_rules[r]= new_rules[r].replace(nt2, new_nt)
			if new_rules[r]==new_nt:
				delList.append(r) # case 1
		for d in sorted(list(set(delList)), reverse=True):
			del(new_rules[d])
		new_rules= uniquify(new_rules) # case 2
		if len(new_rules)==0:
			self.bug('This actually happened. While merging, no rules are left on merged RHS !')

		# update grammar
		for r in new_rules:
			g= self.addRule(g, new_nt, r)
		g= self.delRule(g, nt1)
		g= self.delRule(g, nt2)

		# update DLT
		for s in dlt:
			if nt1 in s:
				new_string= s.replace(nt1, new_nt)
				if new_string not in dlt:
					dlt[new_string]= dlt.pop(s)
				else:
					dlt[new_string]['terms'].extend(dlt[s]['terms'])
					dlt[new_string]['terms']= uniquify(dlt[new_string]['terms'])
					################ THIS IS HACK #####################################
					# TO AVOID MULTIPLE PARENTS WHICH ACTUALLY DOESN'T AFFECT PROGRAM #
					###################################################################
					if dlt[s]['parent']: dlt[new_string]['parent']= dlt[s]['parent']
					del(dlt[s])
		for s in dlt:
			if nt2 in s:
				new_string= s.replace(nt2, new_nt)
				if new_string not in dlt:
					dlt[new_string]= dlt.pop(s)
				else:
					dlt[new_string]['terms'].extend(dlt[s]['terms'])
					dlt[new_string]['terms']= uniquify(dlt[new_string]['terms'])
					# TO AVOID MULTIPLE PARENTS WHICH ACTUALLY DOESN'T AFFECT THE PERFORMANCE
					if dlt[s]['parent']: dlt[new_string]['parent']= dlt[s]['parent']
					del(dlt[s])

		# THINK: SHOULD DLT UPDATED HERE?
		dlt= self.updateDLT(g, dlt)

		############# THIS IS SLOWER METHOD ########################
		# Instead of adding new_nt, just merge into higher-order NT
		# e.g. MERGE(Z,X)=Z
		############################################################
		if self.upper.index(nt1) < self.upper.index(nt2):
			nt_old= new_nt
			nt_new= nt1
		else:
			nt_old= nt1
			nt_new= new_nt
		for s in self.getNTlist(g):
			for x in range(len(g[s])):
				r= g[s][x]
				if r in dlt and dlt[r]['parent'] != None:
					dlt[r]['parent']= dlt[r]['parent'].replace(nt_old, nt_new)
				g[s][x]= r.replace(nt_old, nt_new)
			if s==nt_old:
				if nt_new in g:  self.bug('self.merge(): nt_new(%s) is already in G!'% nt_new)
				g[nt_new]= g.pop(nt_old)
		for s in dlt:
			if nt_old in s:
				new_s= s.replace(nt_old, nt_new)
				if new_s in dlt:  self.bug('self.merge(): new_s(%s) is already in G!'% new_s)
				dlt[new_s]= dlt.pop(s)
		###########################################################

		# DLT should be updated here!
		dlt= self.updateDLT(g, dlt)
		g, dlt= self.sortSymbols(g, dlt)
		#if self.getLikelihood(g, dlt) < self.max_mdl:  return g, dlt
		#else:  return None, None
		return g, dlt

	def pruneGrammar(self, g, dlt):
		"""
		Prune rules with low probability
		"""

		pruned= []
		for nt in self.getNTlist(g):
			sum= 0.0 # sum of all rule scores belonging to nt
			for r in range(len(g[nt])): # for each RHS rule of a LHS symbol
				rulescore= 0.0
				for t in dlt[g[nt][r]]['terms']:
					if t not in self.t_stat:
						self.bug('%s key is not in self.t_stat'%t, g, dlt)
					rulescore += self.t_stat[t]['prob'] * self.t_stat[t]['count']
				#rulescore *= len(g[nt][r])
				sum += rulescore
			if sum==0: # all rule probs are identical; cannot continue
				return g, dlt, pruned

			# compute rule probs and prune if needed
			for r in g[nt]:
				rulescore= 0.0 # the score of each rule
				for t in dlt[r]['terms']:
					rulescore += self.t_stat[t]['prob'] * self.t_stat[t]['count']
				#rulescore *= len(rs)
				ruleprob= rulescore / sum
				if ruleprob < PRUNE_P:
					if dlt[r]['parent']==nt:
						dlt[r]['parent']= None
					pruned.append( (nt, r, ruleprob) )
					g[nt].remove(r)

			# at least one rule must be remained
			if len(g[nt])==0:
				self.bug('NT %s has no rules!'% nt, g, dlt)

		return g, dlt, pruned

	def substituteMulti(self, argList):
		"""
		argList= [gn, ntlist.pop(), new_gid, P_LOCK]
		"""

		gn, nt, new_gid, P_LOCK= argList[0], argList[1], argList[2], argList[3]
		timer= qc.Timer()
		worse= gn.worse

		g_new, dlt_new= self.substitute(deepcopy(gn.g), nt, self.getNextNT(gn.g), deepcopy(gn.dlt), gn.gid)
		if g_new != None:
			new_pri= self.getPrior(g_new)
			new_lik= self.getLikelihood(g_new, dlt_new)

			g_new, dlt_new, pruned= self.pruneGrammar(g_new, dlt_new)
			if pruned:
				new_pri= self.getPrior(g_new)
				new_lik= self.getLikelihood(g_new, dlt_new)

			new_mdl= new_pri + new_lik
			'''
			if new_mdl >= gn.bestmdl:
				worse += 1
			else:
				worse= 0
			'''

			if VERBOSE > 1:
				msg= '[#%d] After SUBSTITUTE(%s) on #%d\n'% (new_gid, nt, gn.gid)
			else:
				msg= '[#%d from #%d] '% (new_gid,gn.gid)
			msg += 'Worse=%d, Pri=%.3f, Lik=%.3f, OldMDL=%.3f, NewMDL=%.3f, bestBranchMDL=%.3f <%s: %.2f sec>'% \
				(worse, new_pri, new_lik, gn.mdl, new_mdl, gn.bestmdl, current_process().name, timer.sec())

			P_LOCK.acquire()
			if VERBOSE > 1:
				for p in pruned:  self.printMsg(2, '>> Rule %s -> %s [%0.6f] got pruned.' % (p[0],p[1],p[2]) )
				self.printGrammar(2, g_new, dlt_new, msg)
			else:
				self.printMsg(1, msg)
			P_LOCK.release()
		#print( current_process().name, 'finished at %.3f'% time.time() )

		if new_mdl < gn.bestmdl:
			bestmdl= new_mdl
		else:
			bestmdl= gn.bestmdl
		return GNode(g_new, dlt_new, new_pri, new_lik, new_mdl, bestmdl, new_gid, worse)

	def mergeMulti(self, argList):
		"""
		argList= [gn, ntlist.pop(), new_gid, P_LOCK]
		"""

		#print('###### <%s> self.mergeMulti() STARTED -- %.3f'% (current_process().name, time.time()-self.runtime) )
		gn, nt, new_gid, P_LOCK= argList[0], argList[1], argList[2], argList[3]
		timer= qc.Timer()
		worse= gn.worse

		# merge grammar
		g_new, dlt_new= self.merge(nt[0], nt[1], self.getNextNT(gn.g), deepcopy(gn.g), deepcopy(gn.dlt), gn.gid)
		if g_new != None:
			new_pri= self.getPrior(g_new)
			new_lik= self.getLikelihood(g_new, dlt_new)

			g_new, dlt_new, pruned= self.pruneGrammar(g_new, dlt_new)
			if pruned:
				new_pri= self.getPrior(g_new)
				new_lik= self.getLikelihood(g_new, dlt_new)

			new_mdl= new_pri + new_lik
			'''
			if new_mdl >= gn.bestmdl:
				worse += 1
			else:
				worse= 0
			'''

			# debug info
			if VERBOSE > 1:
				msg= '[#%d] After MERGE(%s,%s) on #%d\n'% (new_gid, nt[0], nt[1], gn.gid)
			else:
				msg= '[#%d from #%d] '% (new_gid,gn.gid)
			msg+= 'Worse=%d, Pri=%.3f, Lik=%.3f, OldMDL=%.3f, NewMDL=%.3f, bestBranchMDL=%.3f <%s: %.2f sec>'% \
				(worse, new_pri, new_lik, gn.mdl, new_mdl, gn.bestmdl, current_process().name, timer.sec())

			P_LOCK.acquire()
			if VERBOSE > 1:
				for p in pruned:  self.printMsg(2, '>> Rule %s -> %s [%0.6f] got pruned.' % (p[0],p[1],p[2]) )
				self.printGrammar(2, g_new, dlt_new, msg)
			else:
				self.printMsg(1, msg)
			P_LOCK.release()
		#print('###### <%s> self.mergeMulti() FINISHED -- %.3f'% (current_process().name, time.time()-self.runtime) )

		if new_mdl < gn.bestmdl:
			bestmdl= new_mdl
		else:
			bestmdl= gn.bestmdl
		return GNode(g_new, dlt_new, new_pri, new_lik, new_mdl, bestmdl, new_gid, worse)

	def learnMulti(self, gn, top_n=5):
		"""
		Learn using multiprocessing

		gn: initial grammar
		top_n: save the best top_n grammars
		"""

		# settings
		max_process= cpu_count() # processes to run in parallel; or use cpu_count()

		pool= Pool(max_process)
		P_LOCK= Manager().Lock() # mutex lock for printing
		gidCounter= 0
		results= []
		gList= [gn]
		gnBest= [deepcopy(gn)]
		bestmdl_last= gn.mdl # best MDL on the upper level nodes

		# uncomment this for debugging
		#history_pri_lik= set() # history of prior & likelihood values

		while len(gList) > 0:
			gn= gList.pop(0) # grammar node to be expanded
			self.printMsg(1,'>> gList size: %d, bestMDL: %.3f (#%d)'% (len(gList),gnBest[0].mdl,gnBest[0].gid))

			# substitute if possible
			ntlist= self.getFirstDLThack(gn.dlt)
			if len(ntlist) > 0:
				self.printMsg(2, '>> Possible substitutions on #%d:'%(gn.gid), ntlist)
				while len(ntlist) > 0:
					gidCounter+= 1
					argList= [gn, ntlist.pop(), gidCounter, P_LOCK]
					results.append(pool.apply_async(self.substituteMulti, [argList]))
			else:
				self.printMsg(2, '>> No more SUBSTITUTE possible on #%d\n'% gn.gid)

			# merge if possible
			ntlist= self.mergeSet(gn.g)
			if len(ntlist) > 0:
				self.printMsg(2, '>> Possible merges on #%d:'%(gn.gid), ntlist)
				while len(ntlist) > 0:
					gidCounter+= 1
					argList= [gn, ntlist.pop(), gidCounter, P_LOCK]
					results.append(pool.apply_async(self.mergeMulti, [argList]))
			else:
				self.printMsg(2, '>> No more MERGE possible on #%d.\n'% gn.gid)

			del gn
			delList= []
			bestmdl= bestmdl_last

			# search next level in the search tree
			while len(results) >= 1:
				time.sleep(0.001) # avoid wasting resource
				for r in range(len(results)):
					if results[r].ready():
						delList.append(r)
						gn_new= results[r].get()

						# uncomment for debugging
						#if gn_new.lik < self.max_mdl:
						#	history_pri_lik.add((gn_new.pri,gn_new.lik))

						# save the best-N grammars
						if len(gnBest) < top_n:
							gnBest.append(gn_new)
							gnBest= sorted(gnBest, key=lambda gn: gn.mdl)
						elif gn_new.mdl < gnBest[-1].mdl:
							del gnBest[-1]
							gnBest.append( deepcopy(gn_new) )
							gnBest= sorted(gnBest, key=lambda gn: gn.mdl)

						# save this level's best mdl
						if gn_new.mdl < bestmdl:
							bestmdl= gn_new.mdl

						# beam search: compare with the best mdl on the upper level
						if gn_new.mdl >= bestmdl_last:
							gn_new.worse+= 1
						if gn_new.worse < BEAMSIZE:
							gList.append(gn_new)
						else:
							del gn_new

				delList.sort()
				for d in range(len(delList)):
					del results[delList.pop()]
			bestmdl_last= bestmdl

		pool.close()
		pool.join()

		return gnBest


	def go(self, symbolFile, seqDir):
		global ALGORITHM

		self.upper= sorted(ascii_uppercase, reverse=True)
		self.max_mdl= float("inf")
		self.algorithm= ALGORITHM
		self.term_p_other = None  # computed once in the beginning based on the number of terminals
		self.t_dic_rev = {}  # reverse look-up of self.t_dic defined once in the beginning
		self.t_stat= {}
		self.input_list= [] # history of inputs
		self.testfile= '%s/s%%02d.seq'% (seqDir)
		self.t_seq= open(symbolFile).readline().split()
		self.t_dic= {x:x.upper() for x in self.t_seq}
		#self.runtime= time.time() # for debugging
		assert len(self.t_dic) <= 25
		assert len(self.t_seq) <= 25

		for t in self.t_dic:
			self.t_dic_rev[self.t_dic[t]]=t

		if len(self.t_dic)==1:
			self.term_p_other= 0
		else:
			self.term_p_other= (1.0-TERM_P) / (len(self.t_dic)-1) # prob of other terminals

		rawdata= []
		print('Reading input data')
		for f in qc.get_file_list(seqDir, fullpath=True):
			if f[-4:] != '.seq': continue
			print(f)

			syms = []
			for l in open(f):
				probs= [float(x) for x in l.split()]
				max_sym= qc.get_index_max(probs)
				syms.extend( [ self.t_seq[max_sym], probs[max_sym] ] )
			rawdata.append(syms)

		print('\n>> Algorithm %s started using following parameters:'% self.algorithm)


		'''""""""""""""""""""""""""""""""""""""""""""""""""""""""
		 Build initial grammar from input
		""""""""""""""""""""""""""""""""""""""""""""""""""""""'''
		print(">> Let's go !")
		G= {}
		dlt_init= OrderedDict()
		dlt_init['Z']= {'terms':[],'score':-1,'count':0,'parent':None}
		for t in self.t_dic:
			G= self.addRule(G, self.t_dic[t], t)
			self.t_stat[t]= {'count':0, 'prob':0}
		for rawinput in rawdata:
			input= self.buildInput(rawinput)
			self.input_list.append(input)
			cin= ''.join(input['symbols'])
			G= self.addRule(G, 'Z', cin)
			self.updateTStat(input)
			dlt_init= self.inputDLT(input, dlt_init)
			dlt_init['Z']['terms'].append(self.conv2T(cin))
			#dlt_init[cin]['parent']= 'Z'

		# keep counts only
		if self.algorithm=='STOLCKE':
			for input in self.input_list:
				cin= ''.join(input['symbols'])
				self.t_stat[self.conv2T(cin)]['prob']= 1.0

		self.printTSTAT(1, 'Terminal Symbol Statistics')
		self.printGrammar(1, G, dlt_init, 'After adding new input')
		dlt_init= self.updateDLT(G, dlt_init)
		self.printDLT(1, dlt_init)
		timer= qc.Timer()


		'''""""""""""""""""""""""""""""""""""""""""""""""""""""""
		 Search for the best grammar
		""""""""""""""""""""""""""""""""""""""""""""""""""""""'''
		timer.reset()
		pri= self.getPrior(G)
		lik= self.getLikelihood(G, dlt_init)
		mdl= pri+lik
		gn= GNode(G, dlt_init, pri, lik, mdl, mdl, 0)
		gList= self.learnMulti(gn) # multi-process searching
		timeStamp= timer.sec()


		'''""""""""""""""""""""""""""""""""""""""""""""""""""""""
		 Post-process
		""""""""""""""""""""""""""""""""""""""""""""""""""""""'''
		# delete duplicates
		delList= []
		for i in range(len(gList)): # sort RHS rules to compare equalities
			gList[i].g= self.sortRules( gList[i].g )
		rangePr=[self.max_mdl,0] # min, max of DL(prior)
		rangeLi=[self.max_mdl,0] # min, max of DL(likelihood)
		for i in range(len(gList)):
			pr= gList[i].pri
			li= gList[i].lik
			if ( pr < rangePr[0] ): rangePr[0]= pr
			if ( pr > rangePr[1] ): rangePr[1]= pr
			if li < self.max_mdl:
				if ( li < rangeLi[0] ): rangeLi[0]= li
				if ( li > rangeLi[1] ): rangeLi[1]= li
			for k in range(i+1, len(gList)):
				if gList[i].g==gList[k].g:
					# THINK: SOMETIMES MDL IS DIFF FOR SAME GRAMMARS DUE TO DIFF DLT
					if gList[i].mdl != gList[k].mdl:
						pass
					delList.append(k)
		for d in sorted(list(set(delList)), reverse=True):
			del(gList[d])

		gList= sorted(gList, key=lambda gnode: gnode.mdl)
		self.printGrammarList(0, gList, 'MDL values:')
		self.exportGrammarList(gList)

		print('\n>> Learning finished. Showing input symbols.')
		for input in self.input_list:
			print('    '.join(input['symbols']) )
			for x in input['values']:
				print('%0.2f ' % x, end='')
			print()
		print('\n>> Time consumed: %.1f secs'% timeStamp)
		print('Prune probability: %0.2f'% PRUNE_P)
		print('Terminal confidence (for exporting grammar): %0.2f'% TERM_P)
		print('Search beam size: %d'% BEAMSIZE)
		print('>> Finished.')


if __name__=="__main__":
	if len(sys.argv) < 3:
		print('Usage: %s [SYMBOL_FILE] [INPUT_DIR]'% sys.argv[0])
		print('Example: %s terminals.txt seq/'% sys.argv[0])
		sys.exit()
	symbolFile= sys.argv[1]
	inputDir= sys.argv[2]

	sl= ScfgLearner()
	sl.go(symbolFile, inputDir)
