from __future__ import absolute_import, division, print_function

'''""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Inferring SCFG grammar rules from input with uncertainties.

Author:
Kyuhwa Lee
Imperial College of Science, Technology and Science


Basics of Merging & Substituting:
A. Stolcke, PhD Thesis, UCB, p.93-97

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

GNode= class{g, dlt, pri, pos, mdl, bestmdl, gid, worse}
 Grammar node of a search tree, gList.
 bestmdl: best MDL score observed so far in the current branch
 worse: for beam search (worse += 1 if new_mdl > bestmdl)

T_STAT= {string: {count, prob}}
Statistics of terminal symbols.

T_LIST= {'a':'A', 'b':'B',...}
 Global terminal list.

""""""""""""""""""""""""""""""""""""""""""""""""""""""""'''
import math, os, sys, time, random, ipdb
from collections import OrderedDict
from string import ascii_uppercase
from copy import deepcopy
from operator import itemgetter
from subprocess import Popen, PIPE
from timer import Timer
from multiprocessing import Manager, cpu_count, current_process, active_children, Pool
import sartparser as sp

'''""""""""""""""""""""""""""""""""
 Settings
""""""""""""""""""""""""""""""""'''
ALGORITHM= 'LQ' # 'LQ' or 'STOLCKE'
VERBOSE= 0; # min 0, max 2.
BEAMSIZE= 3 # search-space for beam searching (1= best-first search)
PRUNE_P= 0.01 # prune a rule if low probability
NORM_MDL= False # normalize MDL according to pri/pos ranges
TERM_P= 0.9 # prob of terminal 'a' being really 'a' (for exporting grammar)
MAX_ROUNDS= 1 # the number of times a normalization factor should be updated
MAX_NGRAMS= 46 # maximum sub-patterns to consider
EXPORT_INPUT= False # export input into *.seq each time for parsing

'''""""""""""""""""""""""""""""""""
 Global constants
""""""""""""""""""""""""""""""""'''
RUNTIME= time.time()
TERM_P_OTHER= -1
UPPER= sorted(ascii_uppercase, reverse=True)
MAX_MDL= 10000000 # dl_posterior max value
T_LIST_REV= {} # reverse look-up of T_LIST

'''""""""""""""""""""""""""""""""""
 Grammar node object
""""""""""""""""""""""""""""""""'''
class GNode:
	def __init__(self, g,dlt,pri,pos,mdl,bestmdl,gid,worse=0):
		self.g= g
		self.dlt= dlt
		self.pri= pri
		self.pos= pos
		self.mdl= mdl
		self.bestmdl= bestmdl
		self.gid= gid
		self.worse= worse

'''""""""""""""""""""""""""""""""""
 General-Purpose Helper Functions
""""""""""""""""""""""""""""""""'''
# uniform Dirichlet distribution with sigma(alpha)=1.0
def dirichlet(n):
	alpha= 1.0/n
	return 1/beta(alpha,n) * ((1/n)**(alpha-1))**n

# multinomial Beta function with uniform alpha values
def beta(alpha, n): # n: number of rule probabilities
	return math.gamma(alpha)**n / math.gamma(n*alpha)

# Poisson distribution. We use k-1 since the minimum length is 1, not 0.
def poisson(mean, k):
	return (mean**(k-1) * math.exp(-mean)) / math.factorial(k-1)

# return the mean of a list
def average(vlist):
	assert len(vlist)>0, '>> Error: len(vlist)==0 !'
	return sum(vlist, 0.0) / len(vlist)

# return the variance of a list
def variance(vlist):
	assert len(vlist)>0, '>> Error: len(vlist)==0 !'
	if len(vlist)==1: return 0
	s= 0
	m= average(vlist)
	for v in vlist:
		s+= (v-m)**2
	return s / (len(vlist)-1) # Bessel's correction

# return standard deviation of a list
def stddev(vlist):
	return math.sqrt(variance(vlist))

# return the key of dictionary dic given the value
def find_key(dic, val):
	for x in dic:
		if val==dic[x]:
			return x
	return None

# remove duplicates: doesn't preserve orders
def	uniquify(l):
	return list(set(l))

# build n-grams from seq up to n=maxw
def ngrams(seq, maxw=MAX_NGRAMS):
	nglist=[]
	inlen= len(seq)
	if maxw > inlen: maxw= inlen

	# n-grams with 1 <= n <= maxw
	for w in range(1,maxw+1):
		for x in range(inlen-w+1):
			nt= ''.join(seq[x:x+w])
			if nt not in nglist:
				nglist.append(nt)
	return nglist

'''""""""""""""""""""""""""""""""""
 Main Functions
""""""""""""""""""""""""""""""""'''
# convert input terminals into corresponding NT's
def conv2NT(t):
	str= ''
	for x in t:
		str += T_LIST[x]
	return str

# convert preterminals to terminals
def conv2T(nt):
	str= ''
	for x in nt:
		str += T_LIST_REV[x]
	return str

# description length of prior probability P(G)
# Stolcke PhD Thesis,"Bayesian learning of probabilistic language models",Sec 2.5.5
def getPrior(g):
	# parameter prior (terminal symobls + non-terminal symbols)
	num_symbols= len(T_LIST.keys()) + len(g.keys())
	dl_theta= -math.log(dirichlet(num_symbols),2)

	# structure prior (Poisson distribution of the grammar length)
	dl_S= 0
	mu= 3
	for s in g:
		for r in g[s]:
			rlen= len(r)+1
			# expected bits for (length prior + each rule length * num_symbols)
			dl_S += -math.log(poisson(mu,rlen),3) + rlen*math.log(num_symbols,2)
			#dl_S += -math.log(poisson(mu,rlen),3)

	pri= dl_S+dl_theta
	#pri= dl_S

	# prior correction
	global normer
	pri= abs( (pri-normer['pri']['mean'])/normer['pri']['std'] )

	return pri


def grammar2sartgrammar(g, dlt, method=0):
	global T_LIST_REV

	gs= sp.CFGrammar()

	# Define axiom
	gs.addAxiom('Z')

	# Define non-terminals
	for s in g.keys():
		gs.addNonTerminal(s)
	if method==2:
		skip= ascii_uppercase[len(T_LIST)] # add a SKIP terminal
		gs.addNonTerminal(skip)

	# Define terminals
	for s in T_SEQ:
		gs.addTerminal(s)

	# Define rules
	# Default method
	if method==0:
		for t in T_LIST:
			gs.addRule(T_LIST[t], [t], 1.0)

	# Robust method 1: A -> a|b|c|d, B -> a|b|c|d ...
	elif method==1:
		for nt in T_LIST_REV:
			for t in T_LIST:
				if T_LIST[t]==nt:
					rulescore= TERM_P
				else:
					rulescore= TERM_P_OTHER
				gs.addRule(nt, [t], rulescore)

	# Robust method2: A -> a|SKIP, SKIP -> SKIP SKIP|a|b|c|d ...
	elif method==2:
		pskipself= 0.01
		pskip= (1-pskipself) / len(T_LIST)
		gs.addRule(skip, [skip, skip], pskipself)
		for t in T_LIST:
			gs.addRule(skip, t, pskip)
		for nt in T_LIST_REV:
			gs.addRule(nt, [T_LIST_REV[t]], TERM_P)
			gs.addRule(nt, skip, 1-TERM_P)

	# Actual grammar body
	for nt in getNTlist(g):
		sum= 0.0 # sum of all rule scores belonging to nt
		for r in g[nt]: # for each rule of a non-terminal
			rulescore= 0.0
			for t in dlt[r]['terms']:
				if t not in T_STAT:
					bug('%s key is not in T_STAT'%t, g, dlt)
				rulescore += T_STAT[t]['prob'] * T_STAT[t]['count']
			#rulescore *= len(r)
			sum += rulescore
		for r in g[nt]:
			rulescore= 0.0 # the score of each rule
			for t in dlt[r]['terms']:
				rulescore += T_STAT[t]['prob'] * T_STAT[t]['count']
			#rulescore *= len(r)
			if sum==0: # when input probabilities of all terminals are 0
				gs.addRule(nt, list(r), 1.0/len(g[nt]))
			else:
				gs.addRule(nt, list(r), rulescore/sum)
	return gs


def getPosterior(g, dlt, verbose=VERBOSE):
	global INPUT_LIST, TESTFILE

	gs= grammar2sartgrammar(g, dlt)
	parser= sp.SParser(gs)

	for i in range(len(INPUT_LIST)):
		fs= open(TESTFILE % i)
		for s in fs:
			sw= s.strip()
			if len(sw)==0: continue
			if sw[0] != '#':
				parser.parseLine(sw)

	pv= parser.getViterbiParse()
	psc= pv.probability.scaled
	print(psc)
	return min(MAX_MDL, psc)

def getPosterior_KILLME(g, dlt, verbose=VERBOSE):
	gFile= 'Grammars/%s.grm'% (DATASET+'_'+str(time.time())+str(random.random()) )
	if os.access(gFile, os.F_OK)==True:
		bug('Duplicate grammar file exists!')
	exportGrammar(g, dlt, gFile, METHOD=2)
	sum= 0
	ipdb.set_trace()

	for i in range(len(INPUT_LIST)):
		cmd= './tryPS_ram %s %s | tail -n1' % (gFile, TESTFILE%i)
		res= Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE, close_fds=True).communicate()
		stdout= res[0].decode("utf-8")
		stderr= res[1].decode("utf-8").strip()
		#print (cmd)
		#print (stdout)
		result= stdout.split()
		if stderr != '':
			if verbose > 0:
				print('*'*60)
				printGrammar(0, g, dlt, '*'*20+' DEBUG INFORMATION' +'*'*20)
				if verbose > 1: printDLT(0, dlt)
				print('## Parsing error: stderr=',stderr)
				print('Command was:', cmd)
				print('*'*60)
				wtime= 1
				print('## Resuming after %d secs. Press Ctrl+C to stop. Usually ok to continue.'% wtime)
				time.sleep(wtime)
			else:
				print('## Parsing error: stderr=',stderr)
			sum= MAX_MDL
			break
		else:
			try:
				p= float(result[-1])
				if p==0: sum= MAX_MDL
				else: sum += -math.log(p,2)
			except:
				if result[-1] != 'found':
					print( 'getPosterior(): Exception Occurred!', sys.exc_info()[1] )
					bug('getPosterior(): sum=%.1f, result=%s'% (sum, result), g, dlt)
				else: # final state not found
					print( 'getPosterior() no final state. result:', ' '.join(result) )
					print('cmd=%s'% cmd)
					sum= MAX_MDL
	os.remove(gFile)
	if sum < MAX_MDL:
		pos= sum/len(INPUT_LIST) # average of posteriors over input sequences
		# posterior correction
		global normer
		pos= abs( (pos-normer['pos']['mean'])/normer['pos']['std'] )
	else:
		pos= MAX_MDL

	return pos

# MDL score
def getMDL(g, dlt, verbose=False):
	return getPrior(g) + getPosterior(g, dlt, verbose)

# count the number of appearances of given string in RHS of g
def getStringCount(g, strings):
	c= 0
	for s in g:
		for r in g[s]:
			c += r.count(strings)
	return c

# returns the list of NT's in g except terminal NT's. e.g. [Z,Y,X..]
def getNTlist(g):
	ntlist=[]
	for x in g.keys():
		if x not in T_LIST.values():
			ntlist.append(x)
	return sorted(ntlist, reverse=True)

def printMsg(minVerbosity, *args):
	if VERBOSE < minVerbosity: return
	for msg in args:
		print(msg, end=' ')
	print()

# print input strings
def printInput(minVerbosity, inp):
	if VERBOSE < minVerbosity: return
	print('\n-- New Input Sequence --')
	for x in range(len(inp['symbols'])):
		print('%s  %0.2f'% (inp['symbols'][x],inp['values'][x]))
	print()

def printDLT(minVerbosity, dlt, msg='Description Length Table'):
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

def printTSTAT(minVerbosity, msg='Terminal Symbol Statistics'):
	if VERBOSE < minVerbosity: return
	print(' --',msg,'--','<%s>'% current_process().name)
	print('%-5s  %-8s  %s' % ('COUNT','PROB','STRING') )
	for s in T_STAT:
		print( '%-5d  %0.6f  %s' % (T_STAT[s]['count'],T_STAT[s]['prob'],s) )
	print()

def printMDL(minVerbosity, g, dlt):
	if VERBOSE < minVerbosity: return
	pri= getPrior(g)
	pos= getPosterior(g, dlt)
	print('DL_prior: %.3f'% pri )
	print('DL_posterior: %.3f'% pos )
	print('MDL: %.3f'% (pri+pos) )
	print()

# print grammar with msg
def printGrammar(minVerbosity, g, dlt, msg=''):
	if VERBOSE < minVerbosity: return
	print(msg)
	for nt in getNTlist(g):
		sum= 0.0 # sum of all rule scores belonging to nt
		sortedRules= [] # grammar with each rule sorted by scores
		for r in range(len(g[nt])): # for each RHS rule of a LHS symbol
			rulescore= 0.0
			for t in dlt[g[nt][r]]['terms']:
				if t not in T_STAT:
					bug('%s key is not in T_STAT'%t, g, dlt)
				rulescore += T_STAT[t]['prob'] * T_STAT[t]['count']
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
				rulescore += T_STAT[t]['prob'] * T_STAT[t]['count']
				terms += t + ' '
			#rulescore *= len(rs)
			if sum== 0:
				print('%-18s [%0.6f] (%0.2f) <%s> %s'%
					(rs, 1.0, rulescore, dlt[rs]['parent'], terms))
			else:
				print('%-18s [%0.6f] (%0.2f) <%s> %s'%
					(rs, rulescore/sum, rulescore, dlt[rs]['parent'], terms))
			if dlt[rs]['parent']== None:
				#bug('no parent', g, dlt)
				pass
	print()

# print grammar without rule probabilities
def printGrammarSimple(minVerbosity, g, msg=''):
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

def printGrammarMDL(minVerbosity, g, dlt, msg=''):
	if VERBOSE < minVerbosity: return
	printGrammar(g, dlt, msg)
	printMDL(minVerbosity, g, dlt)

def printGrammarList(minVerbosity, gList, msg=''):
	if VERBOSE < minVerbosity: return
	print('\n'+'='*60)
	print(' ', msg, 'Showing the best grammars')
	print('='*60)
	for n in range(len(gList)):
		printGrammar(minVerbosity, gList[n].g, gList[n].dlt, \
			'#%d, MDL: %.6f, DL_pri: %.6f, DL_pos: %.6f'%\
			(n+1, gList[n].mdl, gList[n].pri, gList[n].pos) )
		#printDLT(gList[n].dlt)
		print('-'*60)

def exportGrammarList(gList, trial):
	global DATASET, ALGORITHM
	if not os.path.exists('Output'):
		os.mkdir('Output')
	for n in range(len(gList)):
		exportGrammar(gList[n].g, gList[n].dlt, 'Output/rank%d_%s_%s.grm'%\
			(n, DATASET, trial), 2 )

def exportInput():
	# export input data
	tlist= T_SEQ
	for r in range(len(INPUT_LIST)):
		f=open(TESTFILE%r, 'w')
		f.write('#')
		for t in tlist: f.write('%8s '% t)
		f.write('\n#')
		for t in tlist: f.write('%-9s'% ('-'*8) )
		f.write('\n')
		input= INPUT_LIST[r]
		for i in range(len(input['symbols'])):
			p= input['values'][i]
			po= (1-p) / (len(tlist)-1)
			s= T_LIST_REV[input['symbols'][i]]
			for t in tlist:
				if s==t: f.write(' %0.6f'% p)
				else: f.write(' %0.6f'% po)
			f.write(' # %s'% s)
			f.write('\n')
		f.close()

def bug(msg, g='', dlt=''):
	sys.stderr.write('>> BUG FOUND: %s'% msg)
	print('#'*80)
	print('>> BUG FOUND: %s'% msg)
	print('#'*80)
	if g: printGrammarSimple(0, g, '## Raw Grammar Rules')
	if dlt: printDLT(0, dlt)
	print('#'*80)
	sys.exit()

def exportGrammar(g, dlt, filename, METHOD=0):
	global T_LIST_REV

	f=open(filename,'w')
	f.write('Section Terminals\n')
	for s in T_SEQ: f.write('%s '% s)
	f.write('\n\nSection NonTerminals\n')
	for s in g: f.write('%s '% s)
	if METHOD==2:
		skip= ascii_uppercase[len(T_LIST)]
		f.write(skip)
	f.write('\n\nSection Axiom\n')
	f.write('Z\n\nSection Productions\n')

	# default
	if METHOD==0:
		for t in T_LIST:
			f.write('%s: %-20s [%0.6f]\n' % (T_LIST[t], t, 1.0))
	# Robust Input Method 1: A -> a|b|c|d, B -> a|b|c|d ...
	elif METHOD==1:
		for nt in T_LIST_REV:
			firstLine= True
			for t in T_LIST:
				if firstLine:
					firstLine= False
					id='%s:'% nt
				else:
					id=' |'
				if T_LIST[t]==nt:
					rulescore= TERM_P
				else:
					rulescore= TERM_P_OTHER
				f.write('%s %-20s [%0.6f]\n' % (id, t, rulescore))
	# Robust Input Method2: A -> a|SKIP, SKIP -> SKIP SKIP|a|b|c|d ...
	elif METHOD==2:
		pskipself= 0.01
		pskip= (1-pskipself) / len(T_LIST)
		f.write('%s: %s %-18s [%0.6f]\n' % (skip, skip, skip, pskipself) )
		for t in T_LIST:
			f.write(' | %-20s [%0.6f]\n' % (t, pskip) )
		for t in T_LIST_REV:
			f.write('%s: %-20s [%0.6f]\n' % (t, T_LIST_REV[t], TERM_P))
			f.write(' | %-20s [%0.6f]\n' % (skip, 1-TERM_P))

	for nt in getNTlist(g):
		sum= 0.0 # sum of all rule scores belonging to nt
		for r in g[nt]: # for each rule of a non-terminal
			rulescore= 0.0
			for t in dlt[r]['terms']:
				if t not in T_STAT:
					bug('%s key is not in T_STAT'%t, g, dlt)
				rulescore += T_STAT[t]['prob'] * T_STAT[t]['count']
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
				rulescore += T_STAT[t]['prob'] * T_STAT[t]['count']
			#rulescore *= len(r)
			if sum==0: # when input probabilities of all terminals are 0
				f.write('%s %-20s [%0.6f]\n' % (id, ' '.join(list(r)), 1.0/len(g[nt])))
			else:
				f.write('%s %-20s [%0.6f]\n' % (id, ' '.join(list(r)), rulescore/sum))
	f.close()

# build input from raw data
def buildInput(seq):
	assert len(seq) % 2==0, 'You have wrong length of input sequence.'
	input= {'symbols':[], 'values':[]}
	for i in range(0,len(seq),2):
		input['symbols'].append(conv2NT(seq[i]))
		input['values'].append(float(seq[i+1]))
	return input

# return the next available NT symbol
def getNextNT(g):
	for s in UPPER:
		if s not in g:
			assert s not in T_LIST.values(), 'getNextNT(): assertion error!'
			return s

# return the list of strings split by NT and chunk of T's. [NT,T,NT,NT,T,NT...]
def splitTNT(str):
	seq= []
	wasNT= True
	for x in str:
		if x not in T_LIST or wasNT:
			seq += x
		else:
			seq[-1] += x
		wasNT= x not in T_LIST
	return seq

# register a new rule that belongs to nt
def addRule(g, nt, string):
	if nt not in g:
		g[nt]= [string]
	elif string not in g[nt]:
		g[nt].append(string) # if not identical input
	else:
		#print('>> AddRule(): Ignored adding pre-existing strings',string)
		pass
	return g

# delete a rule that belongs to nt
def delRule(g, nt):
	if nt not in g: bug('delRule(): NT(%s) is not in G!'% nt)
	del g[nt]
	return g

# sort RHS rules in dictionary order
def sortRules(g):
	for s in getNTlist(g): g[s]= sorted(g[s])
	return g

# sort LHS symbols in reversed alphabetical order(e.g. W, X -> Z, Y)
def sortSymbols(g, dlt):
	ntlist= getNTlist(g)
	for x in range(len(ntlist)):
		if ntlist[x] != UPPER[x]:
			nt_old= ntlist[x]
			nt_new= UPPER[x]
			printMsg(2, '>> Re-ordering %s to %s <%s>'% (nt_old, nt_new, current_process().name) )
			for s in g:
				for x in range(len(g[s])):
					r= g[s][x]
					if nt_old in r:
						if r in dlt and dlt[r]['parent'] != None:
							dlt[r]['parent']= dlt[r]['parent'].replace(nt_old, nt_new)
						g[s][x]= r.replace(nt_old, nt_new)
				if s==nt_old:
					if nt_new in g:  bug('sortSymbols(): nt_new(%s) already in G!'% nt_new)
					g[nt_new]= g.pop(nt_old)
			for s in dlt:
				if nt_old in s:
					new_s= s.replace(nt_old, nt_new)
					if new_s in dlt:  bug('sortSymbols(): new_s(%s) already in DLT!'% new_s)
					dlt[new_s]= dlt.pop(s)
	return g, dlt

# update DLT's count and score to reflect the current grammar
def updateDLT(g, dlt, maxw=MAX_NGRAMS):
	# invalidate DLT
	for x in dlt:
		dlt[x]['count']= 0

	# build n-grams
	for s in getNTlist(g):
		for r in g[s]:
			nglist= ngrams(r, maxw)
			for nt in nglist:
				if nt not in dlt:
					bug('updateDLT(): %s is not in DLT'% nt, g, dlt)
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

# update T_STAT with new input (substrings up to length=maxw)
def updateTStat(input, maxw=MAX_NGRAMS):
	global T_STAT, T_LIST_REV, ALGORITHM
	inlen= len(input['symbols'])
	if maxw > inlen: maxw= inlen

	# n-grams with 1 <= n <= maxw
	for w in range(1,maxw+1):
		for x in range(inlen-w+1):
			if ALGORITHM=='STOLCKE':
				njp= 0.0
			else:
				njp= 1.0
			term= ''
			for i in range(x,x+w):
				njp *= input['values'][i]
				term += T_LIST_REV[input['symbols'][i]]
			njp= math.pow(njp,1/w)

			if term not in T_STAT:
				T_STAT[term]={'count':1, 'prob':njp}
			else:
				t= T_STAT[term]
				T_STAT[term]['prob']= (t['count'] * t['prob']	+ njp ) / (t['count']+1)
				T_STAT[term]['count'] += 1

	''' consider MAX_NGRAMS
	# and the whole input regardless of maxw
	njp= 1.0
	term= ''
	for i in range(inlen):
		njp *= input['values'][i]
		term += T_LIST_REV[input['symbols'][i]]
	njp= math.pow(njp, 1/w)

	if term not in T_STAT:
		T_STAT[term]= {'count':1, 'prob':njp}
	else:
		t= T_STAT[term]
		T_STAT[term]['prob']= (t['count'] * t['prob']	+ njp ) / (t['count']+1)
		T_STAT[term]['count'] += 1
	'''

# update DLT with new input (substrings up to length=maxw)
def inputDLT(input, dlt, maxw=MAX_NGRAMS):
	global T_LIST_REV
	inlen= len(input['symbols'])
	if maxw > inlen: maxw= inlen

	# n-grams with 1 <= n <= maxw
	for w in range(1,maxw+1):
		for x in range(inlen-w+1):
			nt= ''.join([z[0] for z in input['symbols'][x:x+w]])
			term= ''
			for i in range(x,x+w):
				term += T_LIST_REV[input['symbols'][i]]
			if nt not in dlt:
				dlt[nt]={'score':-1,'terms':[term],'count':0,'parent':None}

	''' consider MAX_NGRAMS
	# and the whole input regardless of maxw
	nt= ''.join(input['symbols'])
	term= ''
	for t in input['symbols']:
		term += T_LIST_REV[t]
	if nt not in dlt:
		dlt[nt]={'score':-1,'terms':[term],'count':0,'parent':None}
	'''

	return dlt

# return the highest score string X in DLT with conditions:
# 1. parent: {False: X has no parent | True: X has parent}
# 2. strings: X is substring of strings; '' if don't care
def getFirstDLT(dlt, strings='', parent=False):
	global PRUNE_P
	lastScore= -1 # minimum score
	maxItem= []
	for s in dlt:
		if dlt[s]['score'] < lastScore:  break
		#if parent and parent==(dlt[s]['parent']==None): continue
		if strings != '' and s not in strings:  continue
		termprob= 0.0
		for t in dlt[s]['terms']:
			termprob += T_STAT[t]['prob']
		if termprob <= PRUNE_P:  continue
		lastScore= dlt[s]['score']
		maxItem.append(s)
	return maxItem

# Consider only limited-length words for higher speed
def getFirstDLThack(dlt, strings='', parent=False):
	global PRUNE_P
	maxItem= []
	for s in dlt:
		if len(s) < 2 or len(s) > 5: continue
		termprob= 0.0
		for t in dlt[s]['terms']:
			termprob += T_STAT[t]['prob']
		if termprob <= PRUNE_P:  continue
		if dlt[s]['count'] > 1:
			maxItem.append(s)
	return maxItem
##################################################################

# substitute symbols in input and update DLT
def substituteInput(g, r, dlt, strings, new_nt, nt='Z'):
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
	for ngram in ngrams(g[nt][r]): # for each sub-pattern
		newgram= ngram.replace('_', new_nt)
		if newgram in dlt: continue
		ref= ngram.replace('_', strings)
		#if len(ref) > MAX_NGRAMS: continue
		try:
			dlt[newgram]= deepcopy(dlt[ref])
		except Exception:
			bug('in substitute(): while trying to copy dlt[%s] to dlt[%s]'%(ref,ngram), g, dlt)
	g[nt][r]= g[nt][r].replace('_', new_nt) # back to normal
	dlt= updateDLT(g, dlt)
	return g, dlt

# substitute symbols
def substitute(g, strings, new_nt, dlt, gid):
	#print( '>> SUBSTITUTE(%s)=%s on Grammar #%d'% (strings, new_nt, gid) )
	#assert(new_nt not in g)

	# update existing rules
	for nt in getNTlist(g):
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
	g= addRule(g, new_nt, strings)

	# update child-parent relationship
	if new_nt not in dlt:
		dlt[new_nt]={'terms':deepcopy(dlt[strings]['terms']),'score':-1,'count':0,'parent':None}
	dlt[strings]['parent']= new_nt

	# build new n-grams including new NT
	for s in getNTlist(g): # for each NT in g
		for r in range(len(g[s])): # for each rule in NT
			for ngram in ngrams(g[s][r]): # for each sub-pattern
				newgram= ngram.replace('_', new_nt)
				if newgram in dlt: continue
				ref= ngram.replace('_', strings)
				#if len(ref) > MAX_NGRAMS: continue
				try:
					dlt[newgram]= deepcopy(dlt[ref])
				except Exception:
					bug('in substitute(): while trying to copy dlt[%s] to dlt[%s]'%(ref,ngram), g, dlt)
			g[s][r]= g[s][r].replace('_', new_nt) # back to normal
		g[s]= uniquify(g[s])
	dlt= updateDLT(g, dlt)

	return g, dlt

# return nt combinations to merge
def mergeSet(g):
	s= []
	l= getNTlist(g)
	for n1 in range(len(l)):
		for n2 in range(n1+1,len(l)):
			s.append( (l[n1],l[n2]) )
	return s

def merge(nt1, nt2, new_nt, g, dlt, gid):
	'''"""""""""""""""""""""""""""""""""""
	     - Two special cases -

	1. need to check self-recursion!
	 Y -> X
	 merge(X,Y):
	 Y -> Y should be deleted

	2. need to check for uniqueness!
	 Z1 -> AYB (c1)
	    -> AXB (c2)
	 merge(X,Y):
	 Z1 -> AYB (c1)
	    -> AYB (c2)
	 should become:
	 Z1 -> AYB (c1+c2)
	"""""""""""""""""""""""""""""""""""'''
	#print( '>> MERGE(%s,%s)=%s on Grammar #%d'% (nt1, nt2, new_nt, gid) )

	# replace nt1, nt2 to new_nt in both LHS & RHS
	for s in getNTlist(g):
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
		bug('This actually happened. While merging, no rules are left on merged RHS !')

	# update grammar
	for r in new_rules:
		g= addRule(g, new_nt, r)
	g= delRule(g, nt1)
	g= delRule(g, nt2)

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
				################ THIS IS HACK #####################################
				# TO AVOID MULTIPLE PARENTS WHICH ACTUALLY DOESN'T AFFECT PROGRAM #
				###################################################################
				if dlt[s]['parent']: dlt[new_string]['parent']= dlt[s]['parent']
				del(dlt[s])

	# THINK: SHOULD DLT UPDATED HERE?
	dlt= updateDLT(g, dlt)

	############# THIS IS SLOWER METHOD ########################
	# Instead of adding new_nt, just merge into higher-order NT
	# e.g. MERGE(Z,X)=Z
	############################################################
	if UPPER.index(nt1) < UPPER.index(nt2):
		nt_old= new_nt
		nt_new= nt1
	else:
		nt_old= nt1
		nt_new= new_nt
	for s in getNTlist(g):
		for x in range(len(g[s])):
			r= g[s][x]
			if r in dlt and dlt[r]['parent'] != None:
				dlt[r]['parent']= dlt[r]['parent'].replace(nt_old, nt_new)
			g[s][x]= r.replace(nt_old, nt_new)
		if s==nt_old:
			if nt_new in g:  bug('merge(): nt_new(%s) is already in G!'% nt_new)
			g[nt_new]= g.pop(nt_old)
	for s in dlt:
		if nt_old in s:
			new_s= s.replace(nt_old, nt_new)
			if new_s in dlt:  bug('merge(): new_s(%s) is already in G!'% new_s)
			dlt[new_s]= dlt.pop(s)
	###########################################################

	# DLT should be updated here!
	dlt= updateDLT(g, dlt)
	g, dlt= sortSymbols(g, dlt)
	#if getPosterior(g, dlt) < MAX_MDL:  return g, dlt
	#else:  return None, None
	return g, dlt

# prune rules with low probability
def pruneGrammar(g, dlt):
	pruned= []
	for nt in getNTlist(g):
		sum= 0.0 # sum of all rule scores belonging to nt
		for r in range(len(g[nt])): # for each RHS rule of a LHS symbol
			rulescore= 0.0
			for t in dlt[g[nt][r]]['terms']:
				if t not in T_STAT:
					bug('%s key is not in T_STAT'%t, g, dlt)
				rulescore += T_STAT[t]['prob'] * T_STAT[t]['count']
			#rulescore *= len(g[nt][r])
			sum += rulescore
		if sum==0: # all rule probs are identical; cannot continue
			return g, dlt, pruned

		# compute rule probs and prune if needed
		for r in g[nt]:
			rulescore= 0.0 # the score of each rule
			for t in dlt[r]['terms']:
				rulescore += T_STAT[t]['prob'] * T_STAT[t]['count']
			#rulescore *= len(rs)
			ruleprob= rulescore / sum
			if ruleprob < PRUNE_P:
				if dlt[r]['parent']==nt:
					dlt[r]['parent']= None
				pruned.append( (nt, r, ruleprob) )
				g[nt].remove(r)

		# at least one rule must be remained
		if len(g[nt])==0:
			bug('NT %s has no rules!'% nt, g, dlt)

	return g, dlt, pruned

# argList= [gn, ntlist.pop(), new_gid, P_LOCK]
def substituteMulti(argList):
	gn, nt, new_gid, P_LOCK= argList[0], argList[1], argList[2], argList[3]
	timer= Timer()
	worse= gn.worse

	g_new, dlt_new= substitute(deepcopy(gn.g), nt, getNextNT(gn.g), deepcopy(gn.dlt), gn.gid)
	if g_new != None:
		new_pri= getPrior(g_new)
		new_pos= getPosterior(g_new, dlt_new)

		g_new, dlt_new, pruned= pruneGrammar(g_new, dlt_new)
		if pruned:
			new_pri= getPrior(g_new)
			new_pos= getPosterior(g_new, dlt_new)

		new_mdl= new_pri + new_pos
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
		msg += 'Worse=%d, Pri=%.3f, Pos=%.3f, OldMDL=%.3f, NewMDL=%.3f, bestBranchMDL=%.3f <%s: %.2f sec>'% \
			(worse, new_pri, new_pos, gn.mdl, new_mdl, gn.bestmdl, current_process().name, timer.read())

		P_LOCK.acquire()
		if VERBOSE > 1:
			for p in pruned:  printMsg(2, '>> Rule %s -> %s [%0.6f] got pruned.' % (p[0],p[1],p[2]) )
			printGrammar(2, g_new, dlt_new, msg)
		else:
			printMsg(1, msg)
		P_LOCK.release()
	#print( current_process().name, 'finished at %.3f'% time.time() )

	if new_mdl < gn.bestmdl:
		bestmdl= new_mdl
	else:
		bestmdl= gn.bestmdl
	return GNode(g_new, dlt_new, new_pri, new_pos, new_mdl, bestmdl, new_gid, worse)

# argList= [gn, ntlist.pop(), new_gid, P_LOCK]
def mergeMulti(argList):
	#print('###### <%s> mergeMulti() STARTED -- %.3f'% (current_process().name, time.time()-RUNTIME) )
	gn, nt, new_gid, P_LOCK= argList[0], argList[1], argList[2], argList[3]
	timer= Timer()
	worse= gn.worse

	# merge grammar
	g_new, dlt_new= merge(nt[0], nt[1], getNextNT(gn.g), deepcopy(gn.g), deepcopy(gn.dlt), gn.gid)
	if g_new != None:
		new_pri= getPrior(g_new)
		new_pos= getPosterior(g_new, dlt_new)

		g_new, dlt_new, pruned= pruneGrammar(g_new, dlt_new)
		if pruned:
			new_pri= getPrior(g_new)
			new_pos= getPosterior(g_new, dlt_new)

		new_mdl= new_pri + new_pos
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
		msg+= 'Worse=%d, Pri=%.3f, Pos=%.3f, OldMDL=%.3f, NewMDL=%.3f, bestBranchMDL=%.3f <%s: %.2f sec>'% \
			(worse, new_pri, new_pos, gn.mdl, new_mdl, gn.bestmdl, current_process().name, timer.read())

		P_LOCK.acquire()
		if VERBOSE > 1:
			for p in pruned:  printMsg(2, '>> Rule %s -> %s [%0.6f] got pruned.' % (p[0],p[1],p[2]) )
			printGrammar(2, g_new, dlt_new, msg)
		else:
			printMsg(1, msg)
		P_LOCK.release()
	#print('###### <%s> mergeMulti() FINISHED -- %.3f'% (current_process().name, time.time()-RUNTIME) )

	if new_mdl < gn.bestmdl:
		bestmdl= new_mdl
	else:
		bestmdl= gn.bestmdl
	return GNode(g_new, dlt_new, new_pri, new_pos, new_mdl, bestmdl, new_gid, worse)

# multiprocessing
def learnMulti(gn, normer, mfile):
	# settings
	MAX_PROCESS= cpu_count() # processes to run in parallel; or use cpu_count()
	TOP_N= 5 # save Best-N grammars

	pool= Pool(MAX_PROCESS)
	P_LOCK= Manager().Lock() # mutex lock for printing
	gidCounter= 0
	results= []
	gList= [gn]
	gnBest= [deepcopy(gn)]
	bestmdl_last= gn.mdl # best MDL on the upper level nodes
	pripos= set() # history of prior & posterior values

	while len(gList) > 0:
		gn= gList.pop(0) # grammar node to be expanded
		printMsg(1,'>> gList size: %d, bestMDL: %.3f (#%d)'% (len(gList),gnBest[0].mdl,gnBest[0].gid))

		# substitute if possible
		ntlist= getFirstDLThack(gn.dlt)
		if len(ntlist) > 0:
			printMsg(2, '>> Possible substitutions on #%d:'%(gn.gid), ntlist)
			while len(ntlist) > 0:
				gidCounter+= 1
				argList= [gn, ntlist.pop(), gidCounter, P_LOCK]
				results.append(pool.apply_async(substituteMulti, [argList]))
		else:
			printMsg(2, '>> No more SUBSTITUTE possible on #%d\n'% gn.gid)

		# merge if possible
		ntlist= mergeSet(gn.g)
		if len(ntlist) > 0:
			printMsg(2, '>> Possible merges on #%d:'%(gn.gid), ntlist)
			while len(ntlist) > 0:
				gidCounter+= 1
				argList= [gn, ntlist.pop(), gidCounter, P_LOCK]
				results.append(pool.apply_async(mergeMulti, [argList]))
		else:
			printMsg(2, '>> No more MERGE possible on #%d.\n'% gn.gid)

		del gn
		delList= []
		bestmdl= bestmdl_last

		# search next level in the search tree
		while len(results) >= 1:#MAX_PROCESS: # for pipe-lining
			time.sleep(0.001) # avoid wasting resource
			for r in range(len(results)):
				if results[r].ready():
					delList.append(r)
					gn_new= results[r].get()
					if gn_new.pos < MAX_MDL:
						pripos.add((gn_new.pri,gn_new.pos))

					# save best-N grammars
					if len(gnBest) < TOP_N:
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

	priHistory= []
	posHistory= []
	while len(pripos) > 0:
		pp= pripos.pop()
		priHistory.append(pp[0])
		posHistory.append(pp[1])
	normer['pri']= {'mean':average(priHistory), 'std':stddev(priHistory)}
	normer['pos']= {'mean':average(posHistory), 'std':stddev(posHistory)}

	# save prior and posteriors to mfile
	mstr= 'priBest=%.12f;\nposBest=%.12f;\n'%(gnBest[0].pri,gnBest[0].pos)
	mstr+= 'priMean=%.12f;\npriStd=%.12f;\n'%(normer['pri']['mean'],normer['pri']['std'])
	mstr+= 'posMean=%.12f;\nposStd=%.12f;\n'%(normer['pos']['mean'],normer['pos']['std'])
	mstr+= 'priHistory=['
	for p in priHistory:
		mstr+= '%.12f,'%p
	mstr= mstr[:-1]+'];\n'
	mstr+= 'posHistory=['
	for p in posHistory:
		mstr+= '%.12f,'%p
	mstr= mstr[:-1]+'];\n'
	f= open(mfile, 'w')
	f.write(mstr)
	f.close()
	return gnBest, normer


def go(dataset, trial, inputFile, testDir):
	global T_STAT, INPUT_LIST, DATASET, TESTFILE, ALGORITHM
	global TERM_P_OTHER, T_LIST, T_SEQ, T_LIST_REV, normer
	DATASET= dataset # noise type
	NORM_FACTOR= 1 # initial value of prior DL normalizer
	T_STAT= {}
	INPUT_LIST= [] # history of inputs
	vars= {}

	# prepare ramdisk for higher performance
	os.system('ramdisk')

	TESTFILE= '%s/%s_%s_s%%02d.seq'% (testDir, DATASET, trial)
	f= open(inputFile)
	str= ''
	for l in f:
		str += l
	exec(str, vars)
	raw= vars['raw']
	T_LIST= vars['T_LIST']
	T_SEQ= vars['T_SEQ']
	print('\n>> Algorithm %s started using following parameters:'% ALGORITHM)
	print('Dataset %s\nVerbosity %s\n'% (DATASET, VERBOSE))

	# global constants for convenience
	for t in T_LIST: T_LIST_REV[T_LIST[t]]=t
	if len(T_LIST)==1:
		TERM_P_OTHER= 0
	else:
		TERM_P_OTHER= (1.0-TERM_P) / (len(T_LIST)-1) # prob of other terminals

	rawdata= raw[ '%s_%s'% (DATASET,trial) ]

	'''""""""""""""""""""""""""""""""""""""""""""""""""""""""
	 build initial grammar from input
	""""""""""""""""""""""""""""""""""""""""""""""""""""""'''
	print(">> Let's go !")
	G= {}
	dlt_init= OrderedDict()
	dlt_init['Z']= {'terms':[],'score':-1,'count':0,'parent':None}
	for t in T_LIST:
		G= addRule(G, T_LIST[t], t)
		T_STAT[t]= {'count':0, 'prob':0}
	for rawinput in rawdata:
		input= buildInput(rawinput)
		INPUT_LIST.append(input)
		cin= ''.join(input['symbols'])
		G= addRule(G, 'Z', cin)
		updateTStat(input)
		dlt_init= inputDLT(input, dlt_init)
		dlt_init['Z']['terms'].append(conv2T(cin))
		#dlt_init[cin]['parent']= 'Z'

	# deterministic; keep count only
	if ALGORITHM=='STOLCKE':
		for input in INPUT_LIST:
			cin= ''.join(input['symbols'])
			T_STAT[conv2T(cin)]['prob']= 1.0

	printTSTAT(1, 'Terminal Symbol Statistics')
	printGrammar(1, G, dlt_init, 'After adding new input')
	dlt_init= updateDLT(G, dlt_init)
	printDLT(1, dlt_init)

	# export input data
	if EXPORT_INPUT:
		exportInput()

	# normalization factor
	normer= {}
	normer['pri']= {'mean':0, 'std':1}
	normer['pos']= {'mean':0, 'std':1}

	timeStamps= []
	timer= Timer()
	for round in range(MAX_ROUNDS):
		print('\n>> Using prior score mean= %.6f, std=%.6f'%
			(normer['pri']['mean'], normer['pri']['std']) )
		print('>> Using posterior score mean= %.6f, std=%.6f'%
			(normer['pos']['mean'], normer['pos']['std']) )

		'''""""""""""""""""""""""""""""""""""""""""""""""""""""""
		 search for the best grammar
		""""""""""""""""""""""""""""""""""""""""""""""""""""""'''
		timer.reset()
		pri= getPrior(G)
		pos= getPosterior(G, dlt_init)
		mdl= pri+pos
		gn= GNode(G, dlt_init, pri, pos, mdl, mdl, 0)
		mfile= '%s/pripos.m'%testDir
		gList, normer= learnMulti(gn, normer, mfile) # multi-process searching
		timeStamps.append(timer.read())
		assert normer['pri']['mean'] > 0
		print('\n>> Obtained prior score mean= %.6f, std=%.6f'%
			(normer['pri']['mean'], normer['pri']['std']) )
		print('>> Obtained posterior score mean= %.6f, std=%.6f'%
			(normer['pos']['mean'], normer['pos']['std']) )

		'''""""""""""""""""""""""""""""""""""""""""""""""""""""""
		 post-processing
		""""""""""""""""""""""""""""""""""""""""""""""""""""""'''
		# delete duplicates
		delList= []
		for i in range(len(gList)): # sort RHS rules to compare equalities
			gList[i].g= sortRules( gList[i].g )
		rangePr=[MAX_MDL,0] # min, max of DL(prior)
		rangePo=[MAX_MDL,0] # min, max of DL(posterior)
		for i in range(len(gList)):
			pr= gList[i].pri
			po= gList[i].pos
			if ( pr < rangePr[0] ): rangePr[0]= pr
			if ( pr > rangePr[1] ): rangePr[1]= pr
			if po < MAX_MDL:
				if ( po < rangePo[0] ): rangePo[0]= po
				if ( po > rangePo[1] ): rangePo[1]= po
			for k in range(i+1, len(gList)):
				if gList[i].g==gList[k].g:
					# THINK: SOMETIMES MDL IS DIFF FOR SAME GRAMMARS DUE TO DIFF DLT
					if gList[i].mdl != gList[k].mdl:
						pass
					delList.append(k)
		for d in sorted(list(set(delList)), reverse=True):
			del(gList[d])

		gList= sorted(gList, key=lambda gnode: gnode.mdl)
		printGrammarList(0, gList, 'MDL values:')

		'''
		if NORM_MDL:
			# normalize MDL scores and sort
			print('===== MDL RANGES=====')
			if rangePr[0] >= rangePr[1] or rangePo[0] >= rangePo[1]:
				print('>> WARNING: new_nfactor is 1. Either data is too noisy or grammar is wrong.')
				new_nfactor= 1
			else:
				new_nfactor= (rangePo[1]-rangePo[0]) / (rangePr[1]-rangePr[0])
			print('Prior: %0.3f - %0.3f, Posterior: %0.3f - %0.3f, new_normFactor: %0.3f' %
				(rangePr[0], rangePr[1], rangePo[0], rangePo[1], new_nfactor) )
			for i in range(len(gList)):
				gl= gList[i]
				gl.mdl= new_nfactor * gl.pri + gl.pos
			gList= sorted(gList, key=lambda gnode: gnode.mdl)
			printGrammarList(1, gList[:5], 'After normalizing MDL values:')
		else:
			new_nfactor= 1
		NORM_FACTOR *= new_nfactor

		# update prior normalization factor
		if abs(1.0 - new_nfactor) < 0.1:
			print('>>> normFactor has converged to %0.6f; breaking loop at round %d'%\
				(round, NORM_FACTOR) )
			break
		'''
		exportGrammarList(gList, trial)

	print('\n>> Learning finished. Showing input symbols.')
	for input in INPUT_LIST:
		print('    '.join(input['symbols']) )
		for x in input['values']:
			print('%0.2f ' % x, end='')
		print()
	print('\n>> Time consumed for each iteration.')
	for t in range(len(timeStamps)):
		print('Iteration #%d: %.1f secs'% (t, timeStamps[t]))
	print('Prune probability: %0.2f'% PRUNE_P)
	#print('Terminal confidence (for exporting grammar): %0.2f'% TERM_P)
	print('Search beam size: %d'% BEAMSIZE)

	# finish ramdisk
	os.system('ramdisk clean')
	print('>> Finished.')
	#return NORM_FACTOR

'''""""""""""""""""""""""""""""""""
 Let's go
""""""""""""""""""""""""""""""""'''
'''
run "scfgLearner.py HANOI" to use sample data in this program
to read data from files, run like "scfgLearner.py HANOI 0 0 inputFile.py Tests/"
'''
if __name__=="__main__":
	if len(sys.argv) < 3:
		print('Usage: %s {DATASET} {TRIAL} [INPUT_FILE] [TESTFILE_DIR]'% sys.argv[0])
		print('Example: %s DANCE q inputFile.py sequences'% sys.argv[0])
		sys.exit()
	else:  dataset= sys.argv[1]
	if len(sys.argv) < 3:  trial= 'q'
	else:  trial= sys.argv[2]
	if len(sys.argv) < 4:  inputFile= 'inputFile.py'
	else:  inputFile= sys.argv[3]
	if len(sys.argv) < 5:  testDir= 'Tests'
	else:  testDir= sys.argv[4]

	'''
	global NORM_INDEX
	NORM_INDEX= int(testDir[-2:])-1
	print('>> Using NORM_INDEX of %d'% NORM_INDEX)
	'''
	go(dataset, trial, inputFile, testDir)
