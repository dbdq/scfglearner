# **Notations used in the code**

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

Basics of Merging & Substituting:
A. Stolcke, PhD Thesis, UCB, p.93-97