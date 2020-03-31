# Stochastic Context-Free Grammar Learner

This module provides the learning of Stochastic Context-Free Grammar (SCFG) structures and parameters.
It was developed to learn hierarchical representations from various human tasks.

### Prerequisites

It depends on Miguel Sarabia del Castillo's SARTParser package. You can download it from:

https://bitbucket.org/miguelsdc/sartparser/src/default/

Although not tested, it should work on a Windows machine since it's written in Python.


### Clone

```
$ git clone https://github.com/dbdq/scfglearner.git
```


### Install pip


```
$ sudo apt-get install python-pip
```


### Install Python dependency


```
$ pip install pathos
```

Depending on your Python installation, you might need root access, i.e.:


```
$ sudo pip install pathos
```


### Input files (See samples)

You need two files:

1) Terminal symbol definition

This file should contain a single line definition of terminal symbols, separated by a white space.

2) Input sequences (time x observation vector)

Each line represents an observation vector at the same time. 
Observation vector is the probability distribution of one or more symbols computed by your symbol detectors (e.g. sensors). If you have N symbol detectors over the period of T time samples, the file will contain T x N numbers.
All sequence files in the input directory will be automatically loaded.
Each file name should end with '.seq' extension.
The column order must match with the terminal order as defined in 1).


### Notations used in the code

- NJP: Normalized Joint Probability ( JP(S)^(1/len(S)) )

- V: Rule score (V= sigma(NJP(S)) * len(S))

- Data structures:
input= {'symbols':[], 'values':[]}
 Input stream with uncertainties.

- G = {'NT':[rule1, rule2, ...], ...}
 Grammar object.

- DLT(global) = OrderedDict{string:{score, count, parent, terms}}
 Description Length Table.

- GNode = Class{g, dlt, pri, pos, mdl, bestmdl, gid, worse}
 Grammar node of a search tree, gList.
 
 - bestmdl: Best MDL score observed so far in the current branch
 
 - worse: For beam search (worse += 1 if new_mdl > bestmdl)

- T_STAT = {string: {count, prob}}
Statistics of terminal symbols.

- T_LIST = {'a':'A', 'b':'B',...}
 Global terminal list.

- Concepts of Merging & Substituting:
A. Stolcke, PhD Thesis, UC Berkeley, p.93-97


### Relevant papers

* Kyuhwa Lee, Dimitri Ognibene, Hyung Jin Chang, Tae-Kyun Kim, Yiannis Demiris, "STARE: Spatio-Temporal Attention RElocation for Multiple Structured Activities Detection", IEEE Transactions on Image Processing (TIP), 2015.

* Kyuhwa Lee, Yanyu Su, Tae-Kyun Kim and Yiannis Demiris, "A Syntactic Approach to Robot Imitation Learning using Probabilistic Activity Grammars", Robotics and Autonomous Systems (RAS), Elsevier, Volume 61, Issue 12, pp.1323-1334, 2013.

* Stolcke A (1995) An Efficient Probabilistic Context-Free Parsing Algorithm that Computes Prefix Probabilities, Computational Linguistics, 21(2), pp:165-201.
