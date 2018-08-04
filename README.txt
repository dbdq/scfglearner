This module provides the learning of Stochastic Context-Free Grammar (SCFG) structures and parameters.
It was developed to learn hierarchical representations from various human tasks.

**Install SARTParser library**

It depends on Miguel Sarabia del Castillo's SARTParser package. You can download it from:

http://miguelsdc.bitbucket.org/SARTParser/

Although not tested, it should work on a Windows machine since it's written in Python.


**Clone**

```
$ hg clone https://bitbucket.org/leekyuh/scfglearner scfglearner
```


**Install pip**


```
$ sudo apt-get install python-pip
```


**Install Python dependency**


```
$ pip install pathos
```

Depending on your Python installation, you might need root access, i.e.:


```
$ sudo pip install pathos
```


**Input files (See samples)**

You need two files:

1) Terminal symbol definition

This file should contain a single line definition of terminal symbols, separated by a white space.

2) Input sequences (time x observation vector)

Each line represents an observation vector at the same time. 
Observation vector is the probability distribution of one or more symbols computed by your symbol detectors (e.g. sensors). If you have N symbol detectors over the period of T time samples, the file will contain T x N numbers.
All sequence files in the input directory will be automatically loaded.
Each file name should end with '.seq' extension.
The column order must match with the terminal order as defined in 1).



**Relevant papers**

* Kyuhwa Lee, Dimitri Ognibene, Hyung Jin Chang, Tae-Kyun Kim, Yiannis Demiris, "STARE: Spatio-Temporal Attention RElocation for Multiple Structured Activities Detection", IEEE Transactions on Image Processing (TIP), 2015.

* Kyuhwa Lee, Yanyu Su, Tae-Kyun Kim and Yiannis Demiris, "A Syntactic Approach to Robot Imitation Learning using Probabilistic Activity Grammars", Robotics and Autonomous Systems (RAS), Elsevier, Volume 61, Issue 12, pp.1323-1334, 2013.

* Stolcke A (1995) An Efficient Probabilistic Context-Free Parsing Algorithm that Computes Prefix Probabilities, Computational Linguistics, 21(2), pp:165-201.


