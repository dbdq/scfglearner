from __future__ import absolute_import, division, print_function

import time

class Timer:
	def __init__(self, autoreset=False):
		self.autoreset = autoreset
		self.reset()

	def reset(self):
		self.begin = time.time()

	def read(self):
		if self.autoreset: self.reset()
		return time.time() - self.begin

	def print(self, msg='Timer reading:'):
		print( msg, '%.3f sec' % (time.time() - self.begin) )
		if self.autoreset: self.reset()
