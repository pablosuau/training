#!/usr/bin/env python

import sys

record = sys.stdin.readline()
letter, value = record.split('\t')
n = 1
cma = float(value) # cumulative moving average

for line in sys.stdin:
    new_letter, value = line.split('\t')
    if new_letter == letter: 
       cma = cma + (float(value) - cma)/(float(n + 1))
       n = n + 1
    else:
       print(letter + '\t' + str(cma) + '_' + str(n))
       letter = new_letter
       n = 1
       cma = float(value)

print(letter + '\t' + str(cma) + '_' + str(n))
